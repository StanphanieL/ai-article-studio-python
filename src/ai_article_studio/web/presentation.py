"""Serialization and template view-model helpers for article delivery."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import ValidationError

from ..anchors import markdown_anchor_sections
from ..composition import reader_facing_markdown
from ..config import Settings
from ..exports import render_markdown
from ..jobs import JobAlreadyRunning, JobRunner
from ..schemas import (
    ArticleModelConfig,
    BaseResponse,
    ImageLayout,
    ImagePlacement,
    ImageProvider,
    ImageResultOption,
    ImageVisualRole,
    OutlineItem,
    QualityReviewResult,
    TaskListStatus,
    WorkflowJobStatus,
    WorkflowRunMode,
)
from ..supervisor import decide_next_action
from ..workflow import ArticleWorkflow


def article_payload(article) -> dict:
    return {
        "id": article.id,
        "taskId": article.task_id,
        "topic": article.topic,
        "style": article.style,
        "modelConfig": article.model_config,
        "phase": article.phase,
        "status": article.status,
        "titleOptions": article.title_options,
        "selectedTitle": article.selected_title,
        "outline": article.outline,
        "content": article.content,
        "fullContent": article.full_content,
        "imagePrompts": article.image_prompts,
        "imageResults": article.image_results,
        "finalMarkdown": article.final_markdown,
        "qualityScore": article.quality_score,
        "reviewReport": article.review_report,
        "reviewRound": article.review_round,
        "supervisorAction": ArticleWorkflow.public_action(article.supervisor_action),
        "supervisorReason": article.supervisor_reason,
        "executionMode": article.execution_mode,
        "errorMessage": article.error_message,
        "createTime": article.create_time.isoformat() if article.create_time else None,
        "updateTime": article.update_time.isoformat() if article.update_time else None,
    }


def log_payload(log) -> dict:
    return {
        "id": log.id,
        "taskId": log.task_id,
        "agentName": log.agent_name,
        "inputText": log.input_text,
        "outputText": log.output_text,
        "status": log.status,
        "costMs": log.cost_ms,
        "errorMessage": log.error_message,
        "createTime": log.create_time.isoformat() if log.create_time else None,
    }


def export_response(exported) -> Response:
    fallback = "article" + Path(exported.file_name).suffix
    return Response(
        exported.content,
        media_type=exported.content_type,
        headers={"Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(exported.file_name)}"},
    )


def detail_step_url(task_id: str, step: str) -> str:
    return f"/article/detail/{task_id}?step={step}#{step}-stage"


def page_image_plan_response(request: Request, article) -> Response:
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content=BaseResponse(data=article_payload(article)).model_dump(by_alias=True))
    return RedirectResponse(detail_step_url(article.task_id, "images"), status_code=303)


async def page_json_or_form_payload(request: Request) -> dict:
    if "application/json" in request.headers.get("content-type", ""):
        try:
            payload = await request.json()
        except json.JSONDecodeError as error:
            raise ValueError("图片编排请求必须是有效 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("图片编排请求必须是对象")
        return payload
    return dict(await request.form())


async def page_reorder_payload(request: Request) -> dict:
    if "application/json" in request.headers.get("content-type", ""):
        return await page_json_or_form_payload(request)
    form = await request.form()
    raw_order = form.get("order_json")
    if raw_order is not None:
        try:
            indices = json.loads(str(raw_order))
        except json.JSONDecodeError as error:
            raise ValueError("图片排序必须是有效 JSON 数组") from error
    else:
        values = form.getlist("indices")
        if len(values) == 1 and values[0].strip().startswith("["):
            try:
                indices = json.loads(values[0])
            except json.JSONDecodeError as error:
                raise ValueError("图片排序必须是有效 JSON 数组") from error
        else:
            indices = values
    return {"indices": indices}


def validate_page_model(model_type, payload: dict):
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        detail = "；".join(str(item.get("msg", "参数不合法")) for item in error.errors())
        raise ValueError(f"图片编排参数不合法：{detail}") from error


def page_model_config(
    settings: Settings,
    *,
    text_model: str | None,
    temperature: float | None,
    max_completion_tokens: int | None,
    content_length_level: str | None,
    image_provider: str | None,
    image_count_mode: str | None,
    image_count: str,
) -> ArticleModelConfig:
    default_provider = ImageProvider.MOCK.value if settings.app_mode.value == "mock" else ImageProvider.AUTO.value
    payload = {
        "text_model": text_model or settings.openai_model,
        "temperature": 0.7 if temperature is None else temperature,
        "max_completion_tokens": 8192 if max_completion_tokens is None else max_completion_tokens,
        "content_length_level": content_length_level or "MEDIUM",
        "image_provider": image_provider or default_provider,
        "image_count_mode": image_count_mode or "AUTO",
        "image_count": image_count.strip() or None,
    }
    try:
        return ArticleModelConfig.model_validate(payload)
    except ValidationError as error:
        detail = "；".join(str(item.get("msg", "参数不合法")) for item in error.errors())
        raise ValueError(f"创作设置不合法：{detail}") from error


def start_job_response(start) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    try:
        return BaseResponse(data=start())
    except JobAlreadyRunning as error:
        return JSONResponse(status_code=409, content=BaseResponse(code=409, message=str(error)).model_dump(by_alias=True))


def wait_for_one_click_api(task_id: str, runner: JobRunner) -> BaseResponse[dict] | JSONResponse:
    try:
        runner.one_click(task_id)
    except JobAlreadyRunning as error:
        article = runner.workflow.get(task_id)
        return JSONResponse(status_code=409, content=BaseResponse(code=409, message=str(error), data=article_payload(article)).model_dump(by_alias=True))
    article = runner.wait_for_terminal(task_id)
    if article.status == "FAILED":
        return JSONResponse(
            status_code=500,
            content=BaseResponse(code=500, message=article.error_message or "一键生成失败，可使用 taskId 继续处理。", data=article_payload(article)).model_dump(
                by_alias=True
            ),
        )
    return BaseResponse(data=article_payload(article))


_PAGE_STEPS = (("title", "标题"), ("outline", "大纲"), ("content", "正文"), ("images", "配图"), ("delivery", "交付"))
_ACTION_PAGE_STEPS = {
    "GENERATE_TITLES": "title",
    "WAIT_FOR_TITLE": "title",
    "GENERATE_OUTLINE": "outline",
    "GENERATE_CONTENT": "content",
    "REGENERATE_CONTENT": "content",
    "GENERATE_IMAGE_PROMPTS": "images",
    "REGENERATE_IMAGE_PROMPTS": "images",
    "GENERATE_IMAGE_RESULTS": "images",
    "REGENERATE_IMAGE_RESULTS": "images",
    "COMPOSE_ARTICLE": "images",
    "REVIEW_ARTICLE": "delivery",
    "MANUAL_REVIEW": "delivery",
    "COMPLETE": "delivery",
}


def resolve_page_navigation(article, next_action: str | None, requested_step: str | None) -> dict[str, object]:
    action = next_action or ""
    if action == "GENERATE_OUTLINE" and getattr(article, "selected_title", None) and not getattr(article, "outline", None):
        # Confirming a title is not a navigation command. Let the author decide
        # when to launch outline generation from the title step.
        default_step = "title"
    elif getattr(article, "status", None) == "FAILED" and action in _ACTION_PAGE_STEPS:
        default_step = _ACTION_PAGE_STEPS[action]
    elif action == "COMPOSE_ARTICLE" and article.final_markdown:
        default_step = "delivery"
    elif getattr(article, "phase", None) == "OUTLINE_EDITING" and getattr(article, "execution_mode", None) != WorkflowRunMode.ONE_CLICK.value:
        default_step = "outline"
    else:
        default_step = _ACTION_PAGE_STEPS.get(action)
    if default_step is None:
        default_step = (
            "delivery"
            if article.final_markdown
            else "images"
            if article.image_prompts or article.image_results
            else "images"
            if article.content
            else "content"
            if article.outline
            else "outline"
            if article.selected_title
            else "title"
        )
    keys = [key for key, _ in _PAGE_STEPS]
    default_position = keys.index(default_step)
    current_step = requested_step if requested_step in keys and keys.index(requested_step) <= default_position else default_step
    workflow_steps = [
        {
            "key": key,
            "label": label,
            "position": position,
            "complete": position < default_position or action == "COMPLETE",
            "reachable": position <= default_position,
            "locked": position > default_position,
            "current": key == current_step,
            "workflow_current": key == default_step,
        }
        for position, (key, label) in enumerate(_PAGE_STEPS)
    ]
    return {"default_step": default_step, "current_step": current_step, "current_position": keys.index(current_step), "workflow_steps": workflow_steps}


def page_context(article, settings: Settings, requested_step: str | None = None) -> dict:
    config, phase = load_json(article.model_config, ArticleModelConfig().model_dump(by_alias=True)), article.phase
    review = parse_review_result(article.review_report)
    blocking_issues, review_issues = (review.blocking_issues, review.issues) if review else ([], [])
    review_needs_revision = bool(review and (not review.passed or blocking_issues))
    decision = decide_next_action(article, review, settings.max_auto_review_rounds)
    labels = {
        "GENERATE_TITLES": "生成标题候选",
        "GENERATE_OUTLINE": "生成大纲",
        "GENERATE_CONTENT": "生成正文",
        "GENERATE_IMAGE_PROMPTS": "生成配图方案",
        "GENERATE_IMAGE_RESULTS": "生成图片",
        "COMPOSE_ARTICLE": "合成图文稿",
        "REVIEW_ARTICLE": "执行质量评审",
        "REGENERATE_CONTENT": "重新生成正文",
        "REGENERATE_IMAGE_PROMPTS": "重新生成配图方案",
        "REGENERATE_IMAGE_RESULTS": "重新生成图片",
    }
    image_prompts, image_results = load_json(article.image_prompts, []), load_json(article.image_results, [])
    has_partial_images = bool(image_prompts) and len(image_results) < len(image_prompts)
    next_action = "GENERATE_IMAGE_RESULTS" if has_partial_images else decision.action.value
    navigation = resolve_page_navigation(
        article, ArticleWorkflow.public_action(article.supervisor_action) if article.status == "FAILED" else next_action, requested_step
    )
    outline_items = load_json(article.outline, [])
    parsed_results = parse_image_results(image_results)
    headings, paragraphs = image_anchor_options(article.content or "")
    phase_labels = {
        "CREATED": "准备选题",
        "TITLE_SELECTION": "选择标题",
        "OUTLINE_EDITING": "确认大纲",
        "CONTENT_GENERATION": "生成图文素材",
        "ARTICLE_COMPOSITION": "合成与评审",
        "COMPLETED": "文章已完成",
        "FAILED": "需要处理错误",
    }
    return {
        "article": article,
        "config": config,
        "image_choices": image_choices(settings),
        "title_options": load_json(article.title_options, []),
        "outline_markdown": outline_preview_markdown(outline_items),
        "outline_html": render_markdown(outline_preview_markdown(outline_items)),
        "image_prompts": image_prompts,
        "image_results": image_results,
        "has_partial_images": has_partial_images,
        "image_roles": list(ImageVisualRole),
        "image_placements": list(ImagePlacement),
        "image_layouts": list(ImageLayout),
        "image_anchor_headings": headings,
        "image_anchor_paragraphs": paragraphs,
        "review": review,
        "review_blocking_issues": blocking_issues,
        "review_issues": review_issues,
        "review_needs_revision": review_needs_revision,
        "can_export_final": bool(review and review.passed and not blocking_issues),
        "content_html": render_markdown(article.content or ""),
        "final_html": render_markdown(reader_facing_markdown(article.final_markdown or ""), images=parsed_results),
        "display_error": display_error_message(article.error_message),
        "phase_label": "待修订" if review_needs_revision else phase_labels.get(phase, phase),
        "next_action": next_action,
        "next_action_label": labels.get(next_action, "继续下一步"),
        "can_continue": next_action in labels
        and next_action != "GENERATE_IMAGE_RESULTS"
        and not review_needs_revision
        and article.status not in {"QUEUED", "RUNNING", "CANCELLING"},
        "can_apply_review_revision": review_needs_revision and decision.auto_revision,
        "show_one_click_utility": article.phase not in {"COMPLETED", "FAILED"}
        and not article.final_markdown
        and not (article.phase == "TITLE_SELECTION" and bool(load_json(article.title_options, []))),
        "can_retry": article.status == "FAILED" and bool(article.supervisor_action),
        "job_status_url": f"/article/{article.task_id}/jobs/status",
        "workflow_hint": workflow_hint(article, load_json(article.title_options, []), review),
        **navigation,
        "show_outline": bool(article.outline),
        "outline_editable": phase == "OUTLINE_EDITING",
        "show_content": phase in {"CONTENT_GENERATION", "ARTICLE_COMPOSITION", "COMPLETED"},
        "outline_collapsed": bool(article.outline) and phase != "OUTLINE_EDITING",
        "content_collapsed": bool(article.content) and next_action not in {"GENERATE_CONTENT", "REGENERATE_CONTENT"},
        "images_collapsed": bool(article.image_results) and next_action not in {"GENERATE_IMAGE_RESULTS", "REGENERATE_IMAGE_RESULTS"},
    }


def image_anchor_options(content: str) -> tuple[list[str], dict[str, list[int]]]:
    headings, paragraph_counts = [], {}
    for section in markdown_anchor_sections(content):
        if not section.heading or section.heading in paragraph_counts:
            continue
        headings.append(section.heading)
        paragraph_counts[section.heading] = list(range(1, len(section.paragraphs) + 1))
    return headings, paragraph_counts


def task_card_context(article) -> dict[str, object]:
    review = parse_review_result(article.review_report)
    attention = bool(review and (not review.passed or review.blocking_issues))
    if article.status == "CANCELLING":
        state, label, icon, tone = "cancelling", "正在取消并删除", "…", "attention"
    elif article.status == "QUEUED":
        state, label, icon, tone = "queued", "排队中", "◌", "running"
    elif article.status == "RUNNING":
        state, label, icon, tone = "running", "进行中", "◌", "running"
    elif article.phase == "COMPLETED" and article.status == "SUCCESS" and review and review.passed and not review.blocking_issues:
        state, label, icon, tone = "completed", "已完成", "✓", "completed"
    elif article.status == "FAILED" or attention:
        state, label, icon, tone = "attention", "需处理", "!", "attention"
    else:
        state, label, icon, tone = "waiting", "待继续", "•", "waiting"
    return {
        "article": article,
        "state": state,
        "label": label,
        "icon": icon,
        "tone": tone,
        "is_queued": article.status == "QUEUED",
        "is_running": article.status == "RUNNING",
        "is_cancelling": article.status == "CANCELLING",
        "progress": article.supervisor_reason if article.status in {"QUEUED", "RUNNING", "CANCELLING"} else None,
    }


def task_list_status(article, job: WorkflowJobStatus) -> TaskListStatus:
    """Derive display-safe live text from persisted task state."""

    card = task_card_context(article)
    progress = job.message.strip() if job.status in {"QUEUED", "RUNNING", "CANCELLING"} and job.message else None
    return TaskListStatus(
        task_id=article.task_id,
        state=str(card["state"]),
        label=str(card["label"]),
        icon=str(card["icon"]),
        tone=str(card["tone"]),
        progress=progress,
    )


def parse_review_result(raw: str | None) -> QualityReviewResult | None:
    review = load_json(raw, None)
    if not isinstance(review, dict):
        return None
    try:
        return QualityReviewResult.model_validate(review)
    except ValidationError:
        return None


def image_choices(settings: Settings) -> list[dict[str, str]]:
    if settings.app_mode.value == "mock":
        return [{"value": ImageProvider.MOCK.value, "label": "示例图片"}]
    choices = [{"value": ImageProvider.AUTO.value, "label": "自动选择"}]
    for key, provider, label in (
        (settings.pexels_api_key, ImageProvider.PEXELS, "Pexels 图库"),
        (settings.siliconflow_api_key, ImageProvider.SILICONFLOW, "SiliconFlow 生成"),
        (settings.google_ai_api_key, ImageProvider.GOOGLE_AI, "Google AI 生成"),
    ):
        if key and key.get_secret_value().strip():
            choices.append({"value": provider.value, "label": label})
    return choices


def display_error_message(value: str | None) -> str | None:
    if value and "api.siliconflow.cn" in value and "401" in value:
        return (
            "SiliconFlow 鉴权失败（401）：请检查 AI_ARTICLE_STUDIO_SILICONFLOW_API_KEY。"
            "百炼 / DashScope 的 Key 不能用于 SiliconFlow 图片接口；修改后重启服务并重试图片。"
        )
    return value


def load_json(raw: str | None, default):
    try:
        return json.loads(raw) if raw else default
    except json.JSONDecodeError:
        return default


def parse_image_results(raw: object) -> list[ImageResultOption]:
    if not isinstance(raw, list):
        return []
    parsed = []
    for item in raw:
        try:
            parsed.append(ImageResultOption.model_validate(item))
        except ValidationError:
            continue
    return parsed


def outline_preview_markdown(items: object) -> str:
    if not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        try:
            outline = OutlineItem.model_validate(item)
        except ValidationError:
            continue
        parts.append(f"## {outline.heading}\n\n{outline.description}")
    return "\n\n".join(parts)


def parse_outline_markdown(markdown: str) -> list[OutlineItem]:
    """Turn the outline editor's small Markdown dialect into workflow data."""

    if not markdown.strip():
        raise ValueError("大纲不能为空；请使用“## 小节标题”并在下方写说明")

    outline: list[OutlineItem] = []
    heading: str | None = None
    description_lines: list[str] = []

    def append_section() -> None:
        nonlocal description_lines
        if heading is None:
            return
        description = "\n".join(description_lines).strip()
        if not description:
            raise ValueError(f"“{heading}”缺少说明，请在标题下补充内容")
        outline.append(OutlineItem(heading=heading, description=description))
        description_lines = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            append_section()
            heading = stripped[3:].strip()
            if not heading:
                raise ValueError("二级标题不能为空")
        elif stripped.startswith("#"):
            raise ValueError("大纲只支持二级标题；请使用“## 小节标题”")
        elif heading is None:
            if stripped:
                raise ValueError("请先使用“## 小节标题”开始大纲")
        else:
            description_lines.append(line)

    append_section()
    if not outline:
        raise ValueError("至少需要一个“## 小节标题”")
    return outline


def workflow_hint(article, title_options: list[dict], review: QualityReviewResult | None) -> str:
    if article.phase == "CREATED":
        return "先生成 3 个标题方向，再由你确认一个。"
    if article.phase == "TITLE_SELECTION":
        return "从候选标题中选择一个，或输入自定义标题；确认后进入生成大纲步骤。" if title_options else "正在准备标题候选。"
    if article.phase == "OUTLINE_EDITING":
        return "检查大纲后生成正文，也可以直接编辑并保存。"
    if article.phase == "CONTENT_GENERATION":
        if not article.content:
            return "大纲已就绪；点击生成正文后将实时显示生成内容。"
        if not article.image_prompts:
            return "正文已生成；下一步生成配图方案。"
        if not article.image_results:
            return "配图方案已生成；请选择图片来源后生成。"
        return "图片已就绪；下一步合成图文稿。"
    if article.phase == "ARTICLE_COMPOSITION":
        return "评审未通过；你可以按建议自动修订一次，或手动调整后继续。" if review and not review.passed else "图文稿已合成；下一步执行质量评审。"
    if article.phase == "COMPLETED":
        return "图文稿已完成，可以预览和导出。"
    return "任务暂时无法继续，请查看错误信息。"


# Backward-compatible helper names for imports that predate the web package.
_detail_step_url = detail_step_url
_image_anchor_options = image_anchor_options
_image_choices = image_choices
_page_context = page_context
_page_model_config = page_model_config
_task_card_context = task_card_context

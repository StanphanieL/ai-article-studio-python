"""Server-rendered article workflow routes and no-JavaScript fallbacks."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from ..schemas import BaseResponse, ImagePlanReorderRequest, ImagePlanUpdateRequest, ImageProvider, OutlineItem, WorkflowJobStatus
from ..workflow import ArticleWorkflow
from .dependencies import JobRunnerDep, SettingsDep, TemplatesDep, WorkflowDep
from .presentation import (
    detail_step_url,
    image_choices,
    page_context,
    page_image_plan_response,
    page_json_or_form_payload,
    page_model_config,
    page_reorder_payload,
    resolve_page_navigation,
    start_job_response,
    task_card_context,
    validate_page_model,
)

router = APIRouter()


@router.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/article/list", status_code=303)


@router.get("/article/list", response_class=HTMLResponse, include_in_schema=False)
def article_list_page(request: Request, workflow: WorkflowDep, settings: SettingsDep, templates: TemplatesDep) -> HTMLResponse:
    total, articles = workflow.list(page_size=50)
    default_provider = ImageProvider.MOCK if settings.app_mode.value == "mock" else ImageProvider.AUTO
    default_config = page_model_config(
        settings,
        text_model=settings.openai_model,
        temperature=None,
        max_completion_tokens=None,
        content_length_level=None,
        image_provider=default_provider.value,
        image_count_mode=None,
        image_count="",
    ).model_dump(by_alias=True)
    return templates.TemplateResponse(
        request=request,
        name="list.html",
        context={
            "articles": articles,
            "task_cards": [task_card_context(article) for article in articles],
            "total": total,
            "default_config": default_config,
            "image_choices": image_choices(settings),
        },
    )


@router.post("/article/{task_id}/delete", include_in_schema=False)
def delete_article_page(task_id: str, request: Request, runner: JobRunnerDep) -> Response:
    deleted = runner.delete(task_id)
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(status_code=200 if deleted else 202, content=BaseResponse(data=True).model_dump(by_alias=True))
    return RedirectResponse("/article/list", status_code=303)


@router.post("/article/create", include_in_schema=False)
def create_article_page(
    topic: Annotated[str, Form()],
    request: Request,
    runner: JobRunnerDep,
    workflow: WorkflowDep,
    settings: SettingsDep,
    style: Annotated[str, Form()] = "TECH",
    mode: Annotated[str, Form()] = "step",
    text_model: Annotated[str | None, Form()] = None,
    temperature: Annotated[float | None, Form()] = None,
    max_completion_tokens: Annotated[int | None, Form()] = None,
    content_length_level: Annotated[str | None, Form()] = None,
    image_provider: Annotated[str | None, Form()] = None,
    image_count_mode: Annotated[str | None, Form()] = None,
    image_count: Annotated[str, Form()] = "",
) -> Response:
    config = page_model_config(
        settings,
        text_model=text_model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        content_length_level=content_length_level,
        image_provider=image_provider,
        image_count_mode=image_count_mode,
        image_count=image_count,
    )
    task_id = workflow.create(topic, style, config)
    if mode == "one_click":
        runner.one_click(task_id)
    detail_url = f"/article/detail/{task_id}"
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(status_code=201, content=BaseResponse(data={"taskId": task_id, "detailUrl": detail_url}).model_dump(by_alias=True))
    return RedirectResponse(detail_url, status_code=303)


@router.get("/article/detail/{task_id}", response_class=HTMLResponse, include_in_schema=False)
def article_detail_page(task_id: str, request: Request, workflow: WorkflowDep, settings: SettingsDep, templates: TemplatesDep) -> HTMLResponse:
    article = workflow.get(task_id)
    requested_step = request.query_params.get("step")
    context = page_context(article, settings, requested_step=requested_step)
    if requested_step is not None and requested_step != context["current_step"]:
        return RedirectResponse(detail_step_url(task_id, context["current_step"]), status_code=303)
    return templates.TemplateResponse(request=request, name="detail.html", context=context, headers={"Cache-Control": "no-store"})


@router.post("/article/{task_id}/continue", response_class=HTMLResponse, include_in_schema=False)
def continue_article_page(
    task_id: str, request: Request, runner: JobRunnerDep, workflow: WorkflowDep, settings: SettingsDep, templates: TemplatesDep
) -> Response:
    runner.start_next(task_id)
    article = workflow.get(task_id)
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(request=request, name="_workflow_panel.html", context=page_context(article, settings))
    action = ArticleWorkflow.public_action(article.supervisor_action) or workflow.next_decision(task_id).action.value
    return RedirectResponse(detail_step_url(task_id, resolve_page_navigation(article, action, None)["default_step"]), status_code=303)


@router.post("/article/{task_id}/one-click", include_in_schema=False)
def one_click_article_page(task_id: str, runner: JobRunnerDep, workflow: WorkflowDep) -> RedirectResponse:
    runner.one_click(task_id)
    article = workflow.get(task_id)
    action = ArticleWorkflow.public_action(article.supervisor_action) or workflow.next_decision(task_id).action.value
    return RedirectResponse(detail_step_url(task_id, resolve_page_navigation(article, action, None)["default_step"]), status_code=303)


@router.post("/article/{task_id}/review-revision", include_in_schema=False)
def review_revision_page(task_id: str, runner: JobRunnerDep) -> RedirectResponse:
    runner.apply_review_revision(task_id)
    return RedirectResponse(detail_step_url(task_id, "delivery"), status_code=303)


@router.post("/article/{task_id}/title", include_in_schema=False)
def save_title_page(task_id: str, selected_title: Annotated[str, Form()], workflow: WorkflowDep) -> RedirectResponse:
    article = workflow.confirm_title(task_id, selected_title)
    return RedirectResponse(
        detail_step_url(task_id, resolve_page_navigation(article, workflow.next_decision(task_id).action.value, None)["default_step"]), status_code=303
    )


@router.post("/article/{task_id}/outline", include_in_schema=False)
def save_outline_page(task_id: str, outline_json: Annotated[str, Form()], workflow: WorkflowDep) -> RedirectResponse:
    try:
        outline = [OutlineItem.model_validate(item) for item in json.loads(outline_json)]
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("大纲必须是有效的 JSON 数组") from error
    workflow.save_outline(task_id, outline)
    return RedirectResponse(detail_step_url(task_id, "content"), status_code=303)


@router.post("/article/{task_id}/content", include_in_schema=False)
def save_content_page(task_id: str, content: Annotated[str, Form()], workflow: WorkflowDep) -> RedirectResponse:
    workflow.save_content(task_id, content)
    return RedirectResponse(detail_step_url(task_id, "images"), status_code=303)


@router.post("/article/{task_id}/image-plan/reorder", include_in_schema=False)
async def reorder_image_plan_page(task_id: str, request: Request, workflow: WorkflowDep) -> Response:
    reorder = validate_page_model(ImagePlanReorderRequest, await page_reorder_payload(request))
    return page_image_plan_response(request, workflow.reorder_image_plan(task_id, reorder.indices))


@router.post("/article/{task_id}/image-plan/{index}", include_in_schema=False)
async def update_image_plan_page(task_id: str, index: int, request: Request, workflow: WorkflowDep) -> Response:
    update = validate_page_model(ImagePlanUpdateRequest, await page_json_or_form_payload(request))
    return page_image_plan_response(request, workflow.update_image_plan(task_id, index, update))


@router.post("/article/{task_id}/config", include_in_schema=False)
def save_config_page(
    task_id: str,
    style: Annotated[str, Form()],
    text_model: Annotated[str, Form()],
    temperature: Annotated[float, Form()],
    max_completion_tokens: Annotated[int, Form()],
    content_length_level: Annotated[str, Form()],
    image_provider: Annotated[str, Form()],
    image_count_mode: Annotated[str, Form()],
    settings: SettingsDep,
    workflow: WorkflowDep,
    image_count: Annotated[str, Form()] = "",
) -> RedirectResponse:
    config = page_model_config(
        settings,
        text_model=text_model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        content_length_level=content_length_level,
        image_provider=image_provider,
        image_count_mode=image_count_mode,
        image_count=image_count,
    )
    workflow.save_model_config(task_id, style, config)
    return RedirectResponse(detail_step_url(task_id, "title"), status_code=303)


@router.post("/article/{task_id}/image/{index}/retry", include_in_schema=False)
def retry_image_page(task_id: str, index: int, runner: JobRunnerDep, provider: Annotated[str, Form()] = "AUTO") -> RedirectResponse:
    runner.retry_image(task_id, index, ImageProvider(provider))
    return RedirectResponse(detail_step_url(task_id, "images"), status_code=303)


@router.post("/article/{task_id}/images", include_in_schema=False)
def generate_images_page(task_id: str, runner: JobRunnerDep, provider: Annotated[str, Form()] = "AUTO") -> RedirectResponse:
    runner.generate_images(task_id, ImageProvider(provider))
    return RedirectResponse(detail_step_url(task_id, "images"), status_code=303)


@router.get("/article/{task_id}/jobs/status", include_in_schema=False)
def workflow_job_status(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus]:
    return BaseResponse(data=runner.status(task_id))


@router.post("/article/{task_id}/jobs/next", include_in_schema=False, status_code=202, response_model=None)
def start_next_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.start_next(task_id))


@router.post("/article/{task_id}/jobs/retry", include_in_schema=False, status_code=202, response_model=None)
def retry_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.retry(task_id))


@router.post("/article/{task_id}/jobs/images", include_in_schema=False, status_code=202, response_model=None)
def generate_images_job(task_id: str, runner: JobRunnerDep, provider: Annotated[str, Form()] = "AUTO") -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.generate_images(task_id, ImageProvider(provider)))


@router.post("/article/{task_id}/jobs/image/{index}/retry", include_in_schema=False, status_code=202, response_model=None)
def retry_image_job(
    task_id: str, index: int, runner: JobRunnerDep, provider: Annotated[str, Form()] = "AUTO"
) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.retry_image(task_id, index, ImageProvider(provider)))


@router.post("/article/{task_id}/jobs/one-click", include_in_schema=False, status_code=202, response_model=None)
def one_click_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.one_click(task_id))


@router.post("/article/{task_id}/jobs/review-revision", include_in_schema=False, status_code=202, response_model=None)
def review_revision_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    return start_job_response(lambda: runner.apply_review_revision(task_id))

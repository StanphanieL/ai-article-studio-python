"""FastAPI application with page-ready, Java-compatible article workflow APIs."""

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from .anchors import markdown_anchor_sections
from .composition import reader_facing_markdown
from .config import Settings, get_settings
from .db import create_session_factory
from .exports import render_markdown
from .jobs import JobAlreadyRunning, JobRunner
from .llm import create_article_agent
from .providers import create_image_service
from .schemas import (
    ArticleCreateRequest,
    ArticleModelConfig,
    BaseResponse,
    ConfirmTitleRequest,
    ImageLayout,
    ImagePlacement,
    ImagePlanReorderRequest,
    ImagePlanUpdateRequest,
    ImageProvider,
    ImageResultOption,
    ImageVisualRole,
    OneClickGenerateRequest,
    OutlineItem,
    QualityReviewResult,
    SaveContentRequest,
    SaveModelConfigRequest,
    SaveOutlineRequest,
    SaveTitleRequest,
    WorkflowJobStatus,
    WorkflowRunMode,
)
from .supervisor import decide_next_action
from .vision import VisionReviewer
from .workflow import ArticleWorkflow, WorkflowBusyError, _safe_error_message


def create_app(
    database_url: str | None = None,
    upload_dir: Path | None = None,
    settings: Settings | None = None,
    job_runner_factory: Callable[[ArticleWorkflow], JobRunner] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    source_settings = settings or get_settings()
    settings = source_settings.model_copy(
        update={
            "database_url": database_url or get_settings().database_url,
            "upload_dir": upload_dir or get_settings().upload_dir,
        }
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    workflow = ArticleWorkflow(
        create_session_factory(settings.database_url),
        settings,
        create_article_agent(settings),
        create_image_service(settings),
        clock=clock,
        vision_reviewer=VisionReviewer(settings),
    )
    # Reconcile pre-quality-gate completed records before recovering RUNNING
    # jobs.  This ordering ensures an interrupted job is never rewritten by the
    # historical migration; it is only marked retryable by recovery below.
    workflow.purge_legacy_soft_deleted_records()
    workflow.purge_pending_deletions()
    workflow.reconcile_historical_quality_reviews()
    workflow.recover_interrupted_jobs()
    job_runner = job_runner_factory(workflow) if job_runner_factory else JobRunner(workflow)

    @asynccontextmanager
    async def app_lifespan(_app: FastAPI):
        yield
        job_runner.shutdown()

    app = FastAPI(title=settings.app_name, lifespan=app_lifespan)
    app.state.workflow = workflow
    app.state.job_runner = job_runner
    app.mount("/uploads/images", StaticFiles(directory=settings.upload_dir), name="uploads")
    package_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")
    templates = Jinja2Templates(directory=str(package_dir / "templates"))

    def service() -> ArticleWorkflow:
        return app.state.workflow

    def job_service() -> JobRunner:
        return app.state.job_runner

    # Keep the dependency annotation concrete.  With postponed annotations a
    # function-local alias becomes an unresolved forward reference and FastAPI
    # treats ``workflow`` as an ordinary query argument.
    WorkflowDep = Annotated[ArticleWorkflow, Depends(service)]
    JobRunnerDep = Annotated[JobRunner, Depends(job_service)]

    @app.exception_handler(ValueError)
    async def value_error_handler(_, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content=BaseResponse(code=400, message=_safe_error_message(error)).model_dump())

    @app.exception_handler(WorkflowBusyError)
    async def workflow_busy_handler(_, error: WorkflowBusyError) -> JSONResponse:
        return JSONResponse(status_code=409, content=BaseResponse(code=409, message=_safe_error_message(error)).model_dump())

    @app.exception_handler(JobAlreadyRunning)
    async def job_already_running_handler(_, error: JobAlreadyRunning) -> JSONResponse:
        return JSONResponse(status_code=409, content=BaseResponse(code=409, message=_safe_error_message(error)).model_dump())

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_, error: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=500, content=BaseResponse(code=500, message=_safe_error_message(error)).model_dump())

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=BaseResponse(code=422, message="请求参数校验失败", data=error.errors()).model_dump(),
        )

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/article/list", status_code=303)

    @app.get("/article/list", response_class=HTMLResponse, include_in_schema=False)
    def article_list_page(request: Request) -> HTMLResponse:
        total, articles = workflow.list(page_size=50)
        default_provider = ImageProvider.MOCK if settings.app_mode.value == "mock" else ImageProvider.AUTO
        default_config = ArticleModelConfig(
            text_model=settings.openai_model,
            image_provider=default_provider,
        ).model_dump(by_alias=True)
        return templates.TemplateResponse(
            request=request,
            name="list.html",
            context={
                "articles": articles,
                "task_cards": [_task_card_context(article) for article in articles],
                "total": total,
                "default_config": default_config,
                "image_choices": _image_choices(settings),
            },
        )

    @app.post("/article/{task_id}/delete", include_in_schema=False)
    def delete_article_page(task_id: str, request: Request, runner: JobRunnerDep) -> Response:
        """Cancel work when needed, then permanently remove the task history."""

        deleted = runner.delete(task_id)
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse(status_code=200 if deleted else 202, content=BaseResponse(data=True).model_dump(by_alias=True))
        return RedirectResponse("/article/list", status_code=303)

    @app.post("/article/create", include_in_schema=False)
    def create_article_page(
        topic: Annotated[str, Form()],
        request: Request,
        runner: JobRunnerDep,
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
        config = _page_model_config(
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
        # Starting on the server makes refresh, no-JavaScript clients and
        # duplicate tabs all observe the same persisted job instead of relying
        # on a hidden client-side trigger.
        if mode == "one_click":
            runner.one_click(task_id)
        detail_url = f"/article/detail/{task_id}"
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse(status_code=201, content=BaseResponse(data={"taskId": task_id, "detailUrl": detail_url}).model_dump(by_alias=True))
        return RedirectResponse(detail_url, status_code=303)

    @app.get("/article/detail/{task_id}", response_class=HTMLResponse, include_in_schema=False)
    def article_detail_page(task_id: str, request: Request) -> HTMLResponse:
        article = workflow.get(task_id)
        requested_step = request.query_params.get("step")
        context = _page_context(
            article,
            settings,
            requested_step=requested_step,
        )
        if requested_step is not None and requested_step != context["current_step"]:
            return RedirectResponse(_detail_step_url(task_id, context["current_step"]), status_code=303)
        return templates.TemplateResponse(
            request=request,
            name="detail.html",
            context=context,
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/article/{task_id}/continue", response_class=HTMLResponse, include_in_schema=False)
    def continue_article_page(task_id: str, request: Request, runner: JobRunnerDep) -> Response:
        # The non-JavaScript fallback must not turn a real model/image call
        # into a long-lived HTTP request.  It submits the same serialized job
        # as the enhanced page, then redirects back to the persisted status.
        runner.start_next(task_id)
        article = workflow.get(task_id)
        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(
                request=request,
                name="_workflow_panel.html",
                context=_page_context(article, settings),
            )
        next_action = ArticleWorkflow.public_action(article.supervisor_action) or workflow.next_decision(task_id).action.value
        step = resolve_page_navigation(article, next_action, None)["default_step"]
        return RedirectResponse(_detail_step_url(task_id, step), status_code=303)

    @app.post("/article/{task_id}/one-click", include_in_schema=False)
    def one_click_article_page(task_id: str, runner: JobRunnerDep) -> RedirectResponse:
        runner.one_click(task_id)
        article = workflow.get(task_id)
        action = ArticleWorkflow.public_action(article.supervisor_action) or workflow.next_decision(task_id).action.value
        step = resolve_page_navigation(article, action, None)["default_step"]
        return RedirectResponse(_detail_step_url(task_id, step), status_code=303)

    @app.post("/article/{task_id}/review-revision", include_in_schema=False)
    def review_revision_page(task_id: str, runner: JobRunnerDep) -> RedirectResponse:
        runner.apply_review_revision(task_id)
        return RedirectResponse(_detail_step_url(task_id, "delivery"), status_code=303)

    @app.post("/article/{task_id}/title", include_in_schema=False)
    def save_title_page(task_id: str, selected_title: Annotated[str, Form()]) -> RedirectResponse:
        article = workflow.confirm_title(task_id, selected_title)
        step = resolve_page_navigation(article, workflow.next_decision(task_id).action.value, None)["default_step"]
        return RedirectResponse(_detail_step_url(task_id, step), status_code=303)

    @app.post("/article/{task_id}/outline", include_in_schema=False)
    def save_outline_page(task_id: str, outline_json: Annotated[str, Form()]) -> RedirectResponse:
        try:
            outline = [OutlineItem.model_validate(item) for item in json.loads(outline_json)]
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("大纲必须是有效的 JSON 数组") from error
        workflow.save_outline(task_id, outline)
        return RedirectResponse(_detail_step_url(task_id, "content"), status_code=303)

    @app.post("/article/{task_id}/content", include_in_schema=False)
    def save_content_page(task_id: str, content: Annotated[str, Form()]) -> RedirectResponse:
        workflow.save_content(task_id, content)
        return RedirectResponse(_detail_step_url(task_id, "images"), status_code=303)

    @app.post("/article/{task_id}/image-plan/reorder", include_in_schema=False)
    async def reorder_image_plan_page(task_id: str, request: Request) -> Response:
        reorder = _validate_page_model(ImagePlanReorderRequest, await _page_reorder_payload(request))
        article = workflow.reorder_image_plan(task_id, reorder.indices)
        return _page_image_plan_response(request, article)

    @app.post("/article/{task_id}/image-plan/{index}", include_in_schema=False)
    async def update_image_plan_page(task_id: str, index: int, request: Request) -> Response:
        update = _validate_page_model(ImagePlanUpdateRequest, await _page_json_or_form_payload(request))
        article = workflow.update_image_plan(task_id, index, update)
        return _page_image_plan_response(request, article)

    @app.post("/article/{task_id}/config", include_in_schema=False)
    def save_config_page(
        task_id: str,
        style: Annotated[str, Form()],
        text_model: Annotated[str, Form()],
        temperature: Annotated[float, Form()],
        max_completion_tokens: Annotated[int, Form()],
        content_length_level: Annotated[str, Form()],
        image_provider: Annotated[str, Form()],
        image_count_mode: Annotated[str, Form()],
        image_count: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        config = _page_model_config(
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
        return RedirectResponse(_detail_step_url(task_id, "title"), status_code=303)

    @app.post("/article/{task_id}/image/{index}/retry", include_in_schema=False)
    def retry_image_page(
        task_id: str,
        index: int,
        runner: JobRunnerDep,
        provider: Annotated[str, Form()] = "AUTO",
    ) -> RedirectResponse:
        runner.retry_image(task_id, index, ImageProvider(provider))
        return RedirectResponse(_detail_step_url(task_id, "images"), status_code=303)

    @app.post("/article/{task_id}/images", include_in_schema=False)
    def generate_images_page(
        task_id: str,
        runner: JobRunnerDep,
        provider: Annotated[str, Form()] = "AUTO",
    ) -> RedirectResponse:
        runner.generate_images(task_id, ImageProvider(provider))
        return RedirectResponse(_detail_step_url(task_id, "images"), status_code=303)

    @app.get("/article/{task_id}/jobs/status", include_in_schema=False)
    def workflow_job_status(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus]:
        return BaseResponse(data=runner.status(task_id))

    @app.post("/article/{task_id}/jobs/next", include_in_schema=False, status_code=202, response_model=None)
    def start_next_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.start_next(task_id))

    @app.post("/article/{task_id}/jobs/retry", include_in_schema=False, status_code=202, response_model=None)
    def retry_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.retry(task_id))

    @app.post("/article/{task_id}/jobs/images", include_in_schema=False, status_code=202, response_model=None)
    def generate_images_job(
        task_id: str,
        runner: JobRunnerDep,
        provider: Annotated[str, Form()] = "AUTO",
    ) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.generate_images(task_id, ImageProvider(provider)))

    @app.post("/article/{task_id}/jobs/image/{index}/retry", include_in_schema=False, status_code=202, response_model=None)
    def retry_image_job(
        task_id: str,
        index: int,
        runner: JobRunnerDep,
        provider: Annotated[str, Form()] = "AUTO",
    ) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.retry_image(task_id, index, ImageProvider(provider)))

    @app.post("/article/{task_id}/jobs/one-click", include_in_schema=False, status_code=202, response_model=None)
    def one_click_workflow_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.one_click(task_id))

    @app.post("/article/{task_id}/jobs/review-revision", include_in_schema=False, status_code=202, response_model=None)
    def review_revision_job(task_id: str, runner: JobRunnerDep) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
        return _start_job_response(lambda: runner.apply_review_revision(task_id))

    @app.post("/api/article/create")
    def create_article(request: ArticleCreateRequest, workflow: WorkflowDep) -> BaseResponse[str]:
        return BaseResponse(data=workflow.create(request.topic, request.style, request.config))

    @app.post("/api/article/one-click-generate", response_model=None)
    def one_click(request: OneClickGenerateRequest, workflow: WorkflowDep, runner: JobRunnerDep) -> BaseResponse[dict] | JSONResponse:
        task_id = workflow.create(request.topic, request.style, request.config)
        return _wait_for_one_click_api(task_id, runner)

    @app.post("/api/article/one-click-generate/{task_id}", response_model=None)
    def one_click_existing(task_id: str, workflow: WorkflowDep, runner: JobRunnerDep) -> BaseResponse[dict] | JSONResponse:
        article = workflow.get(task_id)
        if article.phase == "COMPLETED":
            return BaseResponse(data=article_payload(article))
        return _wait_for_one_click_api(task_id, runner)

    @app.get("/api/article/{task_id}")
    def get_article(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.get(task_id)))

    @app.post("/api/article/list")
    def list_articles(request: dict, workflow: WorkflowDep) -> BaseResponse[dict]:
        page_no = max(int(request.get("pageNo", 1)), 1)
        page_size = min(max(int(request.get("pageSize", 10)), 1), 50)
        total, records = workflow.list(page_no, page_size)
        return BaseResponse(data={"total": total, "pageNo": page_no, "pageSize": page_size, "records": [article_payload(item) for item in records]})

    @app.post("/api/article/delete/{task_id}", response_model=None)
    def delete_article(task_id: str, runner: JobRunnerDep) -> BaseResponse[bool] | JSONResponse:
        deleted = runner.delete(task_id)
        if deleted:
            return BaseResponse(data=True)
        return JSONResponse(status_code=202, content=BaseResponse(data=True).model_dump(by_alias=True))

    @app.post("/api/article/generate-titles/{task_id}")
    def generate_titles(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.generate_titles(task_id)))

    @app.post("/api/article/confirm-title")
    def confirm_title(request: ConfirmTitleRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.confirm_title(request.task_id, request.selected_title)))

    @app.post("/api/article/save-title")
    def save_title(request: SaveTitleRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.save_title(request.task_id, request.selected_title)))

    @app.post("/api/article/save-outline")
    def save_outline(request: SaveOutlineRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.save_outline(request.task_id, request.outline)))

    @app.post("/api/article/generate-content/{task_id}")
    def generate_content(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.generate_content(task_id)))

    @app.get("/api/article/real-stream-generate-content/{task_id}")
    def stream_content(task_id: str, workflow: WorkflowDep) -> StreamingResponse:
        workflow.start_stream_content(task_id)
        return StreamingResponse(workflow.stream_content(task_id), media_type="text/event-stream")

    @app.post("/api/article/save-content")
    def save_content(request: SaveContentRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.save_content(request.task_id, request.content)))

    @app.post("/api/article/generate-image-prompts/{task_id}")
    def generate_image_prompts(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.generate_image_prompts(task_id)))

    @app.post("/api/article/generate-image-results/{task_id}")
    def generate_image_results(task_id: str, workflow: WorkflowDep, provider: ImageProvider = Query(ImageProvider.AUTO)) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.generate_image_results(task_id, provider)))

    @app.post("/api/article/regenerate-image-result/{task_id}")
    def regenerate_image_result(
        task_id: str,
        index: int,
        workflow: WorkflowDep,
        provider: ImageProvider = Query(ImageProvider.AUTO),
    ) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.regenerate_image_result(task_id, index, provider)))

    @app.post("/api/article/compose/{task_id}")
    def compose_article(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.compose(task_id)))

    @app.post("/api/article/continue-workflow/{task_id}")
    def continue_workflow(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.continue_workflow(task_id)))

    @app.post("/api/article/apply-review-revision/{task_id}")
    def apply_review_revision(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.apply_review_revision(task_id)))

    @app.post("/api/article/save-model-config")
    def save_model_config(request: SaveModelConfigRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
        return BaseResponse(data=article_payload(workflow.save_model_config(request.task_id, request.style, request.config)))

    @app.get("/api/article/agent-logs/{task_id}")
    def agent_logs(task_id: str, workflow: WorkflowDep) -> BaseResponse[list[dict]]:
        return BaseResponse(data=[log_payload(log) for log in workflow.logs(task_id)])

    @app.get("/api/article/export-html/{task_id}")
    def export_html(task_id: str, workflow: WorkflowDep) -> Response:
        exported = workflow.export_html(task_id)
        return _export_response(exported)

    @app.get("/api/article/export-markdown/{task_id}")
    def export_markdown(task_id: str, workflow: WorkflowDep) -> Response:
        exported = workflow.export_markdown(task_id)
        return _export_response(exported)

    @app.get("/api/article/export-zip/{task_id}")
    def export_zip(task_id: str, workflow: WorkflowDep) -> Response:
        exported = workflow.export_zip(task_id)
        return _export_response(exported)

    return app


def _export_response(exported) -> Response:
    # HTTP headers are Latin-1; use an ASCII fallback plus RFC 5987's UTF-8
    # parameter so Chinese article titles remain downloadable.
    fallback = "article" + Path(exported.file_name).suffix
    return Response(
        exported.content,
        media_type=exported.content_type,
        headers={"Content-Disposition": (f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(exported.file_name)}")},
    )


def _detail_step_url(task_id: str, step: str) -> str:
    """Return a no-JavaScript-safe detail URL anchored to its visible card."""

    return f"/article/detail/{task_id}?step={step}#{step}-stage"


def _page_image_plan_response(request: Request, article) -> Response:
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(content=BaseResponse(data=article_payload(article)).model_dump(by_alias=True))
    return RedirectResponse(_detail_step_url(article.task_id, "images"), status_code=303)


async def _page_json_or_form_payload(request: Request) -> dict:
    if "application/json" in request.headers.get("content-type", ""):
        try:
            payload = await request.json()
        except json.JSONDecodeError as error:
            raise ValueError("图片编排请求必须是有效 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("图片编排请求必须是对象")
        return payload
    form = await request.form()
    return dict(form)


async def _page_reorder_payload(request: Request) -> dict:
    if "application/json" in request.headers.get("content-type", ""):
        return await _page_json_or_form_payload(request)
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


def _validate_page_model(model_type, payload: dict):
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        detail = "；".join(str(item.get("msg", "参数不合法")) for item in error.errors())
        raise ValueError(f"图片编排参数不合法：{detail}") from error


def _page_model_config(
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
    """Build one validated per-task config for both page create and legacy save.

    Page fields remain optional so older no-JavaScript submissions inherit the
    configured environment model instead of silently falling back to a sample
    schema default.
    """

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


def _start_job_response(start: Callable[[], WorkflowJobStatus]) -> BaseResponse[WorkflowJobStatus] | JSONResponse:
    try:
        return BaseResponse(data=start())
    except JobAlreadyRunning as error:
        return JSONResponse(status_code=409, content=BaseResponse(code=409, message=str(error)).model_dump(by_alias=True))


def _wait_for_one_click_api(task_id: str, runner: JobRunner) -> BaseResponse[dict] | JSONResponse:
    """Preserve the legacy synchronous API while delegating work to JobRunner."""

    try:
        runner.one_click(task_id)
    except JobAlreadyRunning as error:
        article = runner.workflow.get(task_id)
        return JSONResponse(
            status_code=409,
            content=BaseResponse(code=409, message=str(error), data=article_payload(article)).model_dump(by_alias=True),
        )
    article = runner.wait_for_terminal(task_id)
    if article.status == "FAILED":
        return JSONResponse(
            status_code=500,
            content=BaseResponse(
                code=500,
                message=article.error_message or "一键生成失败，可使用 taskId 继续处理。",
                data=article_payload(article),
            ).model_dump(by_alias=True),
        )
    return BaseResponse(data=article_payload(article))


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


_PAGE_STEPS = (
    ("title", "标题"),
    ("outline", "大纲"),
    ("content", "正文"),
    ("images", "配图"),
    ("delivery", "交付"),
)

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
    """Resolve one safe card to display without changing persisted workflow state.

    The real workflow may have several sub-actions within a single card (image
    prompts, image generation, and composition).  This view-only resolver
    therefore treats the Supervisor's next action as authoritative for repair
    paths, then falls back to the latest artifact that is ready to inspect.
    """

    action = next_action or ""
    # Composition leaves its action on the article so a completed background
    # job can be inspected, but its persisted final draft is the authoritative
    # signal that the next card is delivery rather than the image workbench.
    if getattr(article, "status", None) == "FAILED" and action in _ACTION_PAGE_STEPS:
        default_step = _ACTION_PAGE_STEPS[action]
    elif action == "COMPOSE_ARTICLE" and article.final_markdown:
        default_step = "delivery"
    # A generated outline is intentionally a review-and-confirm checkpoint in
    # the manual flow.  One-click execution has no confirmation pause: once
    # it starts generating the body, its persisted action must be allowed to
    # advance to the content card or the client will be bounced back here on
    # every status poll.
    elif (
        getattr(article, "phase", None) == "OUTLINE_EDITING"
        and getattr(article, "execution_mode", None) != WorkflowRunMode.ONE_CLICK.value
    ):
        default_step = "outline"
    else:
        default_step = _ACTION_PAGE_STEPS.get(action)
    if default_step is None:
        if article.final_markdown:
            default_step = "delivery"
        elif article.image_prompts or article.image_results:
            default_step = "images"
        elif article.content:
            default_step = "images"
        elif article.outline:
            default_step = "content"
        elif article.selected_title:
            default_step = "outline"
        else:
            default_step = "title"

    keys = [key for key, _ in _PAGE_STEPS]
    default_position = keys.index(default_step)
    current_step = requested_step if requested_step in keys and keys.index(requested_step) <= default_position else default_step
    current_position = keys.index(current_step)
    all_complete = action == "COMPLETE"
    workflow_steps = [
        {
            "key": key,
            "label": label,
            "position": position,
            "complete": position < default_position or all_complete,
            "reachable": position <= default_position,
            "locked": position > default_position,
            "current": key == current_step,
            "workflow_current": key == default_step,
        }
        for position, (key, label) in enumerate(_PAGE_STEPS)
    ]
    return {
        "default_step": default_step,
        "current_step": current_step,
        "current_position": current_position,
        "workflow_steps": workflow_steps,
    }


def _page_context(
    article,
    settings: Settings,
    requested_step: str | None = None,
) -> dict:
    """Decode persisted JSON once so templates stay declarative and escaped."""

    config = _load_json(article.model_config, ArticleModelConfig().model_dump(by_alias=True))
    phase = article.phase
    phase_labels = {
        "CREATED": "准备选题",
        "TITLE_SELECTION": "选择标题",
        "OUTLINE_EDITING": "确认大纲",
        "CONTENT_GENERATION": "生成图文素材",
        "ARTICLE_COMPOSITION": "合成与评审",
        "COMPLETED": "文章已完成",
        "FAILED": "需要处理错误",
    }
    title_options = _load_json(article.title_options, [])
    # Historical/partially-written model review JSON must never make the whole
    # task page unavailable.  The list view already treats it as unknown; the
    # detail view must use the same tolerant decoding path.
    parsed_review = _parse_review_result(article.review_report)
    review = parsed_review
    review_blocking_issues = parsed_review.blocking_issues if parsed_review else []
    review_issues = parsed_review.issues if parsed_review else []
    review_needs_revision = bool(
        parsed_review and (not parsed_review.passed or review_blocking_issues)
    )
    decision = decide_next_action(article, parsed_review, settings.max_auto_review_rounds)
    next_action_labels = {
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
    image_prompts = _load_json(article.image_prompts, [])
    image_results = _load_json(article.image_results, [])
    has_partial_images = bool(image_prompts) and len(image_results) < len(image_prompts)
    next_action = "GENERATE_IMAGE_RESULTS" if has_partial_images else decision.action.value
    navigation_action = (
        ArticleWorkflow.public_action(article.supervisor_action)
        if article.status == "FAILED"
        else next_action
    )
    navigation = resolve_page_navigation(article, navigation_action, requested_step)
    can_continue = (
        next_action in next_action_labels
        and next_action != "GENERATE_IMAGE_RESULTS"
        and not review_needs_revision
        and article.status not in {"QUEUED", "RUNNING", "CANCELLING"}
    )
    phase_label = phase_labels.get(phase, phase)
    if review_needs_revision:
        phase_label = "待修订"
    outline_items = _load_json(article.outline, [])
    outline_markdown = _outline_preview_markdown(outline_items)
    parsed_image_results = _parse_image_results(image_results)
    image_anchor_headings, image_anchor_paragraphs = _image_anchor_options(article.content or "")
    return {
        "article": article,
        "config": config,
        "image_choices": _image_choices(settings),
        "title_options": title_options,
        "outline_text": json.dumps(outline_items, ensure_ascii=False, indent=2),
        "outline_html": render_markdown(outline_markdown),
        "image_prompts": image_prompts,
        "image_results": image_results,
        "has_partial_images": has_partial_images,
        "image_roles": list(ImageVisualRole),
        "image_placements": list(ImagePlacement),
        "image_layouts": list(ImageLayout),
        "image_anchor_headings": image_anchor_headings,
        "image_anchor_paragraphs": image_anchor_paragraphs,
        "review": review,
        "review_blocking_issues": review_blocking_issues,
        "review_issues": review_issues,
        "review_needs_revision": review_needs_revision,
        "can_export_final": bool(parsed_review and parsed_review.passed and not review_blocking_issues),
        "content_html": render_markdown(article.content or ""),
        "final_html": render_markdown(reader_facing_markdown(article.final_markdown or ""), images=parsed_image_results),
        "display_error": _display_error_message(article.error_message),
        "phase_label": phase_label,
        "next_action": next_action,
        "next_action_label": next_action_labels.get(next_action, "继续下一步"),
        "can_continue": can_continue,
        "can_apply_review_revision": review_needs_revision and decision.auto_revision,
        "show_one_click_utility": (
            article.phase not in {"COMPLETED", "FAILED"}
            and not article.final_markdown
            and not (article.phase == "TITLE_SELECTION" and bool(title_options))
        ),
        "can_retry": article.status == "FAILED" and bool(article.supervisor_action),
        "job_status_url": f"/article/{article.task_id}/jobs/status",
        "workflow_hint": _workflow_hint(article, title_options, review),
        **navigation,
        "show_outline": bool(article.outline),
        "outline_editable": phase == "OUTLINE_EDITING",
        "show_content": phase in {"CONTENT_GENERATION", "ARTICLE_COMPOSITION", "COMPLETED"},
        "outline_collapsed": bool(article.outline) and not phase == "OUTLINE_EDITING",
        "content_collapsed": bool(article.content) and next_action not in {"GENERATE_CONTENT", "REGENERATE_CONTENT"},
        "images_collapsed": bool(article.image_results) and next_action not in {"GENERATE_IMAGE_RESULTS", "REGENERATE_IMAGE_RESULTS"},
    }


def _image_anchor_options(content: str) -> tuple[list[str], dict[str, list[int]]]:
    """Expose only real body anchors to the page-side image-plan editor.

    The workflow still validates these values on every save.  This helper is a
    usability affordance so a user can choose a concrete article section and
    paragraph rather than guessing brittle Markdown offsets.
    """

    headings: list[str] = []
    paragraph_counts: dict[str, list[int]] = {}
    for section in markdown_anchor_sections(content):
        # Duplicated H2 text would make the selector ambiguous.  The workflow
        # still validates persisted anchors; show one deterministic option.
        if not section.heading or section.heading in paragraph_counts:
            continue
        headings.append(section.heading)
        paragraph_counts[section.heading] = list(range(1, len(section.paragraphs) + 1))
    return headings, paragraph_counts


def _task_card_context(article) -> dict[str, object]:
    """Translate persisted workflow state into one safe, list-card status.

    The list must never claim a task is complete solely from a legacy phase or
    score. Invalid historical review JSON is therefore treated as unknown and
    falls back to the resumable state rather than crashing the homepage.
    """

    review = _parse_review_result(article.review_report)
    review_needs_attention = bool(review and (not review.passed or review.blocking_issues))
    is_queued = article.status == "QUEUED"
    is_running = article.status == "RUNNING"
    is_cancelling = article.status == "CANCELLING"
    is_completed = bool(
        article.phase == "COMPLETED"
        and article.status == "SUCCESS"
        and review
        and review.passed
        and not review.blocking_issues
    )

    if is_cancelling:
        state, label, icon, tone = "cancelling", "正在取消并删除", "…", "attention"
    elif is_queued:
        state, label, icon, tone = "queued", "排队中", "◌", "running"
    elif is_running:
        state, label, icon, tone = "running", "进行中", "◌", "running"
    elif is_completed:
        state, label, icon, tone = "completed", "已完成", "✓", "completed"
    elif article.status == "FAILED" or review_needs_attention:
        state, label, icon, tone = "attention", "需处理", "!", "attention"
    else:
        state, label, icon, tone = "waiting", "待继续", "•", "waiting"

    return {
        "article": article,
        "state": state,
        "label": label,
        "icon": icon,
        "tone": tone,
        "is_queued": is_queued,
        "is_running": is_running,
        "is_cancelling": is_cancelling,
    }


def _parse_review_result(raw: str | None) -> QualityReviewResult | None:
    """Best-effort review decoding for older or partially-corrupt records."""

    review = _load_json(raw, None)
    if not isinstance(review, dict):
        return None
    try:
        return QualityReviewResult.model_validate(review)
    except ValidationError:
        return None


def _image_choices(settings: Settings) -> list[dict[str, str]]:
    """Expose concise customer-facing provider choices without implementation detail."""

    if settings.app_mode.value == "mock":
        return [{"value": ImageProvider.MOCK.value, "label": "示例图片"}]

    choices = [{"value": ImageProvider.AUTO.value, "label": "自动选择"}]
    if _has_secret(settings.pexels_api_key):
        choices.append({"value": ImageProvider.PEXELS.value, "label": "Pexels 图库"})
    if _has_secret(settings.siliconflow_api_key):
        choices.append({"value": ImageProvider.SILICONFLOW.value, "label": "SiliconFlow 生成"})
    if _has_secret(settings.google_ai_api_key):
        choices.append({"value": ImageProvider.GOOGLE_AI.value, "label": "Google AI 生成"})
    return choices


def _has_secret(secret) -> bool:
    return bool(secret and secret.get_secret_value().strip())


def _display_error_message(error_message: str | None) -> str | None:
    """Keep old persisted upstream errors useful after the provider message improves."""

    if error_message and "api.siliconflow.cn" in error_message and "401" in error_message:
        return (
            "SiliconFlow 鉴权失败（401）：请检查 AI_ARTICLE_STUDIO_SILICONFLOW_API_KEY。"
            "百炼 / DashScope 的 Key 不能用于 SiliconFlow 图片接口；修改后重启服务并重试图片。"
        )
    return error_message


def _load_json(raw: str | None, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _parse_image_results(raw: object) -> list[ImageResultOption]:
    """Best-effort metadata decoding keeps old/corrupt task pages readable."""

    if not isinstance(raw, list):
        return []
    parsed: list[ImageResultOption] = []
    for item in raw:
        try:
            parsed.append(ImageResultOption.model_validate(item))
        except ValidationError:
            continue
    return parsed


def _outline_preview_markdown(items: object) -> str:
    """Turn saved outline data into safe, readable Markdown for the page preview."""

    if not isinstance(items, list):
        return ""
    sections: list[str] = []
    for item in items:
        try:
            outline_item = OutlineItem.model_validate(item)
        except ValidationError:
            continue
        sections.append(f"## {outline_item.heading}\n\n{outline_item.description}")
    return "\n\n".join(sections)


def _workflow_hint(article, title_options: list[dict], review: QualityReviewResult | None) -> str:
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
        if review and not review.passed:
            return "评审未通过；你可以按建议自动修订一次，或手动调整后继续。"
        return "图文稿已合成；下一步执行质量评审。"
    if article.phase == "COMPLETED":
        return "图文稿已完成，可以预览和导出。"
    return "任务暂时无法继续，请查看错误信息。"


app = create_app()

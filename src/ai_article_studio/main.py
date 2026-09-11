"""FastAPI composition root for the AI Article Studio web application."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Settings, get_settings
from .db import create_session_factory
from .jobs import JobAlreadyRunning, JobRunner
from .llm import create_article_agent
from .providers import create_image_service
from .schemas import BaseResponse
from .vision import VisionReviewer
from .web import api, pages
from .web.presentation import _image_anchor_options, _image_choices, resolve_page_navigation  # noqa: F401
from .workflow import ArticleWorkflow, WorkflowBusyError, _safe_error_message


def create_app(
    database_url: str | None = None,
    upload_dir: Path | None = None,
    settings: Settings | None = None,
    job_runner_factory: Callable[[ArticleWorkflow], JobRunner] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    source_settings = settings or get_settings()
    resolved_settings = source_settings.model_copy(
        update={
            "database_url": database_url or get_settings().database_url,
            "upload_dir": upload_dir or get_settings().upload_dir,
        }
    )
    resolved_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    workflow = ArticleWorkflow(
        create_session_factory(resolved_settings.database_url),
        resolved_settings,
        create_article_agent(resolved_settings),
        create_image_service(resolved_settings),
        clock=clock,
        vision_reviewer=VisionReviewer(resolved_settings),
    )
    workflow.purge_legacy_soft_deleted_records()
    workflow.purge_pending_deletions()
    workflow.reconcile_historical_quality_reviews()
    workflow.recover_interrupted_jobs()
    job_runner = job_runner_factory(workflow) if job_runner_factory else JobRunner(workflow)

    @asynccontextmanager
    async def app_lifespan(_app: FastAPI):
        yield
        job_runner.shutdown()

    app = FastAPI(title=resolved_settings.app_name, lifespan=app_lifespan)
    app.state.settings = resolved_settings
    app.state.workflow = workflow
    app.state.job_runner = job_runner
    package_dir = Path(__file__).parent
    app.state.templates = Jinja2Templates(directory=str(package_dir / "templates"))
    app.mount("/uploads/images", StaticFiles(directory=resolved_settings.upload_dir), name="uploads")
    app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")
    _register_exception_handlers(app)
    app.include_router(pages.router)
    app.include_router(api.router)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
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
        return JSONResponse(status_code=422, content=BaseResponse(code=422, message="请求参数校验失败", data=error.errors()).model_dump())


app = create_app()

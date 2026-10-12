"""Machine-facing article workflow API and SSE routes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from ..schemas import (
    ArticleCreateRequest,
    BaseResponse,
    ConfirmTitleRequest,
    ImageProvider,
    OneClickGenerateRequest,
    SaveContentRequest,
    SaveModelConfigRequest,
    SaveOutlineRequest,
    SaveTitleRequest,
)
from .dependencies import JobRunnerDep, WorkflowDep
from .presentation import article_payload, export_response, log_payload, wait_for_one_click_api
from .streaming import ClosingStreamingResponse

router = APIRouter()


@router.post("/api/article/create")
def create_article(request: ArticleCreateRequest, workflow: WorkflowDep) -> BaseResponse[str]:
    return BaseResponse(data=workflow.create(request.topic, request.style, request.config))


@router.post("/api/article/one-click-generate", response_model=None)
def one_click(request: OneClickGenerateRequest, workflow: WorkflowDep, runner: JobRunnerDep) -> BaseResponse[dict] | JSONResponse:
    return wait_for_one_click_api(workflow.create(request.topic, request.style, request.config), runner)


@router.post("/api/article/one-click-generate/{task_id}", response_model=None)
def one_click_existing(task_id: str, workflow: WorkflowDep, runner: JobRunnerDep) -> BaseResponse[dict] | JSONResponse:
    article = workflow.get(task_id)
    return BaseResponse(data=article_payload(article)) if article.phase == "COMPLETED" else wait_for_one_click_api(task_id, runner)


@router.get("/api/article/{task_id}")
def get_article(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.get(task_id)))


@router.post("/api/article/list")
def list_articles(request: dict, workflow: WorkflowDep) -> BaseResponse[dict]:
    page_no, page_size = max(int(request.get("pageNo", 1)), 1), min(max(int(request.get("pageSize", 10)), 1), 50)
    total, records = workflow.list(page_no, page_size)
    return BaseResponse(data={"total": total, "pageNo": page_no, "pageSize": page_size, "records": [article_payload(item) for item in records]})


@router.post("/api/article/delete/{task_id}", response_model=None)
def delete_article(task_id: str, runner: JobRunnerDep) -> BaseResponse[bool] | JSONResponse:
    if runner.delete(task_id):
        return BaseResponse(data=True)
    return JSONResponse(status_code=202, content=BaseResponse(data=True).model_dump(by_alias=True))


@router.post("/api/article/generate-titles/{task_id}")
def generate_titles(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.generate_titles(task_id)))


@router.post("/api/article/confirm-title")
def confirm_title(request: ConfirmTitleRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.confirm_title(request.task_id, request.selected_title)))


@router.post("/api/article/save-title")
def save_title(request: SaveTitleRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.save_title(request.task_id, request.selected_title)))


@router.post("/api/article/save-outline")
def save_outline(request: SaveOutlineRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.save_outline(request.task_id, request.outline)))


@router.post("/api/article/generate-content/{task_id}")
def generate_content(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.generate_content(task_id)))


@router.get("/api/article/real-stream-generate-content/{task_id}")
def stream_content(task_id: str, workflow: WorkflowDep) -> ClosingStreamingResponse:
    workflow.start_stream_content(task_id)
    return ClosingStreamingResponse(
        workflow.stream_content(task_id),
        on_unstarted_close=lambda: workflow.interrupt_stream_content(task_id),
        media_type="text/event-stream",
    )


@router.post("/api/article/save-content")
def save_content(request: SaveContentRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.save_content(request.task_id, request.content)))


@router.post("/api/article/generate-image-prompts/{task_id}")
def generate_image_prompts(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.generate_image_prompts(task_id)))


@router.post("/api/article/generate-image-results/{task_id}")
def generate_image_results(task_id: str, workflow: WorkflowDep, provider: ImageProvider = Query(ImageProvider.AUTO)) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.generate_image_results(task_id, provider)))


@router.post("/api/article/regenerate-image-result/{task_id}")
def regenerate_image_result(task_id: str, index: int, workflow: WorkflowDep, provider: ImageProvider = Query(ImageProvider.AUTO)) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.regenerate_image_result(task_id, index, provider)))


@router.post("/api/article/compose/{task_id}")
def compose_article(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.compose(task_id)))


@router.post("/api/article/continue-workflow/{task_id}")
def continue_workflow(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.continue_workflow(task_id)))


@router.post("/api/article/apply-review-revision/{task_id}")
def apply_review_revision(task_id: str, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.apply_review_revision(task_id)))


@router.post("/api/article/save-model-config")
def save_model_config(request: SaveModelConfigRequest, workflow: WorkflowDep) -> BaseResponse[dict]:
    return BaseResponse(data=article_payload(workflow.save_model_config(request.task_id, request.style, request.config)))


@router.get("/api/article/agent-logs/{task_id}")
def agent_logs(task_id: str, workflow: WorkflowDep) -> BaseResponse[list[dict]]:
    return BaseResponse(data=[log_payload(log) for log in workflow.logs(task_id)])


@router.get("/api/article/export-html/{task_id}")
def export_html(task_id: str, workflow: WorkflowDep) -> Response:
    return export_response(workflow.export_html(task_id))


@router.get("/api/article/export-markdown/{task_id}")
def export_markdown(task_id: str, workflow: WorkflowDep) -> Response:
    return export_response(workflow.export_markdown(task_id))


@router.get("/api/article/export-zip/{task_id}")
def export_zip(task_id: str, workflow: WorkflowDep) -> Response:
    return export_response(workflow.export_zip(task_id))

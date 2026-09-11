"""Persistent application service for the deterministic article workflow."""

from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from .anchors import heading_key, image_anchor_excerpt, markdown_anchor_sections
from .composition import compose_article
from .config import Settings
from .contracts import ArticleAgent
from .domain import (
    WorkflowBusyError,
    WorkflowCancellationRequested,
    WorkflowExecutionError,
    WorkflowOperation,
    WorkflowValidationError,
    redact_sensitive_text,
)
from .exports import ArticleExporter, ExportFile
from .models import AgentLog, Article
from .providers import ImageGenerationService, calculate_image_count
from .quality import content_contract_issues, factuality_contract_issues, is_quality_gate_issue
from .schemas import (
    ArticleModelConfig,
    ImageCountMode,
    ImagePlacement,
    ImagePromptOption,
    ImageProvider,
    ImageResultOption,
    ImageVisualReview,
    ImageVisualReviewStatus,
    ImageVisualRole,
    OutlineItem,
    QualityReviewResult,
    TitleOption,
    WorkflowJobStatus,
    WorkflowRunMode,
)
from .supervisor import ArticlePhase, ArticleStatus, SupervisorAction, SupervisorDecision, decide_next_action
from .vision import VisionReviewer

SessionFactory = sessionmaker[Session]

_ACTION_PROVIDER_SEPARATOR = "|provider="
_IMAGE_ACTIONS = {
    SupervisorAction.GENERATE_IMAGE_RESULTS.value,
    SupervisorAction.REGENERATE_IMAGE_RESULTS.value,
    "REGENERATE_IMAGE_RESULT",
}
_IMAGE_PLAN_VISUAL_FIELDS = (
    "image_title",
    "usage_scene",
    "prompt_zh",
    "prompt_en",
    "visual_role",
    "stock_query",
)
_GENERATED_ROLE_FALLBACK_ORDER = (
    ImageVisualRole.CONCEPT,
    ImageVisualRole.DIAGRAM,
    ImageVisualRole.SCENE,
    ImageVisualRole.COMPARISON,
    ImageVisualRole.DETAIL,
    ImageVisualRole.DATA,
    ImageVisualRole.CLOSING,
)
class ArticleWorkflow:
    def __init__(
        self,
        sessions: SessionFactory,
        settings: Settings,
        agent: ArticleAgent,
        images: ImageGenerationService,
        clock: Callable[[], datetime] | None = None,
        vision_reviewer: VisionReviewer | None = None,
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.agent = agent
        self.images = images
        self.exporter = ArticleExporter(settings.upload_dir)
        self.vision_reviewer = vision_reviewer or VisionReviewer(settings)
        self._clock = clock or datetime.now
        self._background_job_execution: ContextVar[bool] = ContextVar("background_job_execution", default=False)
        self._task_locks: dict[str, RLock] = {}
        self._task_locks_guard = Lock()

    @contextmanager
    def background_job_execution(self) -> Iterator[None]:
        token = self._background_job_execution.set(True)
        try:
            yield
        finally:
            self._background_job_execution.reset(token)

    @contextmanager
    def task_write_lock(self, task_id: str) -> Iterator[None]:
        """Serialize mutations for one task while allowing unrelated tasks to proceed."""

        lock = self._lock_for_task(task_id)
        if not lock.acquire(blocking=self._background_job_execution.get()):
            raise WorkflowBusyError("任务正在后台执行中，请等待完成后再编辑。")
        try:
            yield
        finally:
            lock.release()

    def create(self, topic: str, style: str = "TECH", config: ArticleModelConfig | None = None) -> str:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise WorkflowValidationError("选题不能为空")
        model_config = config or ArticleModelConfig(text_model=self.settings.openai_model)
        with self.sessions() as session:
            task_id = uuid.uuid4().hex
            session.add(
                Article(
                    task_id=task_id,
                    topic=normalized_topic,
                    style=style or "TECH",
                    model_config=model_config.model_dump_json(by_alias=True),
                    phase=ArticlePhase.CREATED.value,
                    status=ArticleStatus.INIT.value,
                )
            )
            session.commit()
            return task_id

    def get(self, task_id: str) -> Article:
        with self.sessions() as session:
            article = self._get_article(session, task_id)
            session.expunge(article)
            return article

    def list(self, page_no: int = 1, page_size: int = 10) -> tuple[int, list[Article]]:
        with self.sessions() as session:
            records = list(
                session.scalars(
                    select(Article).where(Article.is_deleted == 0).order_by(Article.create_time.desc()).offset(max(page_no - 1, 0) * page_size).limit(page_size)
                )
            )
            total = len(list(session.scalars(select(Article.id).where(Article.is_deleted == 0))))
            for record in records:
                session.expunge(record)
            return total, records

    def permanently_delete(self, task_id: str, *, include_legacy_soft_deleted: bool = False) -> bool:
        """Physically remove one article, its logs, and its task-owned local assets.

        Callers must first stop or mark any active execution. The per-task lock
        prevents a worker from committing a result after the records are gone.
        """

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = (
                    session.scalar(select(Article).where(Article.task_id == task_id))
                    if include_legacy_soft_deleted
                    else self._get_article(session, task_id)
                )
                if not article:
                    raise WorkflowValidationError("文章任务不存在")
                local_paths = self._local_image_paths(article.image_results)
                referenced_paths = self._referenced_local_image_paths(session, excluding_task_id=task_id)
                session.execute(delete(AgentLog).where(AgentLog.task_id == task_id))
                session.delete(article)
                session.commit()
            self._delete_unreferenced_local_assets(task_id, local_paths, referenced_paths)
            return True

    def request_cancellation(self, task_id: str) -> None:
        """Persist cancellation without waiting on a busy worker lock."""

        with self.sessions() as session:
            article = self._get_article(session, task_id)
            article.status = ArticleStatus.CANCELLING.value
            article.supervisor_reason = "正在取消后台任务并永久删除"
            article.error_message = None
            article.update_time = self._clock()
            session.commit()

    def cancellation_requested(self, task_id: str) -> bool:
        with self.sessions() as session:
            status = session.scalar(select(Article.status).where(Article.task_id == task_id, Article.is_deleted == 0))
            return status is None or status == ArticleStatus.CANCELLING.value

    def purge_pending_deletions(self) -> int:
        """Finish an interrupted cancellation after a local service restart."""

        with self.sessions() as session:
            task_ids = list(session.scalars(select(Article.task_id).where(Article.status == ArticleStatus.CANCELLING.value)))
        for task_id in task_ids:
            self.permanently_delete(task_id)
        return len(task_ids)

    def purge_legacy_soft_deleted_records(self) -> int:
        """Migrate prior soft deletions to the current irreversible-delete policy."""

        with self.sessions() as session:
            task_ids = list(session.scalars(select(Article.task_id).where(Article.is_deleted == 1)))
        for task_id in task_ids:
            self.permanently_delete(task_id, include_legacy_soft_deleted=True)
        return len(task_ids)

    def generate_titles(self, task_id: str) -> Article:
        return self._run(task_id, "TitleGeneratorAgent", lambda article: self._generate_titles(article))

    def confirm_title(self, task_id: str, selected_title: str) -> Article:
        normalized_title = selected_title.strip()
        if not normalized_title:
            raise WorkflowValidationError("请选择或填写标题")
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                if article.selected_title == normalized_title:
                    # Re-submitting the already confirmed title is a genuine
                    # no-op.  In particular, do not regress a completed task
                    # to outline editing while retaining its downstream work.
                    session.expunge(article)
                    return article
                title_changed = article.selected_title is not None and article.selected_title != normalized_title
                if title_changed:
                    for field in (
                        "outline",
                        "content",
                        "full_content",
                        "image_prompts",
                        "image_results",
                        "final_markdown",
                        "review_report",
                        "quality_score",
                    ):
                        setattr(article, field, None)
                    article.review_round = 0
                    article.error_message = None
                article.selected_title = normalized_title
                article.phase = ArticlePhase.OUTLINE_EDITING.value
                # One-click mode confirms its recommended title internally.
                # That checkpoint is durable, but the parent background job is
                # still running and must not look terminal to the page.
                article.status = (
                    ArticleStatus.RUNNING.value
                    if self._background_job_execution.get()
                    else ArticleStatus.SUCCESS.value
                )
                article.supervisor_action = SupervisorAction.GENERATE_OUTLINE.value
                article.supervisor_reason = "标题已确认，下一步生成文章大纲"
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def save_title(self, task_id: str, selected_title: str) -> Article:
        return self._update(task_id, selected_title=selected_title.strip())

    def generate_outline(self, task_id: str) -> Article:
        return self._run(task_id, "OutlineGeneratorAgent", lambda article: self._generate_outline(article))

    def save_outline(self, task_id: str, outline: list[OutlineItem]) -> Article:
        return self._update(
            task_id,
            outline=json.dumps([item.model_dump(by_alias=True) for item in outline], ensure_ascii=False),
            content=None,
            full_content=None,
            image_prompts=None,
            image_results=None,
            final_markdown=None,
            review_report=None,
            quality_score=None,
            phase=ArticlePhase.CONTENT_GENERATION.value,
            status=ArticleStatus.SUCCESS.value,
        )

    def generate_content(self, task_id: str) -> Article:
        return self._run(task_id, "ContentGeneratorAgent", lambda article: self._generate_content(article))

    def start_stream_content(self, task_id: str) -> Article:
        """Atomically reserve the task for an SSE content generation lifecycle."""

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                if not article.selected_title or not self._outline(article):
                    raise WorkflowValidationError("请先确认标题并生成大纲")
                article.status = ArticleStatus.RUNNING.value
                article.supervisor_action = SupervisorAction.GENERATE_CONTENT.value
                article.supervisor_reason = "正在生成正文"
                article.error_message = None
                article.update_time = self._clock()
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def stream_content(self, task_id: str) -> Iterator[str]:
        cancelled = False
        interrupted = False
        completed = False
        try:
            with self.task_write_lock(task_id):
                with self.sessions() as session:
                    article = self._get_article(session, task_id)
                    outline = self._outline(article)
                    if article.status != ArticleStatus.RUNNING.value:
                        raise WorkflowValidationError("请先启动流式正文生成")
                    if not article.selected_title or not outline:
                        raise WorkflowValidationError("请先确认标题并生成大纲")
                    config = self._config(article)
                    agent = self._agent_for(article)
                    chunks: list[str] = []
                    started = time.perf_counter()
                    yield _sse("progress", {"message": "正在生成正文"})
                    try:
                        for chunk in agent.stream_content(
                            article.topic,
                            article.selected_title,
                            outline,
                            article.style,
                            config.content_length_level.value,
                        ):
                            if self.cancellation_requested(task_id):
                                raise WorkflowCancellationRequested("任务已请求取消")
                            chunks.append(chunk)
                            yield _sse("content", {"content": chunk})
                        if self.cancellation_requested(task_id):
                            raise WorkflowCancellationRequested("任务已请求取消")
                        candidate = "".join(chunks).strip()
                        article.content, repaired = self._validated_generated_content(article, candidate, agent)
                        if repaired:
                            yield _sse("progress", {"message": "正文通过结构与事实检查后已自动修订"})
                            yield _sse("content_replace", {"content": article.content})
                        article.phase = ArticlePhase.CONTENT_GENERATION.value
                        article.status = ArticleStatus.SUCCESS.value
                        article.image_prompts = None
                        article.image_results = None
                        article.final_markdown = None
                        article.review_report = None
                        article.quality_score = None
                        article.error_message = None
                        self._log(session, task_id, "ContentGeneratorAgent-RealStream", "stream", article.content, started)
                        session.commit()
                        # A completed result is already durable before the
                        # final SSE event is written.  If the browser closes
                        # immediately after this yield, do not regress it to a
                        # retryable failure.
                        completed = True
                        yield _sse("done", {"phase": article.phase, "status": article.status})
                    except WorkflowCancellationRequested:
                        cancelled = True
                        session.rollback()
                    except GeneratorExit:
                        # EventSource is tied to the currently visible card.
                        # Navigating away closes this generator; leaving the
                        # DB row RUNNING would permanently block the task.
                        interrupted = True
                        session.rollback()
                        raise
                    except Exception as error:
                        if self.cancellation_requested(task_id):
                            cancelled = True
                            session.rollback()
                        else:
                            article.status = ArticleStatus.FAILED.value
                            article.error_message = _safe_error_message(error)
                            self._log(session, task_id, "ContentGeneratorAgent-RealStream", "stream", None, started, error)
                            session.commit()
                            yield _sse("fail", {"message": "正文流式生成失败"})
        finally:
            if cancelled or self.cancellation_requested(task_id):
                try:
                    self.permanently_delete(task_id)
                except ValueError:
                    pass
            elif interrupted and not completed:
                self._mark_interrupted_stream_retryable(task_id)
        if cancelled:
            yield _sse("fail", {"message": "任务已取消并永久删除"})

    def save_content(self, task_id: str, content: str) -> Article:
        normalized_content = content.strip()
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                self._raise_for_content_contract(article, normalized_content)
                self._set_content(article, normalized_content)
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def generate_image_prompts(self, task_id: str) -> Article:
        return self._run(task_id, "ImagePromptAgent", lambda article: self._generate_image_prompts(article))

    def generate_image_results(self, task_id: str, provider: ImageProvider) -> Article:
        self._generate_images_incrementally(task_id, provider)
        return self._run(task_id, "ImageVisualReviewAgent", self._review_generated_images)

    def regenerate_image_result(self, task_id: str, index: int, provider: ImageProvider) -> Article:
        generated = self._run(
            task_id,
            "ImageResultAgent-Single",
            lambda article: self._regenerate_image_result(article, index, provider),
        )
        if len(self._images(generated)) < len(self._prompts(generated)):
            return generated
        return self._run(
            task_id,
            "ImageVisualReviewAgent",
            lambda article: self._review_generated_images(article, indices=[index]),
        )

    def update_image_plan(self, task_id: str, index: int, updated_plan: ImagePromptOption) -> Article:
        """Persist a user-arranged image plan without starting a background job.

        Image appearance changes invalidate generated assets; pure placement and
        layout changes keep the asset and update its placement metadata in lock
        step with its prompt record.
        """

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                prompts = self._prompts(article)
                if index < 0 or index >= len(prompts):
                    raise WorkflowValidationError("图片编排索引不存在")
                previous_plan = prompts[index]
                # The anchor excerpt is internal evidence, not an editable
                # page field. Recompute it from the stored article whenever a
                # user changes an anchor or placement.
                updated_plan = self._with_anchor_excerpt(article.content or "", updated_plan)
                visual_change = any(
                    getattr(previous_plan, field) != getattr(updated_plan, field)
                    for field in _IMAGE_PLAN_VISUAL_FIELDS
                )
                prompts[index] = updated_plan
                self._validate_image_plan_contract(article.content or "", prompts)
                article.image_prompts = self._dump_image_records(prompts)

                results = self._images(article)
                if results and len(results) != len(prompts):
                    raise WorkflowValidationError("图片结果与配图方案数量不一致，请重新生成图片后再编辑。")
                if visual_change:
                    article.image_results = None
                    next_action = SupervisorAction.GENERATE_IMAGE_RESULTS.value
                    reason = "配图视觉方案已修改，请重新生成图片素材"
                    output = "已保存视觉方案；已清空图片素材，需重新生成"
                else:
                    if results:
                        results[index] = results[index].model_copy(update=updated_plan.model_dump())
                        article.image_results = self._dump_image_records(results)
                    next_action = (
                        SupervisorAction.COMPOSE_ARTICLE.value if results else SupervisorAction.GENERATE_IMAGE_RESULTS.value
                    )
                    reason = "配图位置已更新，请重新合成图文稿" if results else "配图方案已更新，请生成图片素材"
                    output = "已保存配图位置与版式；图片素材已保留" if results else "已保存配图位置与版式；请生成图片素材"

                article.final_markdown = None
                article.review_report = article.quality_score = None
                article.phase = ArticlePhase.CONTENT_GENERATION.value
                article.status = ArticleStatus.SUCCESS.value
                article.supervisor_action = next_action
                article.supervisor_reason = reason
                article.error_message = None
                started = time.perf_counter()
                self._log(session, task_id, "ImagePlanEditorAgent", f"image_plan_index={index}; visual_change={visual_change}", output, started)
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def reorder_image_plan(self, task_id: str, indices: list[int]) -> Article:
        """Move image-plan/result records as inseparable pairs and re-compose later."""

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                prompts = self._prompts(article)
                expected_indices = set(range(len(prompts)))
                if len(indices) != len(prompts) or set(indices) != expected_indices:
                    raise WorkflowValidationError("图片排序必须包含每一张配图且不得重复")
                results = self._images(article)
                if results and len(results) != len(prompts):
                    raise WorkflowValidationError("图片结果与配图方案数量不一致，请重新生成图片后再排序。")

                reordered_prompts = [prompts[index] for index in indices]
                self._validate_image_plan_contract(article.content or "", reordered_prompts)
                article.image_prompts = self._dump_image_records(reordered_prompts)
                if results:
                    article.image_results = self._dump_image_records([results[index] for index in indices])
                article.final_markdown = None
                article.review_report = article.quality_score = None
                article.phase = ArticlePhase.CONTENT_GENERATION.value
                article.status = ArticleStatus.SUCCESS.value
                article.supervisor_action = (
                    SupervisorAction.COMPOSE_ARTICLE.value if results else SupervisorAction.GENERATE_IMAGE_RESULTS.value
                )
                article.supervisor_reason = "配图顺序已更新，请重新合成图文稿"
                article.error_message = None
                started = time.perf_counter()
                self._log(session, task_id, "ImagePlanEditorAgent", "image_plan_reorder", "已保存配图顺序", started)
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def compose(self, task_id: str) -> Article:
        return self._run(task_id, "ArticleComposeAgent", lambda article: self._compose(article))

    def review(self, task_id: str) -> Article:
        return self._run(task_id, "QualityReviewAgent", lambda article: self._review(article))

    def continue_workflow(self, task_id: str) -> Article:
        with self.task_write_lock(task_id):
            self.ensure_not_busy(task_id)
            decision = self.next_decision(task_id)
            if decision.auto_revision:
                return self._apply_review_revision(task_id, decision)
            return self.execute_action(task_id, decision.action.value)

    def next_decision(self, task_id: str):
        article = self.get(task_id)
        if article.status not in {ArticleStatus.FAILED.value, ArticleStatus.CANCELLING.value} and article.image_prompts:
            prompts = self._prompts(article)
            results = self._images(article)
            if len(results) < len(prompts):
                return SupervisorDecision(
                    action=SupervisorAction.GENERATE_IMAGE_RESULTS,
                    reason=f"已生成 {len(results)}/{len(prompts)} 张图片，下一步继续生成剩余图片",
                    review_round=max(article.review_round or 0, 0),
                )
        return decide_next_action(article, self._review_result(article), self.settings.max_auto_review_rounds)

    def execute_action(self, task_id: str, action: str, provider: ImageProvider = ImageProvider.AUTO) -> Article:
        """Run a persisted supervisor action, including a failed job's retry action."""

        persisted_operation = WorkflowOperation.parse(action)
        persisted_action = persisted_operation.encode()
        provider = self.provider_for_action(persisted_action, provider)
        action = persisted_operation.action
        index: int | None = None
        raw_action, separator, raw_index = action.partition(":")
        if raw_action == "REGENERATE_IMAGE_RESULT":
            if not separator or not raw_index.isdigit():
                raise WorkflowValidationError("图片重试动作不合法")
            index = int(raw_index)
        if raw_action == "ONE_CLICK_GENERATE":
            return self.one_click(task_id)
        if raw_action == "APPLY_REVIEW_REVISION":
            return self.apply_review_revision(task_id)
        if raw_action == "REGENERATE_IMAGE_RESULT":
            assert index is not None
            return self.regenerate_image_result(task_id, index, provider)
        try:
            supervisor_action = SupervisorAction(raw_action)
        except ValueError as error:
            raise WorkflowValidationError("当前任务动作不支持执行") from error
        self._mark_current_action(task_id, persisted_action)
        if supervisor_action is SupervisorAction.GENERATE_TITLES:
            return self.generate_titles(task_id)
        if supervisor_action is SupervisorAction.GENERATE_OUTLINE:
            return self.generate_outline(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_CONTENT, SupervisorAction.REGENERATE_CONTENT}:
            return self.generate_content(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_IMAGE_PROMPTS, SupervisorAction.REGENERATE_IMAGE_PROMPTS}:
            return self.generate_image_prompts(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_IMAGE_RESULTS, SupervisorAction.REGENERATE_IMAGE_RESULTS}:
            return self.generate_image_results(task_id, provider)
        if supervisor_action is SupervisorAction.COMPOSE_ARTICLE:
            return self.compose(task_id)
        if supervisor_action is SupervisorAction.REVIEW_ARTICLE:
            return self.review(task_id)
        if supervisor_action in {
            SupervisorAction.WAIT_FOR_TITLE,
            SupervisorAction.MANUAL_REVIEW,
            SupervisorAction.STOP_FAILED,
            SupervisorAction.COMPLETE,
        }:
            decision = self.next_decision(task_id)
            return self._record_decision(task_id, decision.action.value, decision.reason)
        raise WorkflowValidationError("当前任务动作不支持执行")

    def begin_job(
        self,
        task_id: str,
        action: str,
        message: str,
        run_mode: WorkflowRunMode = WorkflowRunMode.STEP,
    ) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                if article.status in {ArticleStatus.QUEUED.value, ArticleStatus.RUNNING.value}:
                    raise WorkflowExecutionError("任务正在执行中，请等待完成后再试")
                article.status = ArticleStatus.QUEUED.value
                article.supervisor_action = action
                article.supervisor_reason = message
                article.execution_mode = run_mode.value
                article.error_message = None
                article.update_time = self._clock()
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def mark_job_running(self, task_id: str) -> Article:
        """Transition an accepted local job from queueing to active execution."""

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                if article.status == ArticleStatus.QUEUED.value:
                    article.status = ArticleStatus.RUNNING.value
                    article.update_time = self._clock()
                    session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def finish_job(self, task_id: str) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                if article.status == ArticleStatus.RUNNING.value:
                    article.status = ArticleStatus.SUCCESS.value
                    article.error_message = None
                    article.execution_mode = None
                    session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def fail_job(self, task_id: str, error: Exception) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                article.status = ArticleStatus.FAILED.value
                article.error_message = _safe_error_message(error)
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def recover_interrupted_jobs(self) -> int:
        """Make interrupted local work retryable after a process restart."""

        with self.sessions() as session:
            articles = list(
                session.scalars(
                    select(Article).where(Article.status.in_([ArticleStatus.QUEUED.value, ArticleStatus.RUNNING.value]))
                )
            )
            for article in articles:
                article.status = ArticleStatus.FAILED.value
                article.error_message = "任务在服务重启时中断，请重试此步骤。"
            session.commit()
            return len(articles)

    def reconcile_historical_quality_reviews(self) -> int:
        """Reconcile persisted deterministic Markdown checks after a gate update.

        Older databases may predate a gate, and a bug fix may also invalidate a
        previously persisted automated finding.  Only deterministic gate messages
        are replaced; model-authored review feedback is left untouched.  A job
        still marked RUNNING is deliberately left alone so startup reconciliation
        cannot race or overwrite live work.
        """

        with self.sessions() as session:
            articles = list(
                session.scalars(
                    select(Article).where(
                        Article.is_deleted == 0,
                        Article.final_markdown.is_not(None),
                        Article.review_report.is_not(None),
                        Article.status != ArticleStatus.RUNNING.value,
                    )
                )
            )
            corrected = 0
            for article in articles:
                try:
                    review = QualityReviewResult.model_validate_json(article.review_report or "")
                except ValueError:
                    # A malformed legacy payload must never prevent startup or
                    # discard a user's existing article.
                    continue

                markdown_issues = content_contract_issues(
                    article.final_markdown or "",
                    topic=article.topic,
                    title=article.selected_title or article.topic,
                    style=article.style,
                )
                model_blocking_issues = [issue for issue in review.blocking_issues if not is_quality_gate_issue(issue)]
                blocking_issues = list(dict.fromkeys([*model_blocking_issues, *markdown_issues]))
                if blocking_issues == review.blocking_issues:
                    continue

                reconciled_review = review.model_copy(
                    update={
                        "blocking_issues": blocking_issues,
                        **(
                            {
                                "passed": False,
                                "revision_action": review.revision_action or SupervisorAction.REGENERATE_CONTENT.value,
                                "revision_reason": review.revision_reason or "存在阻断交付的问题，需要修订正文。",
                            }
                            if markdown_issues
                            else {}
                        ),
                    }
                )
                if markdown_issues:
                    # Decide from the repaired state, not its former COMPLETED
                    # phase, so the normal one-round revision policy remains the
                    # single source of truth.
                    article.phase = ArticlePhase.ARTICLE_COMPOSITION.value
                    article.status = ArticleStatus.SUCCESS.value
                article.review_report = reconciled_review.model_dump_json(by_alias=True)
                article.quality_score = reconciled_review.score
                if markdown_issues:
                    decision = decide_next_action(article, reconciled_review, self.settings.max_auto_review_rounds)
                    article.supervisor_action = decision.action.value
                    article.supervisor_reason = decision.reason
                corrected += 1

            if corrected:
                session.commit()
            return corrected

    def job_status(self, task_id: str) -> WorkflowJobStatus:
        article = self.get(task_id)
        is_active = article.status in {ArticleStatus.QUEUED.value, ArticleStatus.RUNNING.value}
        started_at = article.update_time if is_active else None
        elapsed_seconds = None
        if started_at is not None:
            elapsed_seconds = max(int((self._clock() - started_at).total_seconds()), 0)
        return WorkflowJobStatus(
            task_id=article.task_id,
            status=article.status,
            action=self.public_action(article.supervisor_action),
            execution_mode=article.execution_mode,
            message=article.supervisor_reason or "",
            error_message=article.error_message,
            started_at=started_at,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def action_label(action: str) -> str:
        action_name = (ArticleWorkflow.public_action(action) or "").partition(":")[0]
        labels = {
            SupervisorAction.GENERATE_TITLES.value: "生成标题候选",
            SupervisorAction.GENERATE_OUTLINE.value: "生成文章大纲",
            SupervisorAction.GENERATE_CONTENT.value: "生成正文",
            SupervisorAction.GENERATE_IMAGE_PROMPTS.value: "生成配图方案",
            SupervisorAction.GENERATE_IMAGE_RESULTS.value: "生成图片素材",
            SupervisorAction.COMPOSE_ARTICLE.value: "合成图文稿",
            SupervisorAction.REVIEW_ARTICLE.value: "执行质量评审",
            SupervisorAction.REGENERATE_CONTENT.value: "重新生成正文",
            SupervisorAction.REGENERATE_IMAGE_PROMPTS.value: "重新生成配图方案",
            SupervisorAction.REGENERATE_IMAGE_RESULTS.value: "重新生成图片素材",
            "REGENERATE_IMAGE_RESULT": "重试图片",
            "ONE_CLICK_GENERATE": "一键生成全文",
            "APPLY_REVIEW_REVISION": "执行评审修订",
        }
        return labels.get(action_name, action_name)

    @staticmethod
    def encode_image_action(action: str, provider: ImageProvider) -> str:
        action_name = action.partition(":")[0]
        if provider is ImageProvider.AUTO or action_name not in _IMAGE_ACTIONS:
            return action
        return WorkflowOperation(action=action, provider=provider.value).encode()

    @staticmethod
    def public_action(action: str | None) -> str | None:
        return WorkflowOperation.parse(action).action if action else None

    @staticmethod
    def provider_for_action(action: str, fallback: ImageProvider = ImageProvider.AUTO) -> ImageProvider:
        operation = WorkflowOperation.parse(action)
        if operation.provider is None:
            return fallback
        try:
            return ImageProvider(operation.provider)
        except ValueError as error:
            raise WorkflowValidationError("图片任务中的供应商信息不合法") from error

    def ensure_not_busy(self, task_id: str) -> None:
        with self.task_write_lock(task_id):
            article = self.get(task_id)
            self._ensure_article_writable(article)

    def apply_review_revision(self, task_id: str) -> Article:
        """Apply the persisted review recommendation exactly once when eligible."""

        with self.task_write_lock(task_id):
            self.ensure_not_busy(task_id)
            article = self.get(task_id)
            decision = decide_next_action(article, self._review_result(article), self.settings.max_auto_review_rounds)
            if not decision.auto_revision:
                return self._record_decision(task_id, decision.action.value, decision.reason)
            return self._apply_review_revision(task_id, decision)

    def _apply_review_revision(self, task_id: str, decision) -> Article:
        """Regenerate the reviewed artifact, then advance the now-invalidated downstream chain."""

        self._update(
            task_id,
            review_round=decision.review_round + 1,
            supervisor_action=decision.action.value,
            supervisor_reason=decision.reason,
            review_report=None,
            quality_score=None,
        )
        if decision.action is SupervisorAction.REGENERATE_CONTENT:
            self.generate_content(task_id)
        elif decision.action is SupervisorAction.REGENERATE_IMAGE_PROMPTS:
            self.generate_image_prompts(task_id)
        elif decision.action is SupervisorAction.REGENERATE_IMAGE_RESULTS:
            self.generate_image_results(task_id, ImageProvider.AUTO)
        else:  # Defensive guard against future supervisor actions.
            raise WorkflowExecutionError(f"unsupported review revision: {decision.action}")
        return self.one_click(task_id)

    def one_click(self, task_id: str) -> Article:
        with self.task_write_lock(task_id):
            self.ensure_not_busy(task_id)
            for _ in range(24):
                article = self.get(task_id)
                if article.phase == ArticlePhase.COMPLETED.value:
                    # ``background_job_execution`` is also used by a few
                    # internal synchronous callers to bypass foreground edit
                    # guards.  Only a persisted ONE_CLICK runner job should
                    # remain RUNNING until JobRunner.finish_job(); a direct
                    # orchestration call is terminal at this point.
                    if (
                        article.status == ArticleStatus.RUNNING.value
                        and article.execution_mode != WorkflowRunMode.ONE_CLICK.value
                    ):
                        return self._update(task_id, status=ArticleStatus.SUCCESS.value)
                    return article
                if article.status == ArticleStatus.FAILED.value:
                    raise WorkflowExecutionError("任务已失败，请通过后台重试继续一键生成")
                if article.phase == ArticlePhase.TITLE_SELECTION.value and not article.selected_title:
                    titles = self._titles(article)
                    if not titles:
                        raise WorkflowExecutionError("标题候选为空")
                    self.confirm_title(task_id, titles[0].title)
                    continue
                updated = self.continue_workflow(task_id)
                if updated.supervisor_action in {SupervisorAction.MANUAL_REVIEW.value, SupervisorAction.WAIT_FOR_TITLE.value}:
                    return updated
            raise WorkflowExecutionError("一键生成超过最大步骤数")

    def one_click_initial_action(self, task_id: str) -> str:
        """Return the first concrete action users should see for a one-click run."""

        article = self.get(task_id)
        if article.status == ArticleStatus.FAILED.value and article.supervisor_action:
            return self.public_action(article.supervisor_action) or article.supervisor_action
        if article.phase == ArticlePhase.TITLE_SELECTION.value and not article.selected_title:
            return SupervisorAction.GENERATE_OUTLINE.value
        return self.next_decision(task_id).action.value

    def save_model_config(self, task_id: str, style: str, config: ArticleModelConfig) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                # A redundant save used to erase a completed task even though
                # it changed nothing.  Keep the existing public endpoint, but
                # make identical settings an idempotent operation.
                if article.style == style and self._config(article) == config:
                    session.expunge(article)
                    return article
                article.style = style
                article.model_config = config.model_dump_json(by_alias=True)
                for field in (
                    "title_options",
                    "selected_title",
                    "outline",
                    "content",
                    "full_content",
                    "image_prompts",
                    "image_results",
                    "final_markdown",
                    "review_report",
                    "quality_score",
                    "supervisor_action",
                    "supervisor_reason",
                    "execution_mode",
                ):
                    setattr(article, field, None)
                article.review_round = 0
                article.phase = ArticlePhase.CREATED.value
                article.status = ArticleStatus.INIT.value
                article.error_message = None
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def export_html(self, task_id: str) -> ExportFile:
        article = self.get(task_id)
        self._require_final_markdown(article)
        return self.exporter.export_html(article.selected_title or article.topic, article.final_markdown or "", self._images(article))

    def export_markdown(self, task_id: str) -> ExportFile:
        article = self.get(task_id)
        self._require_final_markdown(article)
        return self.exporter.export_markdown(article.selected_title or article.topic, article.final_markdown or "")

    def export_zip(self, task_id: str) -> ExportFile:
        article = self.get(task_id)
        self._require_final_markdown(article)
        return self.exporter.export_zip(article.selected_title or article.topic, article.final_markdown or "", self._images(article))

    def logs(self, task_id: str) -> list[AgentLog]:
        with self.sessions() as session:
            logs = list(session.scalars(select(AgentLog).where(AgentLog.task_id == task_id).order_by(AgentLog.create_time.desc())))
            for log in logs:
                session.expunge(log)
            return logs

    def _generate_titles(self, article: Article) -> str:
        agent = self._agent_for(article)
        titles = [
            item
            for item in agent.titles(article.topic, article.style)
            if not factuality_contract_issues(f"{item.title}\n{item.reason}")
        ]
        if not titles:
            raise WorkflowValidationError("生成的标题包含不可核验的资源或背书，请重试生成标题。")
        article.title_options = json.dumps([item.model_dump(by_alias=True) for item in titles], ensure_ascii=False)
        article.phase = ArticlePhase.TITLE_SELECTION.value
        article.status = ArticleStatus.SUCCESS.value
        return article.title_options

    def _generate_outline(self, article: Article) -> str:
        if not article.selected_title:
            raise WorkflowValidationError("请先确认标题")
        agent = self._agent_for(article)
        outline = agent.outline(article.topic, article.selected_title, article.style)
        outline_issues = factuality_contract_issues("\n".join(f"{item.heading}\n{item.description}" for item in outline))
        if outline_issues:
            raise WorkflowValidationError("生成的大纲未通过事实与资源检查：" + "；".join(outline_issues))
        article.outline = json.dumps([item.model_dump(by_alias=True) for item in outline], ensure_ascii=False)
        article.phase = ArticlePhase.OUTLINE_EDITING.value
        article.status = ArticleStatus.SUCCESS.value
        return article.outline

    def _generate_content(self, article: Article) -> str:
        if not article.selected_title or not self._outline(article):
            raise WorkflowValidationError("请先确认标题并生成大纲")
        config = self._config(article)
        agent = self._agent_for(article)
        candidate = agent.content(
            article.topic,
            article.selected_title,
            self._outline(article),
            article.style,
            config.content_length_level.value,
        )
        article.content, _ = self._validated_generated_content(article, candidate, agent)
        self._set_content(article, article.content)
        return article.content

    def _generate_image_prompts(self, article: Article) -> str:
        if not article.content or not article.selected_title:
            raise WorkflowValidationError("请先生成正文")
        config = self._config(article)
        count = config.image_count if config.image_count_mode is ImageCountMode.CUSTOM else calculate_image_count(article.content)
        prompts = self._agent_for(article).image_prompts(article.topic, article.selected_title, article.content, article.style, count or 3)
        prompts, adjusted = self._normalize_generated_image_plan(article.content, prompts)
        self._validate_image_plan_contract(article.content, prompts)
        article.image_prompts = json.dumps([item.model_dump(by_alias=True) for item in prompts], ensure_ascii=False)
        article.image_results = article.final_markdown = None
        article.review_report = article.quality_score = None
        article.phase = ArticlePhase.CONTENT_GENERATION.value
        article.status = ArticleStatus.SUCCESS.value
        return "配图方案已生成" + (f"；已自动校准 {adjusted} 项图片编排" if adjusted else "")

    def _generate_images_incrementally(self, task_id: str, requested_provider: ImageProvider) -> None:
        """Generate and persist each asset at an independent recovery boundary.

        A provider can fail on image 3 after two successful, billable calls.
        Keeping the earlier assets prevents a retry from repeating calls or
        making the user wait through the whole batch again.
        """

        with self.task_write_lock(task_id):
            article = self.get(task_id)
            self._ensure_article_writable(article)
            raw_prompts = self._prompts(article)
            prompts = [self._with_anchor_excerpt(article.content or "", prompt) for prompt in raw_prompts]
            if not prompts:
                raise WorkflowValidationError("请先生成图片提示词")
            if prompts != raw_prompts:
                # Existing task records predate anchor excerpts. Upgrade them
                # lazily at the safe pre-generation boundary so retrying an
                # older plan gets the same grounded provider prompt as a newly
                # created plan, without forcing the user to regenerate it.
                self._update(task_id, image_prompts=self._dump_image_records(prompts))
            results = self._images(article)
            if len(results) > len(prompts):
                raise WorkflowValidationError("图片结果数量异常，请重新生成图片素材")

            config = self._config(article)
            if requested_provider is ImageProvider.AUTO and config.image_provider is not ImageProvider.AUTO:
                # An article-level setting is an explicit user choice in both
                # manual and one-click flows. Only AUTO is allowed to try a
                # different supplier or a local fallback.
                provider = config.image_provider
            else:
                provider = requested_provider

            for index in range(len(results), len(prompts)):
                if self.cancellation_requested(task_id):
                    raise WorkflowCancellationRequested("任务已请求取消")
                self._record_image_generation_progress(task_id, results, index, len(prompts), provider)
                started = time.perf_counter()
                try:
                    result = self._generate_image_for_task(
                        prompts[index],
                        provider,
                        index,
                        task_id,
                    )
                except Exception as error:
                    self._record_image_generation_failure(task_id, index, len(prompts), provider, error, started)
                    raise
                results.append(result)
                self._record_image_generation_progress(task_id, results, index + 1, len(prompts), provider, started=started)

    def _record_image_generation_progress(
        self,
        task_id: str,
        results: list[ImageResultOption],
        completed: int,
        total: int,
        provider: ImageProvider,
        *,
        started: float | None = None,
    ) -> None:
        """Publish real progress and keep the successfully generated prefix."""

        with self.sessions() as session:
            article = self._get_article(session, task_id)
            article.image_results = self._dump_image_records(results) if results else None
            article.final_markdown = None
            article.review_report = article.quality_score = None
            article.phase = ArticlePhase.CONTENT_GENERATION.value
            article.supervisor_action = self.encode_image_action(SupervisorAction.GENERATE_IMAGE_RESULTS.value, provider)
            article.supervisor_reason = (
                f"已生成 {completed}/{total} 张图片" if completed >= total else f"正在生成第 {completed + 1}/{total} 张图片"
            )
            if self._background_job_execution.get():
                article.status = ArticleStatus.RUNNING.value
            else:
                article.status = ArticleStatus.SUCCESS.value
            article.error_message = None
            article.update_time = self._clock()
            if started is not None:
                self._log(
                    session,
                    task_id,
                    "ImageResultAgent",
                    f"image={completed}/{total}; provider={provider.value}",
                    article.supervisor_reason,
                    started,
                )
            session.commit()

    def _record_image_generation_failure(
        self,
        task_id: str,
        index: int,
        total: int,
        provider: ImageProvider,
        error: Exception,
        started: float,
    ) -> None:
        with self.sessions() as session:
            article = self._get_article(session, task_id)
            article.status = ArticleStatus.FAILED.value
            article.supervisor_action = self.encode_image_action(SupervisorAction.GENERATE_IMAGE_RESULTS.value, provider)
            article.supervisor_reason = f"第 {index + 1}/{total} 张图片生成失败，可重试剩余图片"
            article.error_message = _safe_error_message(error)
            article.update_time = self._clock()
            self._log(
                session,
                task_id,
                "ImageResultAgent",
                f"image={index + 1}/{total}; provider={provider.value}",
                None,
                started,
                error,
            )
            session.commit()

    def _regenerate_image_result(self, article: Article, index: int, provider: ImageProvider) -> str:
        raw_prompts, results = self._prompts(article), self._images(article)
        prompts = [self._with_anchor_excerpt(article.content or "", prompt) for prompt in raw_prompts]
        if prompts != raw_prompts:
            article.image_prompts = self._dump_image_records(prompts)
        if index < 0 or index >= len(prompts) or len(results) > len(prompts):
            raise WorkflowValidationError("图片索引或图片数据不合法")
        if index > len(results):
            raise WorkflowValidationError("请先重试前一张未完成的图片")
        generated = self._generate_image_for_task(prompts[index], provider, index, article.task_id)
        if index == len(results):
            results.append(generated)
        else:
            results[index] = generated
        article.image_results = json.dumps([item.model_dump(by_alias=True) for item in results], ensure_ascii=False)
        article.final_markdown = None
        article.review_report = article.quality_score = None
        article.phase = ArticlePhase.CONTENT_GENERATION.value
        article.status = ArticleStatus.SUCCESS.value
        article.supervisor_action = None
        article.supervisor_reason = "单张图片已重试，请继续合成图文稿并重新评审"
        return article.image_results

    def _review_generated_images(self, article: Article, indices: list[int] | None = None) -> str:
        """Attach optional visual checks without changing workflow state or assets."""

        prompts, results = self._prompts(article), self._images(article)
        if len(prompts) != len(results):
            raise WorkflowValidationError("图片结果与配图方案数量不一致，无法执行视觉质检")
        selected_indices = indices if indices is not None else list(range(len(results)))
        for index in selected_indices:
            if index < 0 or index >= len(results):
                raise WorkflowValidationError("图片索引不存在，无法执行视觉质检")
            results[index] = results[index].model_copy(
                update={"visual_review": self._safe_visual_review(results[index], prompts[index])}
            )
        article.image_results = self._dump_image_records(results)
        counts: dict[str, int] = {}
        for index in selected_indices:
            status = results[index].visual_review.status.value
            counts[status] = counts.get(status, 0) + 1
        return "视觉质检：" + "，".join(f"{status} {count}" for status, count in sorted(counts.items()))

    def _safe_visual_review(self, image: ImageResultOption, prompt: ImagePromptOption):
        try:
            return self.vision_reviewer.review(image, prompt)
        except Exception:
            # Third-party visual review must never fail a usable generated image.
            return ImageVisualReview(
                status=ImageVisualReviewStatus.UNAVAILABLE,
                issues=["视觉质检结果不可用，可按需重试。"],
            )

    def _compose(self, article: Article) -> str:
        article.final_markdown = compose_article(article.selected_title or "", article.content or "", self._images(article))
        article.phase = ArticlePhase.ARTICLE_COMPOSITION.value
        article.status = ArticleStatus.SUCCESS.value
        article.review_report = article.quality_score = None
        return article.final_markdown

    def _review(self, article: Article) -> str:
        if not article.final_markdown:
            raise WorkflowValidationError("请先生成最终图文稿")
        result = self._agent_for(article).review(article.final_markdown)
        blocking_issues = list(
            dict.fromkeys(
                [
                    *result.blocking_issues,
                    *content_contract_issues(
                        article.final_markdown,
                        topic=article.topic,
                        title=article.selected_title or article.topic,
                        style=article.style,
                    ),
                ]
            )
        )
        if blocking_issues:
            result = result.model_copy(
                update={
                    "passed": False,
                    "blocking_issues": blocking_issues,
                    "revision_action": result.revision_action or SupervisorAction.REGENERATE_CONTENT.value,
                    "revision_reason": result.revision_reason or "存在阻断交付的问题，需要修订正文。",
                }
            )
        article.quality_score = result.score
        article.review_report = result.model_dump_json(by_alias=True)
        decision = decide_next_action(article, result, self.settings.max_auto_review_rounds)
        article.supervisor_action = decision.action.value
        article.supervisor_reason = decision.reason
        if decision.action is SupervisorAction.COMPLETE:
            article.phase = ArticlePhase.COMPLETED.value
            article.status = ArticleStatus.SUCCESS.value
        return article.review_report

    def _run(self, task_id: str, agent_name: str, operation: Callable[[Article], str]) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                started = time.perf_counter()
                try:
                    if self.cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    output = operation(article)
                    if self.cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    if self._background_job_execution.get():
                        # Each agent persists its own artifact and normally
                        # labels that local boundary SUCCESS.  A JobRunner may
                        # still be executing later one-click substeps, though;
                        # only finish_job() is allowed to expose a terminal
                        # success to the page and compatibility APIs.
                        article.status = ArticleStatus.RUNNING.value
                    article.error_message = None
                    # Keep active-job elapsed time on the application clock.
                    # SQLite's ``func.now()`` uses UTC while this local UI
                    # intentionally displays the configured local clock.
                    article.update_time = self._clock()
                    self._log(session, task_id, agent_name, "input managed by workflow", output, started)
                    session.commit()
                    session.refresh(article)
                    session.expunge(article)
                    return article
                except WorkflowCancellationRequested:
                    session.rollback()
                    raise
                except Exception as error:
                    article.status = ArticleStatus.FAILED.value
                    article.error_message = _safe_error_message(error)
                    self._log(session, task_id, agent_name, "input managed by workflow", None, started, error)
                    session.commit()
                    raise

    def _mark_interrupted_stream_retryable(self, task_id: str) -> None:
        """Turn an abandoned SSE response into a normal, retryable content job."""

        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                # A cancellation request wins over a client disconnect and is
                # handled by the caller's deletion path.  Likewise, never
                # overwrite a successfully persisted stream result.
                if article.status != ArticleStatus.RUNNING.value or article.content:
                    return
                article.status = ArticleStatus.FAILED.value
                article.supervisor_action = SupervisorAction.GENERATE_CONTENT.value
                article.supervisor_reason = "正文流式连接中断，可重试生成正文"
                article.error_message = "正文生成连接中断，未保存不完整内容；请重试。"
                article.update_time = self._clock()
                session.commit()

    def _update(self, task_id: str, **values: object) -> Article:
        with self.task_write_lock(task_id):
            with self.sessions() as session:
                article = self._get_article(session, task_id)
                self._ensure_article_writable(article)
                for field, value in values.items():
                    setattr(article, field, value)
                article.update_time = self._clock()
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def _update_downstream(self, task_id: str, **values: object) -> Article:
        values.update({"image_prompts": None, "image_results": None, "final_markdown": None, "review_report": None, "quality_score": None})
        return self._update(task_id, **values)

    def _record_decision(self, task_id: str, action: str, reason: str) -> Article:
        return self._update(task_id, supervisor_action=action, supervisor_reason=reason)

    def _mark_current_action(self, task_id: str, action: str) -> Article:
        """Persist the actual sub-step before invoking a provider.

        A one-click job is an orchestration mode, not a retry target.  Keeping
        the concrete operation here means an exception can be shown honestly
        and a normal step retry still has an exact provider-aware action.
        """

        return self._update(
            task_id,
            supervisor_action=action,
            supervisor_reason=f"正在执行：{self.action_label(action)}",
        )

    def _ensure_article_writable(self, article: Article) -> None:
        if article.status == ArticleStatus.CANCELLING.value:
            raise WorkflowBusyError("任务正在取消并永久删除，请等待完成。")
        if article.status in {ArticleStatus.QUEUED.value, ArticleStatus.RUNNING.value} and not self._background_job_execution.get():
            raise WorkflowBusyError("任务正在后台执行中，请等待完成后再编辑。")

    @staticmethod
    def _normalize_generated_image_plan(
        content: str,
        plans: list[ImagePromptOption],
    ) -> tuple[list[ImagePromptOption], int]:
        """Repair recoverable model output before applying hard invariants.

        A planner receives an authoritative catalogue, but a model can still
        mistake a wrapped line for a second paragraph or reuse an otherwise
        valid visual role. Neither should turn a harmless planning variation
        into a failed one-click article. User edits deliberately bypass this
        method and remain strict.
        """

        sections = markdown_anchor_sections(content)
        by_key: dict[str, object] = {}
        for section in sections:
            by_key.setdefault(section.key, section)
        normalized: list[ImagePromptOption] = []
        adjusted_indices: set[int] = set()
        fallback_cursor = 0

        def choose_fallback(excluded: set[str]) -> tuple[str, int, ImagePlacement]:
            nonlocal fallback_cursor
            if not sections:
                return "", 0, ImagePlacement.AFTER_INTRO
            for offset in range(len(sections)):
                section = sections[(fallback_cursor + offset) % len(sections)]
                candidate = f"section-end:{section.key}"
                if candidate not in excluded:
                    fallback_cursor = (fallback_cursor + offset + 1) % len(sections)
                    return section.heading, 0, ImagePlacement.SECTION_END
            section = sections[fallback_cursor % len(sections)]
            fallback_cursor = (fallback_cursor + 1) % len(sections)
            return section.heading, 0, ImagePlacement.SECTION_END

        previous_anchor = ""
        for index, plan in enumerate(plans):
            original = plan
            if plan.placement is ImagePlacement.COVER:
                plan = plan.model_copy(
                    update={"placement": ImagePlacement.COVER, "anchor_heading": "", "anchor_paragraph": 0}
                )
            elif plan.placement is ImagePlacement.AFTER_INTRO:
                plan = plan.model_copy(update={"anchor_heading": "", "anchor_paragraph": 0})
            else:
                section = by_key.get(heading_key(plan.anchor_heading))
                if section is None:
                    heading, paragraph, placement = choose_fallback({previous_anchor})
                    plan = plan.model_copy(
                        update={"anchor_heading": heading, "anchor_paragraph": paragraph, "placement": placement}
                    )
                elif plan.placement is ImagePlacement.AFTER_PARAGRAPH:
                    if section.paragraphs:
                        paragraph = min(max(plan.anchor_paragraph, 1), len(section.paragraphs))
                        plan = plan.model_copy(
                            update={"anchor_heading": section.heading, "anchor_paragraph": paragraph}
                        )
                    else:
                        plan = plan.model_copy(
                            update={
                                "anchor_heading": section.heading,
                                "anchor_paragraph": 0,
                                "placement": ImagePlacement.SECTION_END,
                            }
                        )
                else:
                    plan = plan.model_copy(
                        update={"anchor_heading": section.heading, "anchor_paragraph": 0, "placement": ImagePlacement.SECTION_END}
                    )

            if plan != original:
                adjusted_indices.add(index)
            normalized.append(plan)
            previous_anchor = ArticleWorkflow._image_anchor_key(plan)

        # ``visual_role`` is presentation metadata, not a structural anchor.
        # The model is told to keep it unique, but repairing a duplicate here
        # is safer than failing before any image-provider call. COVER remains
        # deliberately strict: a missing, duplicated, or misplaced cover is a
        # structural error that must be shown to the user rather than guessed.
        used_roles: set[ImageVisualRole] = set()
        role_normalized: list[ImagePromptOption] = []
        for index, plan in enumerate(normalized):
            if plan.visual_role in used_roles and plan.visual_role is not ImageVisualRole.COVER:
                replacement = ArticleWorkflow._replacement_visual_role(plan, used_roles)
                if replacement is not None:
                    plan = plan.model_copy(update={"visual_role": replacement})
                    adjusted_indices.add(index)
            role_normalized.append(plan)
            used_roles.add(plan.visual_role)
        grounded: list[ImagePromptOption] = []
        for index, plan in enumerate(role_normalized):
            evidence_bound = ArticleWorkflow._with_anchor_excerpt(content, plan)
            if evidence_bound != plan:
                adjusted_indices.add(index)
            grounded.append(evidence_bound)
        return grounded, len(adjusted_indices)

    @staticmethod
    def _with_anchor_excerpt(content: str, plan: ImagePromptOption) -> ImagePromptOption:
        """Bind a plan to the authoritative prose it should make visible."""

        return plan.model_copy(
            update={
                "anchor_excerpt": image_anchor_excerpt(
                    content,
                    plan.placement,
                    plan.anchor_heading,
                    plan.anchor_paragraph,
                )
            }
        )

    @staticmethod
    def _replacement_visual_role(
        plan: ImagePromptOption,
        used_roles: set[ImageVisualRole],
    ) -> ImageVisualRole | None:
        """Choose a deterministic unused role for a duplicated model plan.

        The role remains meaningful for AUTO provider routing, so first prefer
        candidates indicated by the image's title, scene, or prompts. The
        stable fallback order makes retries reproducible when the text offers
        no useful hint.
        """

        semantic_text = " ".join(
            (plan.image_title, plan.usage_scene, plan.prompt_zh, plan.prompt_en)
        ).casefold()
        preferred: list[ImageVisualRole] = []

        def add_if_matches(role: ImageVisualRole, keywords: tuple[str, ...]) -> None:
            if any(keyword in semantic_text for keyword in keywords):
                preferred.append(role)

        add_if_matches(
            ImageVisualRole.DIAGRAM,
            ("diagram", "flow", "workflow", "architecture", "interface", "ui", "图解", "流程", "架构", "界面", "信息图"),
        )
        add_if_matches(
            ImageVisualRole.DATA,
            ("data", "chart", "metric", "dashboard", "数据", "图表", "指标", "看板"),
        )
        add_if_matches(
            ImageVisualRole.COMPARISON,
            ("compare", "comparison", "versus", "对比", "比较", "前后"),
        )
        add_if_matches(
            ImageVisualRole.SCENE,
            ("scene", "photo", "portrait", "people", "人物", "场景", "照片"),
        )
        add_if_matches(
            ImageVisualRole.DETAIL,
            ("detail", "closeup", "局部", "细节", "特写"),
        )
        add_if_matches(
            ImageVisualRole.CLOSING,
            ("closing", "conclusion", "ending", "结尾", "结语", "收束"),
        )

        for candidate in (*preferred, *_GENERATED_ROLE_FALLBACK_ORDER):
            if candidate not in used_roles:
                return candidate
        return None

    @staticmethod
    def _validate_image_plan_anchor(content: str, plan: ImagePromptOption) -> None:
        """Reject stale or impossible editor anchors before persisting them."""

        if not content.strip():
            raise WorkflowValidationError("请先生成正文，再调整图片位置")
        anchor_heading = plan.anchor_heading.strip()
        placement = plan.placement
        if placement in {ImagePlacement.COVER, ImagePlacement.AFTER_INTRO}:
            if anchor_heading or plan.anchor_paragraph:
                raise WorkflowValidationError("封面或导语后的配图不能指定章节或段落锚点")
            return

        sections = ArticleWorkflow._markdown_sections(content)
        matching = next(
            ((heading, blocks) for heading, blocks in sections if ArticleWorkflow._heading_key(heading) == ArticleWorkflow._heading_key(anchor_heading)),
            None,
        )
        if matching is None:
            raise WorkflowValidationError(f"未找到锚点标题：{anchor_heading or '（未填写）'}")
        _heading, blocks = matching
        if placement is ImagePlacement.AFTER_PARAGRAPH:
            if not 1 <= plan.anchor_paragraph <= len(blocks):
                raise WorkflowValidationError(f"锚点段落无效：标题“{anchor_heading}”下共有 {len(blocks)} 段，不能定位第 {plan.anchor_paragraph} 段")
            return
        if plan.anchor_paragraph != 0:
            raise WorkflowValidationError("章节末尾配图的段落序号必须为 0")

    @staticmethod
    def _validate_image_plan_contract(content: str, plans: list[ImagePromptOption]) -> None:
        """Keep generated and manually arranged plans structurally composable."""

        if not plans:
            raise WorkflowValidationError("配图方案不能为空")
        for plan in plans:
            ArticleWorkflow._validate_image_plan_anchor(content, plan)

        cover_indices = [index for index, plan in enumerate(plans) if plan.visual_role is ImageVisualRole.COVER]
        if len(cover_indices) != 1:
            raise WorkflowValidationError("配图方案必须且只能包含一张封面图")
        if cover_indices[0] != 0:
            raise WorkflowValidationError("封面图必须位于配图方案第 1 项")
        if plans[0].placement is not ImagePlacement.COVER or any(
            plan.placement is ImagePlacement.COVER for plan in plans[1:]
        ):
            raise WorkflowValidationError("封面图必须使用 COVER 位置，且其他配图不能使用 COVER 位置")

        roles = [plan.visual_role for plan in plans]
        if len(set(roles)) != len(roles):
            raise WorkflowValidationError("同一配图方案中的视觉角色不能重复")

        for previous, current in zip(plans, plans[1:]):
            if ArticleWorkflow._image_anchor_key(previous) == ArticleWorkflow._image_anchor_key(current):
                raise WorkflowValidationError("相邻配图不能使用同一锚点")

    @staticmethod
    def _image_anchor_key(plan: ImagePromptOption) -> str:
        if plan.placement is ImagePlacement.COVER:
            return "cover"
        if plan.placement is ImagePlacement.AFTER_INTRO:
            return "intro"
        heading = ArticleWorkflow._heading_key(plan.anchor_heading)
        if plan.placement is ImagePlacement.AFTER_PARAGRAPH:
            return f"paragraph:{heading}:{plan.anchor_paragraph}"
        return f"section-end:{heading}"

    @staticmethod
    def _markdown_sections(content: str) -> list[tuple[str, tuple[str, ...]]]:
        return [(section.heading, section.paragraphs) for section in markdown_anchor_sections(content)]

    @staticmethod
    def _heading_key(value: str) -> str:
        return heading_key(value)

    @staticmethod
    def _dump_image_records(records: list[ImagePromptOption] | list[ImageResultOption]) -> str:
        return json.dumps([record.model_dump(by_alias=True) for record in records], ensure_ascii=False)

    def _local_image_paths(self, raw_image_results: str | None) -> set[Path]:
        """Resolve only app-hosted image URLs; never touch remote supplier URLs."""

        try:
            results = json.loads(raw_image_results or "[]")
        except json.JSONDecodeError:
            return set()
        if not isinstance(results, list):
            return set()

        root = self.settings.upload_dir.resolve()
        paths: set[Path] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            image_url = result.get("imageUrl")
            if not isinstance(image_url, str) or not image_url.startswith("/uploads/images/"):
                continue
            relative = image_url.removeprefix("/uploads/images/")
            if not relative:
                continue
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root):
                paths.add(candidate)
        return paths

    def _referenced_local_image_paths(self, session: Session, excluding_task_id: str) -> set[Path]:
        referenced: set[Path] = set()
        rows = session.scalars(select(Article.image_results).where(Article.task_id != excluding_task_id))
        for raw_image_results in rows:
            referenced.update(self._local_image_paths(raw_image_results))
        return referenced

    def _delete_unreferenced_local_assets(self, task_id: str, local_paths: set[Path], referenced_paths: set[Path]) -> None:
        """Clean this task's generated files without touching shared/manual assets."""

        for path in local_paths - referenced_paths:
            path.unlink(missing_ok=True)

        root = self.settings.upload_dir.resolve()
        task_dir = (root / task_id).resolve()
        if not task_dir.is_relative_to(root) or not task_dir.is_dir():
            return
        for path in sorted(task_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and path.resolve() not in referenced_paths:
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            task_dir.rmdir()
        except OSError:
            pass

    def _generate_images_for_task(
        self,
        prompts: list[ImagePromptOption],
        provider: ImageProvider,
        task_id: str,
    ) -> list[ImageResultOption]:
        generate = self.images.generate
        if _accepts_task_id(generate):
            return generate(prompts, provider, task_id=task_id)
        return generate(prompts, provider)

    def _generate_image_for_task(
        self,
        prompt: ImagePromptOption,
        provider: ImageProvider,
        index: int,
        task_id: str,
    ) -> ImageResultOption:
        generate_one = self.images.generate_one
        kwargs: dict[str, object] = {}
        if _accepts_task_id(generate_one):
            kwargs["task_id"] = task_id
        return generate_one(prompt, provider, index, **kwargs)

    def _lock_for_task(self, task_id: str) -> RLock:
        with self._task_locks_guard:
            return self._task_locks.setdefault(task_id, RLock())

    @staticmethod
    def _get_article(session: Session, task_id: str) -> Article:
        article = session.scalar(select(Article).where(Article.task_id == task_id, Article.is_deleted == 0))
        if not article:
            raise WorkflowValidationError("文章任务不存在")
        return article

    @staticmethod
    def _log(
        session: Session,
        task_id: str,
        agent_name: str,
        input_text: str,
        output_text: str | None,
        started: float,
        error: Exception | None = None,
    ) -> None:
        session.add(
            AgentLog(
                task_id=task_id,
                agent_name=agent_name,
                input_text=_redact_sensitive_text(input_text)[:4000],
                output_text=_redact_sensitive_text(output_text or "")[:8000] or None,
                status="FAILED" if error else "SUCCESS",
                cost_ms=int((time.perf_counter() - started) * 1000),
                error_message=_safe_error_message(error)[:1000] if error else None,
            )
        )

    @staticmethod
    def _config(article: Article) -> ArticleModelConfig:
        return ArticleModelConfig.model_validate_json(article.model_config or "{}")

    def _agent_for(self, article: Article) -> ArticleAgent:
        """Bind request parameters to real adapters without mutating shared state."""

        configure = getattr(self.agent, "with_model_config", None)
        return configure(self._config(article)) if callable(configure) else self.agent

    @staticmethod
    def _set_content(article: Article, content: str) -> None:
        article.content = content
        article.phase = ArticlePhase.CONTENT_GENERATION.value
        article.status = ArticleStatus.SUCCESS.value
        article.image_prompts = article.image_results = article.final_markdown = None
        article.review_report = article.quality_score = None

    @staticmethod
    def _content_contract(article: Article, content: str) -> list[str]:
        return content_contract_issues(
            content,
            topic=article.topic,
            title=article.selected_title or article.topic,
            style=article.style,
        )

    def _raise_for_content_contract(self, article: Article, content: str) -> None:
        issues = self._content_contract(article, content)
        if issues:
            raise WorkflowValidationError("正文未通过内容质量检查：" + "；".join(issues))

    def _validated_generated_content(self, article: Article, candidate: str, agent: ArticleAgent) -> tuple[str, bool]:
        normalized_candidate = candidate.strip()
        issues = self._content_contract(article, normalized_candidate)
        if not issues:
            return normalized_candidate, False
        repair = getattr(agent, "repair_content", None)
        if not callable(repair):
            raise WorkflowValidationError("正文未通过内容质量检查，且当前模型不支持自动修订：" + "；".join(issues))
        repaired = repair(
            article.topic,
            article.selected_title or article.topic,
            self._outline(article),
            article.style,
            self._config(article).content_length_level.value,
            normalized_candidate,
            issues,
        )
        if not isinstance(repaired, str) or not repaired.strip():
            raise WorkflowValidationError("正文自动修订未返回有效内容")
        self._raise_for_content_contract(article, repaired.strip())
        return repaired.strip(), True

    @staticmethod
    def _titles(article: Article) -> list[TitleOption]:
        return [TitleOption.model_validate(item) for item in json.loads(article.title_options or "[]")]

    @staticmethod
    def _outline(article: Article) -> list[OutlineItem]:
        return [OutlineItem.model_validate(item) for item in json.loads(article.outline or "[]")]

    @staticmethod
    def _prompts(article: Article) -> list[ImagePromptOption]:
        return [ImagePromptOption.model_validate(item) for item in json.loads(article.image_prompts or "[]")]

    @staticmethod
    def _images(article: Article) -> list[ImageResultOption]:
        return [ImageResultOption.model_validate(item) for item in json.loads(article.image_results or "[]")]

    @staticmethod
    def _review_result(article: Article) -> QualityReviewResult | None:
        return QualityReviewResult.model_validate_json(article.review_report) if article.review_report else None

    @staticmethod
    def _require_final_markdown(article: Article) -> None:
        if not article.final_markdown:
            raise WorkflowValidationError("请先完成图文稿合成后再导出")


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _safe_error_message(error: Exception) -> str:
    """Return an error safe for persistence and page/API diagnostics."""

    return _redact_sensitive_text(str(error))


def _redact_sensitive_text(value: str) -> str:
    """Redact common auth headers plus JSON, query, and key-value secrets."""

    return redact_sensitive_text(value)


def _accepts_task_id(operation: object) -> bool:
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "task_id" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)

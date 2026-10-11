"""Private task dispatch, review continuation, and transactional agent execution.

The facade owns shared locks and the background ContextVar. This collaborator
uses that same context and invokes facade operations; it introduces no second
state machine, cached adapter, or independent transaction/lock registry.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._workflow_persistence import safe_error_message as _safe_error_message
from .domain import AdapterError, WorkflowCancellationRequested, WorkflowExecutionError, WorkflowOperation, WorkflowValidationError
from .models import Article
from .schemas import ImageProvider
from .supervisor import ArticlePhase, ArticleStatus, SupervisorAction, decide_next_action

if TYPE_CHECKING:
    from .workflow import ArticleWorkflow


class _WorkflowExecution:
    def __init__(self, workflow: ArticleWorkflow) -> None:
        self._workflow = workflow

    def continue_workflow(self, task_id: str) -> Article:
        workflow = self._workflow
        with workflow.task_write_lock(task_id):
            workflow.ensure_not_busy(task_id)
            decision = workflow.next_decision(task_id)
            if decision.auto_revision:
                return workflow._apply_review_revision(task_id, decision)
            return workflow.execute_action(task_id, decision.action.value)

    def execute_action(self, task_id: str, action: str, provider: ImageProvider = ImageProvider.AUTO) -> Article:
        """Run a persisted supervisor action, including a failed job's retry action."""
        workflow = self._workflow

        persisted_operation = WorkflowOperation.parse(action)
        persisted_action = persisted_operation.encode()
        provider = workflow.provider_for_action(persisted_action, provider)
        action = persisted_operation.action
        index: int | None = None
        raw_action, separator, raw_index = action.partition(":")
        if raw_action == "REGENERATE_IMAGE_RESULT":
            if not separator or not raw_index.isdigit():
                raise WorkflowValidationError("图片重试动作不合法")
            index = int(raw_index)
        if raw_action == "ONE_CLICK_GENERATE":
            return workflow.one_click(task_id)
        if raw_action == "APPLY_REVIEW_REVISION":
            return workflow.apply_review_revision(task_id)
        if raw_action == "REGENERATE_IMAGE_RESULT":
            assert index is not None
            return workflow.regenerate_image_result(task_id, index, provider)
        try:
            supervisor_action = SupervisorAction(raw_action)
        except ValueError as error:
            raise WorkflowValidationError("当前任务动作不支持执行") from error
        workflow._mark_current_action(task_id, persisted_action)
        if supervisor_action is SupervisorAction.GENERATE_TITLES:
            return workflow.generate_titles(task_id)
        if supervisor_action is SupervisorAction.GENERATE_OUTLINE:
            return workflow.generate_outline(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_CONTENT, SupervisorAction.REGENERATE_CONTENT}:
            return workflow.generate_content(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_IMAGE_PROMPTS, SupervisorAction.REGENERATE_IMAGE_PROMPTS}:
            return workflow.generate_image_prompts(task_id)
        if supervisor_action in {SupervisorAction.GENERATE_IMAGE_RESULTS, SupervisorAction.REGENERATE_IMAGE_RESULTS}:
            return workflow.generate_image_results(task_id, provider)
        if supervisor_action is SupervisorAction.COMPOSE_ARTICLE:
            return workflow.compose(task_id)
        if supervisor_action is SupervisorAction.REVIEW_ARTICLE:
            return workflow.review(task_id)
        if supervisor_action in {
            SupervisorAction.WAIT_FOR_TITLE,
            SupervisorAction.MANUAL_REVIEW,
            SupervisorAction.STOP_FAILED,
            SupervisorAction.COMPLETE,
        }:
            decision = workflow.next_decision(task_id)
            return workflow._record_decision(task_id, decision.action.value, decision.reason)
        raise WorkflowValidationError("当前任务动作不支持执行")

    def apply_review_revision(self, task_id: str) -> Article:
        """Apply the persisted review recommendation exactly once when eligible."""
        workflow = self._workflow

        with workflow.task_write_lock(task_id):
            workflow.ensure_not_busy(task_id)
            article = workflow.get(task_id)
            decision = decide_next_action(article, workflow._review_result(article), workflow.settings.max_auto_review_rounds)
            if not decision.auto_revision:
                return workflow._record_decision(task_id, decision.action.value, decision.reason)
            return workflow._apply_review_revision(task_id, decision)

    def execute_review_revision(self, task_id: str, decision) -> Article:
        """Regenerate the reviewed artifact, then advance the now-invalidated downstream chain."""
        workflow = self._workflow

        workflow._update(
            task_id,
            review_round=decision.review_round + 1,
            supervisor_action=decision.action.value,
            supervisor_reason=decision.reason,
            review_report=None,
            quality_score=None,
        )
        if decision.action is SupervisorAction.REGENERATE_CONTENT:
            workflow.generate_content(task_id)
        elif decision.action is SupervisorAction.REGENERATE_IMAGE_PROMPTS:
            workflow.generate_image_prompts(task_id)
        elif decision.action is SupervisorAction.REGENERATE_IMAGE_RESULTS:
            workflow.generate_image_results(task_id, ImageProvider.AUTO)
        else:  # Defensive guard against future supervisor actions.
            raise WorkflowExecutionError(f"unsupported review revision: {decision.action}")
        return workflow.one_click(task_id)

    def one_click(self, task_id: str) -> Article:
        workflow = self._workflow
        with workflow.task_write_lock(task_id):
            workflow.ensure_not_busy(task_id)
            for _ in range(24):
                article = workflow.get(task_id)
                if article.phase == ArticlePhase.COMPLETED.value:
                    # ``background_job_execution`` is also used by a few
                    # internal synchronous callers to bypass foreground edit
                    # guards. Persisted runner jobs remain RUNNING until
                    # JobRunner.finish_job(); direct orchestration is terminal.
                    if article.status == ArticleStatus.RUNNING.value and article.execution_mode is None:
                        return workflow._update(task_id, status=ArticleStatus.SUCCESS.value)
                    return article
                if article.status == ArticleStatus.FAILED.value:
                    raise WorkflowExecutionError("任务已失败，请通过后台重试继续一键生成")
                if article.phase == ArticlePhase.ARTICLE_COMPOSITION.value and article.final_markdown:
                    # Generating (or revising) a delivery draft never grants
                    # permission to review it. Leave that decision to the user,
                    # even when a previous review recommends another revision.
                    if not article.review_report:
                        article = workflow._record_decision(
                            task_id,
                            SupervisorAction.REVIEW_ARTICLE.value,
                            "图文稿已生成，等待手动审核",
                        )
                    if article.status == ArticleStatus.RUNNING.value and article.execution_mode is None:
                        return workflow._update(task_id, status=ArticleStatus.SUCCESS.value)
                    return article
                if article.phase == ArticlePhase.TITLE_SELECTION.value and not article.selected_title:
                    titles = workflow._titles(article)
                    if not titles:
                        raise WorkflowExecutionError("标题候选为空")
                    workflow.confirm_title(task_id, titles[0].title)
                    continue
                updated = workflow.continue_workflow(task_id)
                if updated.supervisor_action in {SupervisorAction.MANUAL_REVIEW.value, SupervisorAction.WAIT_FOR_TITLE.value}:
                    return updated
            raise WorkflowExecutionError("一键生成超过最大步骤数")

    def one_click_initial_action(self, task_id: str) -> str:
        """Return the first concrete action users should see for a one-click run."""
        workflow = self._workflow

        article = workflow.get(task_id)
        if article.status == ArticleStatus.FAILED.value and article.supervisor_action:
            return workflow.public_action(article.supervisor_action) or article.supervisor_action
        if article.phase == ArticlePhase.TITLE_SELECTION.value and not article.selected_title:
            return SupervisorAction.GENERATE_OUTLINE.value
        return workflow.next_decision(task_id).action.value

    def run(self, task_id: str, agent_name: str, operation: Callable[[Article], str]) -> Article:
        workflow = self._workflow
        with workflow.task_write_lock(task_id):
            with workflow.sessions() as session:
                article = workflow._get_article(session, task_id)
                workflow._ensure_article_writable(article)
                started = time.perf_counter()
                try:
                    if workflow.cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    output = operation(article)
                    if workflow.cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    if workflow._background_job_execution.get():
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
                    article.update_time = workflow._clock()
                    workflow._log(session, task_id, agent_name, "input managed by workflow", output, started)
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
                    diagnostics = error.diagnostics if isinstance(error, AdapterError) else None
                    workflow._log(session, task_id, agent_name, "input managed by workflow", diagnostics, started, error)
                    session.commit()
                    raise

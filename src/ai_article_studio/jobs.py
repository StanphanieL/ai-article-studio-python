"""Single-user background execution for non-streaming article workflow actions."""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import Lock
from time import sleep

from .schemas import ImageProvider, WorkflowJobStatus, WorkflowRunMode
from .supervisor import SupervisorAction
from .workflow import ArticleWorkflow, WorkflowCancellationRequested


class JobAlreadyRunning(RuntimeError):
    """Raised when a task is already running locally or waiting in the queue."""


class JobRunner:
    """Serialize expensive, non-streaming workflow operations for one local user."""

    def __init__(self, workflow: ArticleWorkflow, executor: Executor | None = None) -> None:
        self.workflow = workflow
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="article-workflow")
        self._owns_executor = executor is None
        self._active_task_ids: set[str] = set()
        self._futures: dict[str, Future[object]] = {}
        self._cancelling_task_ids: set[str] = set()
        self._deleting_task_ids: set[str] = set()
        self._lock = Lock()

    def status(self, task_id: str) -> WorkflowJobStatus:
        return self.workflow.job_status(task_id)

    def start_next(self, task_id: str) -> WorkflowJobStatus:
        with self.workflow.task_write_lock(task_id):
            decision = self.workflow.next_decision(task_id)
            if decision.action in {
                SupervisorAction.WAIT_FOR_TITLE,
                SupervisorAction.MANUAL_REVIEW,
                SupervisorAction.STOP_FAILED,
                SupervisorAction.COMPLETE,
            }:
                raise ValueError(decision.reason)
            if decision.auto_revision:
                return self._start_action(task_id, "APPLY_REVIEW_REVISION", "任务已开始：执行评审修订")
            action = decision.action.value
            return self._start_action(task_id, action, f"任务已开始：{self.workflow.action_label(action)}")

    def retry(self, task_id: str) -> WorkflowJobStatus:
        with self.workflow.task_write_lock(task_id):
            article = self.workflow.get(task_id)
            if article.execution_mode == WorkflowRunMode.ONE_CLICK.value:
                failed_action = article.supervisor_action or "生成内容"
                return self._start_action(
                    task_id,
                    "ONE_CLICK_GENERATE",
                    f"正在从失败步骤继续一键生成：{self.workflow.action_label(failed_action)}",
                    run_mode=WorkflowRunMode.ONE_CLICK,
                    display_action=self.workflow.one_click_initial_action(task_id),
                )
            action = article.supervisor_action
            if not action:
                raise ValueError("当前任务没有可重试的工作流动作")
            provider = self.workflow.provider_for_action(action)
            return self._start_action(task_id, action, f"正在重试：{self.workflow.action_label(action)}", provider=provider)

    def generate_images(self, task_id: str, provider: ImageProvider) -> WorkflowJobStatus:
        action = SupervisorAction.GENERATE_IMAGE_RESULTS.value
        action = self.workflow.encode_image_action(action, provider)
        return self._start_action(
            task_id,
            action,
            f"任务已开始：{self.workflow.action_label(action)}（{provider.value}）",
            provider=provider,
        )

    def retry_image(self, task_id: str, index: int, provider: ImageProvider) -> WorkflowJobStatus:
        action = self.workflow.encode_image_action(f"REGENERATE_IMAGE_RESULT:{index}", provider)
        return self._start_action(
            task_id,
            action,
            f"任务已开始：重试第 {index + 1} 张图片（{provider.value}）",
            provider=provider,
        )

    def one_click(self, task_id: str) -> WorkflowJobStatus:
        display_action = self.workflow.one_click_initial_action(task_id)
        return self._start_action(
            task_id,
            "ONE_CLICK_GENERATE",
            "任务已进入队列：一键生成全文",
            run_mode=WorkflowRunMode.ONE_CLICK,
            display_action=display_action,
        )

    def apply_review_revision(self, task_id: str) -> WorkflowJobStatus:
        return self._start_action(task_id, "APPLY_REVIEW_REVISION", "任务已开始：执行评审修订")

    def delete(self, task_id: str) -> bool:
        """Cancel a queued/running job and permanently remove its local history.

        ``True`` means deletion is complete now. ``False`` means an in-flight
        provider call must reach its next safe cancellation boundary first.
        """

        with self._lock:
            if task_id in self._deleting_task_ids:
                return False
            self._deleting_task_ids.add(task_id)
            future = self._futures.get(task_id)
            if future is not None:
                self._cancelling_task_ids.add(task_id)
                if future.cancel():
                    self._active_task_ids.discard(task_id)
                    self._futures.pop(task_id, None)
                    self._cancelling_task_ids.discard(task_id)
                    delete_now = True
                else:
                    delete_now = False
            else:
                article = self.workflow.get(task_id)
                if article.status in {"QUEUED", "RUNNING", "CANCELLING"}:
                    self._cancelling_task_ids.add(task_id)
                    # SSE work is not represented by a JobRunner Future. Its
                    # persisted CANCELLING state blocks new workflow starts.
                    self._deleting_task_ids.discard(task_id)
                    delete_now = False
                else:
                    delete_now = True

        try:
            if delete_now:
                return self.workflow.permanently_delete(task_id)
            self.workflow.request_cancellation(task_id)
            return False
        except Exception:
            with self._lock:
                self._cancelling_task_ids.discard(task_id)
                self._deleting_task_ids.discard(task_id)
            raise
        finally:
            if delete_now:
                with self._lock:
                    self._deleting_task_ids.discard(task_id)

    def shutdown(self) -> None:
        with self._lock:
            if getattr(self, "_shutdown", False):
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def wait_for_terminal(self, task_id: str, poll_seconds: float = 0.02):
        """Wait for the submitted runner job, not an intermediate article state.

        Individual workflow agents deliberately persist a successful article
        state after every safe boundary.  That is useful to the asynchronous
        page, but it is *not* the terminal condition for the synchronous
        compatibility API: a one-click run may still have several agents left.
        The runner-owned active set is therefore the source of truth while a
        local future exists.
        """

        while True:
            with self._lock:
                active = task_id in self._active_task_ids
            if not active:
                return self.workflow.get(task_id)
            sleep(poll_seconds)

    def _start_action(
        self,
        task_id: str,
        action: str,
        message: str | None = None,
        provider: ImageProvider = ImageProvider.AUTO,
        run_mode: WorkflowRunMode = WorkflowRunMode.STEP,
        display_action: str | None = None,
    ) -> WorkflowJobStatus:
        with self.workflow.task_write_lock(task_id):
            with self._lock:
                if task_id in self._active_task_ids or task_id in self._cancelling_task_ids or task_id in self._deleting_task_ids:
                    raise JobAlreadyRunning("任务正在执行中，请等待完成后再试")
                try:
                    self.workflow.begin_job(
                        task_id,
                        display_action or action,
                        message or f"任务已开始：{self.workflow.action_label(action)}",
                        run_mode=run_mode,
                    )
                except RuntimeError as error:
                    if self.workflow.get(task_id).status in {"QUEUED", "RUNNING"}:
                        raise JobAlreadyRunning("任务正在执行中，请等待完成后再试") from error
                    raise
                self._active_task_ids.add(task_id)
                try:
                    self._futures[task_id] = self._executor.submit(self._run, task_id, action, provider)
                except Exception as error:
                    self._active_task_ids.discard(task_id)
                    self.workflow.fail_job(task_id, error)
                    raise
        return self.status(task_id)

    def _run(self, task_id: str, action: str, provider: ImageProvider) -> None:
        try:
            with self.workflow.background_job_execution():
                self.workflow.mark_job_running(task_id)
                with self.workflow.task_write_lock(task_id):
                    if self._cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    self.workflow.execute_action(task_id, action, provider)
                    if self._cancellation_requested(task_id):
                        raise WorkflowCancellationRequested("任务已请求取消")
                    self.workflow.finish_job(task_id)
        except Exception as error:
            if not self._cancellation_requested(task_id):
                with self.workflow.background_job_execution():
                    self.workflow.fail_job(task_id, error)
        finally:
            with self._lock:
                deleting = task_id in self._cancelling_task_ids
                self._active_task_ids.discard(task_id)
                self._futures.pop(task_id, None)
            if deleting:
                self.workflow.permanently_delete(task_id)
                with self._lock:
                    self._cancelling_task_ids.discard(task_id)
                    self._deleting_task_ids.discard(task_id)

    def _cancellation_requested(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._cancelling_task_ids

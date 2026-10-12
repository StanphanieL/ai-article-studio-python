"""Deletion and SSE completion must agree on who finishes task cleanup."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event, Thread

import pytest
from test_async_jobs import ManualExecutor, build_client

from ai_article_studio.domain import WorkflowBusyError, WorkflowValidationError
from ai_article_studio.workflow import ArticleWorkflow


def _started_stream(workflow: ArticleWorkflow) -> tuple[str, Iterator[str]]:
    task_id = workflow.create("SSE 删除交接")
    titles = workflow.generate_titles(task_id)
    workflow.confirm_title(task_id, json.loads(titles.title_options)[0]["title"])
    workflow.generate_outline(task_id)
    workflow.agent.stream_content = lambda *_args: iter(["# 已完成的测试正文"])
    workflow.start_stream_content(task_id)
    stream = workflow.stream_content(task_id)
    assert next(stream).startswith("event: progress")
    assert next(stream).startswith("event: content")
    return task_id, stream


def _thread(operation: Callable[[], object]) -> tuple[Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            operation()
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=run, daemon=True)
    worker.start()
    return worker, errors


def _task_lock_available(workflow: ArticleWorkflow, task_id: str) -> bool:
    """Do not force an impossible interleaving once the two paths share a lock."""

    try:
        with workflow.task_write_lock(task_id):
            return True
    except WorkflowBusyError:
        return False


def test_delete_using_a_running_snapshot_finishes_even_if_stream_commits_done_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with build_client(tmp_path, ManualExecutor()) as client:
        workflow = client.app.state.workflow
        runner = client.app.state.job_runner
        task_id, stream = _started_stream(workflow)
        cancellation_entered, allow_cancellation = Event(), Event()
        real_request_cancellation = workflow.request_cancellation

        def delayed_cancellation(identifier: str) -> None:
            # JobRunner has classified the task as RUNNING, but its durable
            # cancellation marker has not yet been written.
            cancellation_entered.set()
            if not allow_cancellation.wait(5):
                raise TimeoutError("test did not release cancellation")
            real_request_cancellation(identifier)

        monkeypatch.setattr(workflow, "request_cancellation", delayed_cancellation)
        deleter, errors = _thread(lambda: runner.delete(task_id))
        try:
            assert cancellation_entered.wait(5), errors
            if _task_lock_available(workflow, task_id):
                # The old path permits this ordering: done becomes durable
                # before the previously accepted deletion writes CANCELLING.
                assert next(stream).startswith("event: done")
                assert workflow.get(task_id).status == "SUCCESS"
                allow_cancellation.set()
                deleter.join(timeout=5)
            else:
                # A corrected cancellation handoff can serialize before done.
                # Let it finish instead of deadlocking the test on that lock.
                allow_cancellation.set()
                deleter.join(timeout=5)
                assert not deleter.is_alive()
                assert not errors
                assert next(stream).startswith("event: fail")
        finally:
            allow_cancellation.set()
            deleter.join(timeout=5)
            stream.close()

        assert not deleter.is_alive()
        assert not errors
        with pytest.raises(WorkflowValidationError, match="文章任务不存在"):
            workflow.get(task_id)


def test_disconnect_and_delete_cannot_leave_a_cancelled_task_without_an_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with build_client(tmp_path, ManualExecutor()) as client:
        workflow = client.app.state.workflow
        runner = client.app.state.job_runner
        task_id, stream = _started_stream(workflow)
        checked_cancellation, allow_disconnect_cleanup = Event(), Event()
        real_cancellation_requested = workflow.cancellation_requested
        paused = False

        def paused_cancellation_check(identifier: str) -> bool:
            nonlocal paused
            cancelled = real_cancellation_requested(identifier)
            if not paused:
                paused = True
                assert not cancelled
                checked_cancellation.set()
                if not allow_disconnect_cleanup.wait(5):
                    raise TimeoutError("test did not release disconnect cleanup")
            return cancelled

        monkeypatch.setattr(workflow, "cancellation_requested", paused_cancellation_check)
        closer, close_errors = _thread(stream.close)
        deleter: Thread | None = None
        delete_errors: list[BaseException] = []
        try:
            assert checked_cancellation.wait(5), close_errors
            cleanup_is_unlocked = _task_lock_available(workflow, task_id)
            deleter, delete_errors = _thread(lambda: runner.delete(task_id))
            if cleanup_is_unlocked:
                # Reproduce the stale False check: deletion commits before
                # the disconnected stream decides whether to mark FAILED.
                deleter.join(timeout=5)
                assert not deleter.is_alive()
                assert not delete_errors
            allow_disconnect_cleanup.set()
        finally:
            allow_disconnect_cleanup.set()
            closer.join(timeout=5)
            if deleter is not None:
                deleter.join(timeout=5)
            if not closer.is_alive():
                stream.close()

        assert not closer.is_alive()
        assert deleter is not None and not deleter.is_alive()
        assert not close_errors
        assert not delete_errors
        with pytest.raises(WorkflowValidationError, match="文章任务不存在"):
            workflow.get(task_id)

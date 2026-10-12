"""SSE terminal events and disconnects must not strand a task's write lock."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import anyio
import pytest
from test_async_jobs import ManualExecutor, build_client
from test_stream_response import _drive_response

from ai_article_studio.web.api import stream_content as stream_response
from ai_article_studio.workflow import ArticleWorkflow


def _ready_task(workflow: ArticleWorkflow) -> str:
    task_id = workflow.create("流式正文之后生成配图")
    titles = workflow.generate_titles(task_id)
    workflow.confirm_title(task_id, json.loads(titles.title_options)[0]["title"])
    workflow.generate_outline(task_id)
    return task_id


def test_done_event_allows_image_planning_before_the_stream_is_closed(tmp_path: Path) -> None:
    """A browser stops reading at done; it must not need one more next() to unlock."""
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client, ThreadPoolExecutor(max_workers=1) as producer:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        try:
            while not producer.submit(next, stream).result(timeout=5).startswith("event: done"):
                pass
            content = workflow.get(task_id).content
            assert content
            assert workflow.job_status(task_id).status == "SUCCESS"

            submitted = client.post(f"/article/{task_id}/jobs/next")
            assert submitted.status_code == 202
            assert submitted.json()["data"]["action"] == "GENERATE_IMAGE_PROMPTS"
            statuses = client.get("/article/jobs/statuses").json()["data"]
            assert next(item for item in statuses if item["taskId"] == task_id)["state"] == "queued"
        finally:
            producer.submit(stream.close).result(timeout=5)

        # Closing the previous response must not modify the newly queued job.
        assert workflow.job_status(task_id).status == "QUEUED"
        executor.run_all()
        planned = workflow.get(task_id)
        assert planned.status == "SUCCESS"
        assert planned.content == content
        assert json.loads(planned.image_prompts)


def test_fail_event_allows_retry_before_the_stream_is_closed(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client, ThreadPoolExecutor(max_workers=1) as producer:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)

        def failed_stream(*_args: object):
            yield "# 尚未完成的正文"
            raise RuntimeError("temporary stream failure")

        workflow.agent.stream_content = failed_stream
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        try:
            while not producer.submit(next, stream).result(timeout=5).startswith("event: fail"):
                pass
            assert workflow.job_status(task_id).status == "FAILED"
            submitted = client.post(f"/article/{task_id}/jobs/retry")
            assert submitted.status_code == 202
            assert submitted.json()["data"]["action"] == "GENERATE_CONTENT"
        finally:
            producer.submit(stream.close).result(timeout=5)
        assert workflow.job_status(task_id).status == "QUEUED"


def test_sse_can_resume_and_close_on_different_worker_threads(tmp_path: Path) -> None:
    """StreamingResponse dispatches each next() separately, without thread affinity."""
    with build_client(tmp_path) as client, ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)
        workflow.agent.stream_content = lambda *_args: iter(["# 正文\n\n## 第一节\n\n内容。"])
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        try:
            assert first.submit(next, stream).result(timeout=5).startswith("event: progress")
            assert second.submit(next, stream).result(timeout=5).startswith("event: content")
            assert second.submit(next, stream).result(timeout=5).startswith("event: done")
            second.submit(stream.close).result(timeout=5)
            assert client.post(f"/article/{task_id}/jobs/next").status_code == 202
        finally:
            first.submit(stream.close).result(timeout=5)


@pytest.mark.parametrize("body_chunks_read", [0, 1])
@pytest.mark.parametrize("previous_content", [None, "# 原有正文\n\n## 原有章节\n\n应保留的内容。"])
def test_disconnect_is_retryable_even_before_first_chunk_or_during_rewrite(tmp_path: Path, body_chunks_read: int, previous_content: str | None) -> None:
    with build_client(tmp_path) as client, ThreadPoolExecutor(max_workers=1) as worker:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)
        if previous_content:
            workflow.save_content(task_id, previous_content)
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        try:
            assert worker.submit(next, stream).result(timeout=5).startswith("event: progress")
            for _ in range(body_chunks_read):
                assert worker.submit(next, stream).result(timeout=5).startswith("event: content")
        finally:
            worker.submit(stream.close).result(timeout=5)

        interrupted = workflow.get(task_id)
        assert interrupted.status == "FAILED"
        assert interrupted.content == previous_content
        assert interrupted.supervisor_action == "GENERATE_CONTENT"
        assert "连接中断" in interrupted.error_message
        assert client.post(f"/article/{task_id}/jobs/retry").status_code == 202


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
@pytest.mark.parametrize("disconnect_at", ["headers", "progress", "content", "done"])
def test_public_stream_response_releases_its_reservation_on_disconnect(tmp_path: Path, spec_version: str, disconnect_at: str) -> None:
    """Exercise the response built by the API, not only direct generator.close()."""
    with build_client(tmp_path, ManualExecutor()) as client:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)
        response = stream_response(task_id, workflow)
        anyio.run(_drive_response, response, spec_version, disconnect_at)

        article = workflow.get(task_id)
        if disconnect_at == "done":
            assert article.status == "SUCCESS"
            assert article.content
            assert client.post(f"/article/{task_id}/jobs/next").status_code == 202
        else:
            assert article.status == "FAILED"
            assert article.content is None
            assert "连接中断" in article.error_message
            assert client.post(f"/article/{task_id}/jobs/retry").status_code == 202


def test_stream_setup_failure_does_not_leave_a_running_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with build_client(tmp_path, ManualExecutor()) as client:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)

        def unavailable_agent(_article: object) -> None:
            raise RuntimeError("stream adapter unavailable")

        monkeypatch.setattr(workflow, "_agent_for", unavailable_agent)
        workflow.start_stream_content(task_id)
        events = list(workflow.stream_content(task_id))

        assert events[-1].startswith("event: fail")
        assert workflow.get(task_id).status == "FAILED"
        assert client.post(f"/article/{task_id}/jobs/retry").status_code == 202


@pytest.mark.parametrize("ending", ["done", "fail", "disconnect"])
def test_stream_finalization_waits_for_a_short_competing_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str) -> None:
    with build_client(tmp_path) as client, ThreadPoolExecutor(max_workers=1) as worker:
        workflow = client.app.state.workflow
        task_id = _ready_task(workflow)

        def provider_stream(*_args: object):
            yield "# 正文\n\n## 章节\n\n内容。"
            if ending == "fail":
                raise RuntimeError("provider failed")

        workflow.agent.stream_content = provider_stream
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        next(stream)
        next(stream)
        lock = workflow._lock_for_task(task_id)
        attempted = Event()

        class ObservedLock:
            def acquire(self, *, blocking: bool) -> bool:
                if blocking:
                    attempted.set()
                    return lock.acquire(blocking=True)
                acquired = lock.acquire(blocking=False)
                attempted.set()
                return acquired

            def release(self) -> None:
                lock.release()

        monkeypatch.setattr(workflow, "_lock_for_task", lambda _task_id: ObservedLock())
        try:
            # A competing request briefly owns the lock while checking RUNNING.
            with lock:
                future = worker.submit(stream.close) if ending == "disconnect" else worker.submit(next, stream)
                assert attempted.wait(timeout=5)
            event = future.result(timeout=5)
            if ending != "disconnect":
                assert event.startswith(f"event: {ending}")
            assert workflow.get(task_id).status == ("SUCCESS" if ending == "done" else "FAILED")
        finally:
            stream.close()

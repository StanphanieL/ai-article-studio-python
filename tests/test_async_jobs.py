from __future__ import annotations

import json
import sqlite3
from concurrent.futures import Future
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Any

from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.jobs import JobRunner
from ai_article_studio.schemas import ArticleModelConfig, ImagePromptOption, ImageProvider, ImageResultOption, QualityReviewResult
from ai_article_studio.supervisor import ArticlePhase, ArticleStatus, SupervisorAction
from ai_article_studio.workflow import WorkflowBusyError


class ManualExecutor:
    """Deterministic executor used to assert the state before background work runs."""

    def __init__(self) -> None:
        self.pending: list[tuple[Future[object], Any, tuple[object, ...], dict[str, object]]] = []
        self.shutdown_calls: list[dict[str, object]] = []

    def submit(self, fn: Any, /, *args: object, **kwargs: object) -> Future[object]:
        future: Future[object] = Future()
        self.pending.append((future, fn, args, kwargs))
        return future

    def run_all(self) -> None:
        while self.pending:
            future, fn, args, kwargs = self.pending.pop(0)
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as error:  # The runner deliberately records the failure.
                    future.set_exception(error)

    def shutdown(self, **kwargs: object) -> None:
        self.shutdown_calls.append(kwargs)
        return None


def build_client(
    tmp_path: Path,
    executor: ManualExecutor | None = None,
    clock: Any | None = None,
) -> TestClient:
    from ai_article_studio.main import create_app

    runner_executor = executor or ManualExecutor()
    create_kwargs: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / 'articles.db'}",
        "upload_dir": tmp_path / "images",
        "settings": Settings(app_mode=RuntimeMode.MOCK),
        "job_runner_factory": lambda workflow: JobRunner(workflow, executor=runner_executor),
    }
    if clock is not None:
        create_kwargs["clock"] = clock
    return TestClient(
        create_app(**create_kwargs)  # type: ignore[arg-type]
    )


def test_startup_migrates_execution_mode_for_a_preexisting_article_table(tmp_path: Path) -> None:
    """Existing SQLite workspaces gain the retry-mode marker without a reset."""

    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE article (id INTEGER PRIMARY KEY)")

    from ai_article_studio.db import create_session_factory

    create_session_factory(f"sqlite:///{database_path}")
    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(article)")}

    assert "execution_mode" in columns


def test_confirming_title_is_fast_and_waits_to_generate_outline(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "标题确认"}).json()["data"]
        title_article = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = __import__("json").loads(title_article["titleOptions"])[0]["title"]
        calls: list[str] = []

        def record_outline(*_args: object) -> list[object]:
            calls.append("outline")
            return []

        client.app.state.workflow.agent.outline = record_outline  # type: ignore[method-assign]

        confirmed = client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        assert calls == []
        continued = client.post(f"/api/article/continue-workflow/{task_id}")

    assert confirmed.status_code == 200
    assert confirmed.json()["data"]["phase"] == ArticlePhase.OUTLINE_EDITING
    assert confirmed.json()["data"]["outline"] is None
    assert calls == ["outline"]
    assert continued.json()["data"]["outline"] == "[]"


def test_outline_editing_without_outline_routes_to_generation() -> None:
    from ai_article_studio.supervisor import decide_next_action

    state = type(
        "ArticleState",
        (),
        {
            "phase": ArticlePhase.OUTLINE_EDITING,
            "status": ArticleStatus.SUCCESS,
            "selected_title": "已确认标题",
            "outline": None,
            "content": None,
            "image_prompts": None,
            "image_results": None,
            "final_markdown": None,
            "review_round": 0,
        },
    )()

    assert decide_next_action(state).action is SupervisorAction.GENERATE_OUTLINE


def test_job_start_is_immediate_rejects_duplicate_and_finishes(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "异步标题"}).json()["data"]
        started = client.post(f"/article/{task_id}/jobs/next")
        duplicate = client.post(f"/article/{task_id}/jobs/next")
        running = client.get(f"/article/{task_id}/jobs/status")
        executor.run_all()
        finished = client.get(f"/article/{task_id}/jobs/status")

    assert started.status_code == 202
    assert {key: value for key, value in started.json()["data"].items() if key not in {"startedAt", "elapsedSeconds"}} == {
        "taskId": task_id,
        "status": "QUEUED",
        "action": "GENERATE_TITLES",
        "executionMode": "STEP",
        "message": "任务已开始：生成标题候选",
        "errorMessage": None,
    }
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == 409
    assert running.json()["data"]["status"] == "QUEUED"
    assert finished.json()["data"]["status"] == "SUCCESS"
    assert finished.json()["data"]["action"] == "GENERATE_TITLES"


def test_non_javascript_continue_queues_work_instead_of_waiting_for_a_model_call(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "无脚本排队"}).json()["data"]
        response = client.post(f"/article/{task_id}/continue", follow_redirects=False)
        queued = client.get(f"/article/{task_id}/jobs/status").json()["data"]

    assert response.status_code == 303
    assert response.headers["location"] == f"/article/detail/{task_id}?step=title#title-stage"
    assert queued["status"] == "QUEUED"
    assert len(executor.pending) == 1


def test_one_click_job_is_queued_and_tracks_its_real_current_substep(tmp_path: Path) -> None:
    """One-click work must not expose its wrapper action as user-facing progress."""

    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.app.state.workflow.create(
            "一键后台状态",
            config=ArticleModelConfig(image_provider=ImageProvider.MOCK),
        )
        started = client.post(f"/article/{task_id}/jobs/one-click")
        queued = client.get(f"/article/{task_id}/jobs/status")
        executor.run_all()
        completed = client.get(f"/article/{task_id}/jobs/status")

    assert started.status_code == 202
    assert queued.json()["data"]["status"] == "QUEUED"
    assert queued.json()["data"]["action"] == "GENERATE_TITLES"
    assert queued.json()["data"]["executionMode"] == "ONE_CLICK"
    assert completed.json()["data"]["status"] == "SUCCESS"
    assert completed.json()["data"]["action"] == "COMPLETE"


def test_one_click_stays_running_between_persisted_substeps(tmp_path: Path) -> None:
    """Intermediate article artifacts must not masquerade as job completion."""

    from threading import Event

    executor = ManualExecutor()
    entered_outline = Event()
    release_outline = Event()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = workflow.create("一键中间状态", config=ArticleModelConfig(image_provider=ImageProvider.MOCK))
        original_outline = workflow.agent.outline

        def blocking_outline(*args: object):  # type: ignore[no-untyped-def]
            entered_outline.set()
            assert release_outline.wait(timeout=2)
            return original_outline(*args)

        workflow.agent.outline = blocking_outline  # type: ignore[method-assign]
        client.post(f"/article/{task_id}/jobs/one-click")
        worker = Thread(target=executor.run_all)
        worker.start()
        assert entered_outline.wait(timeout=2)
        intermediate = client.get(f"/article/{task_id}/jobs/status").json()["data"]
        release_outline.set()
        worker.join(timeout=2)
        completed = client.get(f"/article/{task_id}/jobs/status").json()["data"]

    assert intermediate["status"] == "RUNNING"
    assert intermediate["executionMode"] == "ONE_CLICK"
    assert intermediate["action"] == "GENERATE_OUTLINE"
    assert not worker.is_alive()
    assert completed["status"] == "SUCCESS"
    assert completed["executionMode"] is None


def test_one_click_content_failure_records_real_action_and_retry_resumes_to_completion(tmp_path: Path) -> None:
    """A one-click retry continues from the failed substep instead of replaying a wrapper."""

    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = workflow.create(
            "一键断点恢复",
            config=ArticleModelConfig(image_provider=ImageProvider.MOCK),
        )
        original_content = workflow.agent.content

        def fail_content(*_args: object) -> str:
            raise RuntimeError("content provider temporarily unavailable")

        workflow.agent.content = fail_content  # type: ignore[method-assign]
        client.post(f"/article/{task_id}/jobs/one-click")
        executor.run_all()
        failed = client.get(f"/article/{task_id}/jobs/status").json()["data"]
        workflow.agent.content = original_content  # type: ignore[method-assign]
        retried = client.post(f"/article/{task_id}/jobs/retry")
        executor.run_all()
        resumed = client.get(f"/api/article/{task_id}").json()["data"]

    assert failed["status"] == "FAILED"
    assert failed["action"] == "GENERATE_CONTENT"
    assert failed["executionMode"] == "ONE_CLICK"
    assert retried.status_code == 202
    assert resumed["phase"] == "COMPLETED"
    assert resumed["status"] == "SUCCESS"


def test_queued_job_is_cancelled_then_permanently_deleted_before_it_runs(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "取消队列任务"}).json()["data"]
        client.post(f"/article/{task_id}/jobs/next")
        deleted = client.post(f"/article/{task_id}/delete", headers={"Accept": "application/json"})
        executor.run_all()
        article = client.get(f"/api/article/{task_id}")
        logs = client.get(f"/api/article/agent-logs/{task_id}")

    assert deleted.status_code == 200
    assert deleted.json()["data"] is True
    assert article.status_code == 400
    assert logs.json()["data"] == []


def test_delete_gate_rejects_a_new_job_while_a_ready_task_is_being_removed(tmp_path: Path) -> None:
    from threading import Event

    executor = ManualExecutor()
    entered_delete = Event()
    release_delete = Event()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "删除启动竞态"}).json()["data"]
        original_delete = workflow.permanently_delete

        def delayed_delete(*args: object, **kwargs: object) -> bool:
            entered_delete.set()
            assert release_delete.wait(timeout=2)
            return original_delete(*args, **kwargs)

        workflow.permanently_delete = delayed_delete  # type: ignore[method-assign]
        deleter = Thread(target=lambda: client.app.state.job_runner.delete(task_id))
        deleter.start()
        assert entered_delete.wait(timeout=2)
        start = client.post(f"/article/{task_id}/jobs/next")
        release_delete.set()
        deleter.join(timeout=2)

    assert start.status_code == 409
    assert not deleter.is_alive()


def test_running_job_deletes_after_its_current_model_call_reaches_a_safe_cancellation_point(tmp_path: Path) -> None:
    from threading import Event

    executor = ManualExecutor()
    entered = Event()
    release = Event()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "取消运行任务"}).json()["data"]

        def blocking_titles(*_args: object) -> list[object]:
            entered.set()
            assert release.wait(timeout=2)
            return []

        client.app.state.workflow.agent.titles = blocking_titles  # type: ignore[method-assign]
        client.post(f"/article/{task_id}/jobs/next")
        worker = Thread(target=executor.run_all)
        worker.start()
        assert entered.wait(timeout=2)
        deleting = client.post(f"/article/{task_id}/delete", headers={"Accept": "application/json"})
        status_while_cancelling = client.get(f"/api/article/{task_id}")
        release.set()
        worker.join(timeout=2)
        deleted = client.get(f"/api/article/{task_id}")
        logs = client.get(f"/api/article/agent-logs/{task_id}")

    assert deleting.status_code == 202
    assert deleting.json()["data"] is True
    assert status_while_cancelling.json()["data"]["status"] == "CANCELLING"
    assert not worker.is_alive()
    assert deleted.status_code == 400
    assert logs.json()["data"] == []


def test_streaming_content_cancellation_never_commits_a_final_chunk_or_log(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "取消流式任务"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish streaming prerequisites.
            task_id,
            selected_title="取消流式任务",
            outline=json.dumps([{"heading": "第一节", "description": "说明"}], ensure_ascii=False),
            phase=ArticlePhase.OUTLINE_EDITING.value,
            status=ArticleStatus.SUCCESS.value,
        )
        workflow.agent.stream_content = lambda *_args: iter(["# 正文\n\n", "## 第一节\n内容"])  # type: ignore[method-assign]
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        progress = next(stream)
        cancelling = client.post(f"/article/{task_id}/delete", headers={"Accept": "application/json"})
        events = "".join(stream)
        article = client.get(f"/api/article/{task_id}")
        logs = client.get(f"/api/article/agent-logs/{task_id}")

    assert "event: progress" in progress
    assert cancelling.status_code == 202
    assert "任务已取消并永久删除" in events
    assert article.status_code == 400
    assert logs.json()["data"] == []


def test_abandoned_stream_becomes_retryable_instead_of_staying_running(tmp_path: Path) -> None:
    """Closing a browser's EventSource must not strand the task in RUNNING."""

    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "中断的流式任务"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish streaming prerequisites.
            task_id,
            selected_title="中断的流式任务",
            outline=json.dumps([{"heading": "第一节", "description": "说明"}], ensure_ascii=False),
            phase=ArticlePhase.OUTLINE_EDITING.value,
            status=ArticleStatus.SUCCESS.value,
        )
        workflow.agent.stream_content = lambda *_args: iter(["# 正文\n\n## 第一节\n内容"])  # type: ignore[method-assign]
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        next(stream)  # progress
        next(stream)  # first content chunk, then emulate an EventSource disconnect.
        stream.close()
        stranded = workflow.job_status(task_id)
        logs = client.get(f"/api/article/agent-logs/{task_id}").json()["data"]

    assert stranded.status == "FAILED"
    assert stranded.action == "GENERATE_CONTENT"
    assert "连接" in (stranded.error_message or "")
    assert logs == []


def test_failed_job_can_retry_the_persisted_supervisor_action(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "异步重试"}).json()["data"]
        original_titles = client.app.state.workflow.agent.titles

        def fail_titles(*_args: object) -> list[object]:
            raise RuntimeError("temporary model failure")

        client.app.state.workflow.agent.titles = fail_titles  # type: ignore[method-assign]
        client.post(f"/article/{task_id}/jobs/next")
        executor.run_all()
        failed = client.get(f"/article/{task_id}/jobs/status")
        client.app.state.workflow.agent.titles = original_titles
        retried = client.post(f"/article/{task_id}/jobs/retry")
        executor.run_all()
        succeeded = client.get(f"/article/{task_id}/jobs/status")

    assert failed.json()["data"]["status"] == "FAILED"
    assert failed.json()["data"]["action"] == "GENERATE_TITLES"
    assert "temporary model failure" in failed.json()["data"]["errorMessage"]
    assert retried.status_code == 202
    assert succeeded.json()["data"]["status"] == "SUCCESS"


def test_failed_one_click_job_retries_from_the_real_failed_stage_and_finishes(tmp_path: Path) -> None:
    """A one-click retry must resume the orchestration, not replay a stale sub-step."""

    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "一键失败续跑"}).json()["data"]
        original_content = workflow.agent.content

        def fail_content(*_args: object) -> str:
            raise RuntimeError("content generation failed")

        workflow.agent.content = fail_content  # type: ignore[method-assign]
        started = client.post(f"/article/{task_id}/jobs/one-click")
        executor.run_all()
        failed = client.get(f"/article/{task_id}/jobs/status").json()["data"]

        workflow.agent.content = original_content
        retried = client.post(f"/article/{task_id}/jobs/retry")
        executor.run_all()
        recovered = client.get(f"/api/article/{task_id}").json()["data"]

    assert started.status_code == 202
    assert failed["status"] == "FAILED"
    assert failed["action"] == "GENERATE_CONTENT"
    assert failed["executionMode"] == "ONE_CLICK"
    assert retried.status_code == 202
    assert recovered["status"] == "SUCCESS"
    assert recovered["phase"] == ArticlePhase.COMPLETED
    assert recovered["content"]


def test_review_revision_job_repairs_when_the_model_returns_an_unrecognized_action(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow

        def minor_review(_markdown: str) -> QualityReviewResult:
            return QualityReviewResult(
                score=94,
                passed=False,
                summary="需要细微修订",
                revision_action="minor",
                revision_reason="修正正文中的代码示例。",
            )

        # This test controls the runner with ManualExecutor.  Build the
        # completed prerequisite directly so the synchronous compatibility
        # endpoint does not wait forever for an executor the test has
        # deliberately paused.
        task_id = workflow.create(
            "评审动作兼容",
            config=ArticleModelConfig(image_provider=ImageProvider.MOCK),
        )
        with workflow.background_job_execution():
            initial_article = workflow.one_click(task_id)
        workflow._update(  # noqa: SLF001 - persists the real-model compatibility payload to repair.
            task_id,
            phase=ArticlePhase.ARTICLE_COMPOSITION.value,
            status=ArticleStatus.SUCCESS.value,
            quality_score=94,
            review_report=QualityReviewResult(
                score=94,
                passed=False,
                summary="需要细微修订",
                revision_action="minor",
                revision_reason="修正正文中的代码示例。",
            ).model_dump_json(by_alias=True),
            supervisor_action=SupervisorAction.MANUAL_REVIEW.value,
        )
        workflow.agent.review = minor_review  # type: ignore[method-assign]
        started = client.post(f"/article/{task_id}/jobs/review-revision")
        executor.run_all()
        revised = client.get(f"/api/article/{task_id}").json()["data"]

    assert initial_article.review_round == 0
    assert started.status_code == 202
    assert revised["reviewRound"] == 1
    assert revised["supervisorAction"] == "MANUAL_REVIEW"


def test_app_creation_recovers_interrupted_running_jobs(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as first_client:
        task_id = first_client.post("/api/article/create", json={"topic": "重启恢复"}).json()["data"]
        first_client.app.state.workflow._update(  # noqa: SLF001 - simulates an interrupted persisted job.
            task_id,
            status=ArticleStatus.RUNNING.value,
            supervisor_action=SupervisorAction.GENERATE_TITLES.value,
            supervisor_reason="任务已开始：生成标题候选",
        )

    with build_client(tmp_path, ManualExecutor()) as recovered_client:
        status = recovered_client.get(f"/article/{task_id}/jobs/status")

    assert {key: value for key, value in status.json()["data"].items() if key not in {"startedAt", "elapsedSeconds"}} == {
        "taskId": task_id,
        "status": "FAILED",
        "action": "GENERATE_TITLES",
        "executionMode": None,
        "message": "任务已开始：生成标题候选",
        "errorMessage": "任务在服务重启时中断，请重试此步骤。",
    }


def test_running_job_blocks_existing_mutation_routes_but_keeps_reads_available(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        task_id = client.post("/api/article/create", json={"topic": "写入互斥"}).json()["data"]
        client.post(f"/article/{task_id}/jobs/next")

        page_content = client.post(f"/article/{task_id}/content", data={"content": "# 不应写入"})
        page_continue = client.post(f"/article/{task_id}/continue")
        api_content = client.post("/api/article/save-content", json={"taskId": task_id, "content": "# 不应写入"})
        api_config = client.post(
            "/api/article/save-model-config",
            json={"taskId": task_id, "style": "BUSINESS", "config": {"imageProvider": "MOCK"}},
        )
        api_continue = client.post(f"/api/article/continue-workflow/{task_id}")
        page_images = client.post(f"/article/{task_id}/images", data={"provider": "MOCK"})
        readable = client.get(f"/api/article/{task_id}")
        job_status = client.get(f"/article/{task_id}/jobs/status")

    for response in (page_content, page_continue, api_content, api_config, api_continue, page_images):
        assert response.status_code == 409
        assert response.json()["code"] == 409
    assert readable.status_code == 200
    assert job_status.status_code == 200
    assert readable.json()["data"]["content"] is None
    assert readable.json()["data"]["style"] == "TECH"


def test_running_job_status_reports_start_and_elapsed_time_with_an_injected_clock(tmp_path: Path) -> None:
    executor = ManualExecutor()
    current_time = datetime(2026, 7, 22, 10, 0, 0)

    def clock() -> datetime:
        return current_time

    with build_client(tmp_path, executor, clock) as client:
        task_id = client.post("/api/article/create", json={"topic": "可观测进度"}).json()["data"]
        started = client.post(f"/article/{task_id}/jobs/next").json()["data"]
        current_time += timedelta(seconds=7)
        later = client.get(f"/article/{task_id}/jobs/status").json()["data"]
        executor.run_all()
        finished = client.get(f"/article/{task_id}/jobs/status").json()["data"]

    assert started["startedAt"].startswith("2026-07-22T10:00:00")
    assert started["elapsedSeconds"] == 0
    assert later["elapsedSeconds"] == 7
    assert finished["startedAt"] is None
    assert finished["elapsedSeconds"] is None


def test_workflow_updates_keep_the_application_clock_for_job_elapsed_time(tmp_path: Path) -> None:
    """SQLite's UTC on-update default must not make a fresh job look hours old."""

    current_time = datetime(2026, 7, 23, 10, 0, 0)
    with build_client(tmp_path, ManualExecutor(), clock=lambda: current_time) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "统一任务时钟"}).json()["data"]
        workflow._update(task_id, supervisor_action=SupervisorAction.GENERATE_TITLES.value)  # noqa: SLF001 - checks persistence timestamp ownership.
        updated = workflow.get(task_id)

    assert updated.update_time == current_time


def test_shutdown_waits_for_running_work_and_cancels_queued_work(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        runner = client.app.state.job_runner
        runner.shutdown()

    assert executor.shutdown_calls == [{"wait": True, "cancel_futures": True}]


def test_retry_restores_the_explicit_image_provider_but_defaults_non_image_actions_to_auto(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        image_task = client.post("/api/article/create", json={"topic": "图片供应商重试"}).json()["data"]
        prompt = ImagePromptOption(
            image_title="封面",
            usage_scene="文章封面",
            prompt_zh="科技感封面",
            prompt_en="technology cover",
        )
        workflow._update(  # noqa: SLF001 - establish the minimal image-generation state.
            image_task,
            selected_title="图片供应商重试",
            content="# 内容",
            image_prompts=json.dumps([prompt.model_dump(by_alias=True)], ensure_ascii=False),
            phase=ArticlePhase.CONTENT_GENERATION.value,
            status=ArticleStatus.SUCCESS.value,
        )
        image_providers: list[ImageProvider] = []

        def generate_one(prompt: ImagePromptOption, provider: ImageProvider, index: int, **_kwargs: object) -> ImageResultOption:
            image_providers.append(provider)
            if len(image_providers) == 1:
                raise RuntimeError("image provider temporarily unavailable")
            return [
                ImageResultOption(
                    **prompt.model_dump(),
                    image_url=f"/uploads/images/retry-{index}.svg",
                    source=provider.value,
                )
            ][0]

        workflow.images.generate_one = generate_one  # type: ignore[method-assign]
        client.post(f"/article/{image_task}/jobs/images", data={"provider": "SILICONFLOW"})
        executor.run_all()
        failed = client.get(f"/article/{image_task}/jobs/status").json()["data"]
        client.post(f"/article/{image_task}/jobs/retry")
        executor.run_all()

        title_task = client.post("/api/article/create", json={"topic": "普通动作重试"}).json()["data"]
        providers_seen_by_execute: list[ImageProvider] = []
        original_execute = workflow.execute_action

        def capture_execute(task_id: str, action: str, provider: ImageProvider = ImageProvider.AUTO):
            providers_seen_by_execute.append(provider)
            return original_execute(task_id, action, provider)

        workflow.execute_action = capture_execute  # type: ignore[method-assign]
        original_titles = workflow.agent.titles
        workflow.agent.titles = lambda *_args: (_ for _ in ()).throw(RuntimeError("title failure"))  # type: ignore[method-assign]
        client.post(f"/article/{title_task}/jobs/next")
        executor.run_all()
        workflow.agent.titles = original_titles
        client.post(f"/article/{title_task}/jobs/retry")
        executor.run_all()

    assert failed["action"] == "GENERATE_IMAGE_RESULTS"
    assert image_providers == [ImageProvider.SILICONFLOW, ImageProvider.SILICONFLOW]
    assert providers_seen_by_execute[-1] is ImageProvider.AUTO


def test_one_click_honors_an_explicit_siliconflow_setting_without_pexels_fallback(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post(
            "/api/article/create",
            json={"topic": "一键供应商语义", "config": {"imageProvider": "SILICONFLOW"}},
        ).json()["data"]
        prompt = ImagePromptOption(
            image_title="封面",
            usage_scene="文章封面",
            prompt_zh="封面",
            prompt_en="cover",
            visual_role="COVER",
            placement="COVER",
            layout="HERO",
        )
        workflow._update(  # noqa: SLF001 - start one-click at the image stage.
            task_id,
            selected_title="一键供应商语义",
            content="# 一键供应商语义\n\n## 第一节\n\n正文。",
            image_prompts=json.dumps([prompt.model_dump(by_alias=True)], ensure_ascii=False),
            phase=ArticlePhase.CONTENT_GENERATION.value,
            status=ArticleStatus.SUCCESS.value,
        )
        providers: list[ImageProvider] = []

        def generate_one(prompt: ImagePromptOption, provider: ImageProvider, index: int, **_kwargs: object) -> ImageResultOption:
            providers.append(provider)
            return ImageResultOption(**prompt.model_dump(), image_url=f"/uploads/images/{index}.svg", source=provider.value)

        workflow.images.generate_one = generate_one  # type: ignore[method-assign]
        client.app.state.job_runner.one_click(task_id)
        executor.run_all()
        completed = client.get(f"/api/article/{task_id}").json()["data"]

    assert providers == [ImageProvider.SILICONFLOW]
    assert completed["status"] == "SUCCESS"
    assert completed["phase"] == "COMPLETED"


def test_image_batch_keeps_completed_prefix_and_retry_only_generates_missing_image(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "逐张图片恢复"}).json()["data"]
        prompts = [
            ImagePromptOption(image_title="封面", usage_scene="封面", prompt_zh="封面", prompt_en="cover"),
            ImagePromptOption(image_title="图一", usage_scene="第一节", prompt_zh="图一", prompt_en="one"),
            ImagePromptOption(image_title="图二", usage_scene="第二节", prompt_zh="图二", prompt_en="two"),
        ]
        workflow._update(  # noqa: SLF001 - establish a resumable image stage.
            task_id,
            selected_title="逐张图片恢复",
            content="# 正文\n\n## 第一节\n\n正文。",
            image_prompts=json.dumps([prompt.model_dump(by_alias=True) for prompt in prompts], ensure_ascii=False),
            phase=ArticlePhase.CONTENT_GENERATION.value,
            status=ArticleStatus.SUCCESS.value,
        )
        calls: list[int] = []

        def generate_one(prompt: ImagePromptOption, provider: ImageProvider, index: int, **_kwargs: object) -> ImageResultOption:
            calls.append(index)
            if index == 2 and calls.count(2) == 1:
                raise RuntimeError("temporary image provider failure")
            return ImageResultOption(**prompt.model_dump(), image_url=f"/uploads/images/{index}.svg", source=provider.value)

        workflow.images.generate_one = generate_one  # type: ignore[method-assign]
        client.post(f"/article/{task_id}/jobs/images", data={"provider": "AUTO"})
        executor.run_all()
        failed = client.get(f"/api/article/{task_id}").json()["data"]
        client.post(f"/article/{task_id}/jobs/retry")
        executor.run_all()
        completed = client.get(f"/api/article/{task_id}").json()["data"]

    assert failed["status"] == "FAILED"
    assert len(json.loads(failed["imageResults"])) == 2
    assert calls == [0, 1, 2, 2]
    assert completed["status"] == "SUCCESS"
    assert len(json.loads(completed["imageResults"])) == 3


def test_per_task_write_lock_rejects_racing_foreground_mutation_without_blocking_other_tasks(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        locked_task = client.post("/api/article/create", json={"topic": "原子互斥"}).json()["data"]
        independent_task = client.post("/api/article/create", json={"topic": "并行任务"}).json()["data"]
        errors: list[Exception] = []

        def write_locked_task() -> None:
            try:
                workflow.save_content(locked_task, "# 不应覆盖\n\n## 小节")
            except Exception as error:
                errors.append(error)

        with workflow.task_write_lock(locked_task):
            contender = Thread(target=write_locked_task)
            contender.start()
            contender.join(timeout=2)
            independent = workflow.save_content(independent_task, "# 可并行保存\n\n## 小节")

    assert not contender.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], WorkflowBusyError)
    assert independent.content == "# 可并行保存\n\n## 小节"


def test_running_task_rejects_image_plan_metadata_edits(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "配图编辑互斥"}).json()["data"]
        prompt = ImagePromptOption(
            image_title="第一节配图",
            usage_scene="第一节说明",
            prompt_zh="流程图",
            prompt_en="workflow diagram",
            placement="AFTER_PARAGRAPH",
            anchor_heading="第一节",
            anchor_paragraph=1,
        )
        workflow._update(  # noqa: SLF001 - establish a valid editable image plan.
            task_id,
            content="# 配图编辑互斥\n\n## 第一节\n\n正文。",
            image_prompts=json.dumps([prompt.model_dump(by_alias=True)], ensure_ascii=False),
            phase=ArticlePhase.CONTENT_GENERATION.value,
            status=ArticleStatus.SUCCESS.value,
        )
        workflow.begin_job(task_id, "GENERATE_IMAGE_RESULTS", "正在生成图片")
        response = client.post(
            f"/article/{task_id}/image-plan/0",
            json=prompt.model_dump(by_alias=True),
            headers={"accept": "application/json"},
        )

    assert response.status_code == 409
    assert response.json()["code"] == 409


def test_sse_content_uses_the_same_running_lifecycle_and_releases_after_completion(tmp_path: Path) -> None:
    executor = ManualExecutor()
    with build_client(tmp_path, executor) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "流式互斥"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish the confirmed title and outline required by SSE.
            task_id,
            selected_title="流式互斥",
            outline=json.dumps([{"heading": "第一节", "description": "说明"}], ensure_ascii=False),
            phase=ArticlePhase.OUTLINE_EDITING.value,
            status=ArticleStatus.SUCCESS.value,
        )
        workflow.agent.stream_content = lambda *_args: iter(["# 流式正文\n\n## 第一节\n内容"])  # type: ignore[method-assign]

        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        progress = next(stream)
        running = workflow.job_status(task_id)
        concurrent_errors: list[Exception] = []

        def write_while_streaming() -> None:
            try:
                workflow.save_content(task_id, "# 并发写入\n\n## 小节")
            except Exception as error:
                concurrent_errors.append(error)

        contender = Thread(target=write_while_streaming)
        contender.start()
        contender.join(timeout=2)
        events = list(stream)
        completed = workflow.job_status(task_id)
        saved = workflow.save_content(task_id, "# 流结束后可保存\n\n## 小节")

    assert "event: progress" in progress
    assert running.status == "RUNNING"
    assert not contender.is_alive()
    assert len(concurrent_errors) == 1 and isinstance(concurrent_errors[0], WorkflowBusyError)
    assert any("event: done" in event for event in events)
    assert completed.status == "SUCCESS"
    assert saved.content == "# 流结束后可保存\n\n## 小节"


def test_article_api_hides_internal_image_provider_encoding(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "动作兼容"}).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - persisted internal job representation.
            task_id,
            supervisor_action="GENERATE_IMAGE_RESULTS|provider=SILICONFLOW",
        )
        payload = client.get(f"/api/article/{task_id}").json()["data"]

    assert payload["supervisorAction"] == "GENERATE_IMAGE_RESULTS"

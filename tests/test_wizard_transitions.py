from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.web.presentation import outline_preview_markdown


def build_client(tmp_path: Path) -> TestClient:
    from ai_article_studio.main import create_app

    return TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=Settings(app_mode=RuntimeMode.MOCK),
        )
    )


def test_page_saves_redirect_to_the_next_wizard_step(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "卡片跳转"}).json()["data"]
        titles = client.post(f"/api/article/generate-titles/{task_id}").json()["data"]
        selected_title = json.loads(titles["titleOptions"])[0]["title"]

        title_response = client.post(
            f"/article/{task_id}/title",
            data={"selected_title": selected_title},
            follow_redirects=False,
        )
        outline = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]["outline"]
        outline_response = client.post(
            f"/article/{task_id}/outline",
            data={"outline_markdown": outline_preview_markdown(json.loads(outline))},
            follow_redirects=False,
        )
        content_response = client.post(
            f"/article/{task_id}/content",
            data={"content": "# 卡片跳转\n\n## 正文\n\n内容。"},
            follow_redirects=False,
        )

    assert title_response.headers["location"] == f"/article/detail/{task_id}?step=title#title-stage"
    assert outline_response.headers["location"] == f"/article/detail/{task_id}?step=content#content-stage"
    assert content_response.headers["location"] == f"/article/detail/{task_id}?step=images#images-stage"


def test_stream_script_refreshes_the_workspace_after_a_successful_content_stream() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")

    assert 'loadWizardStep("images", "replace", backgroundRefreshOptions)' in script
    assert 'searchParams.set("step", step)' in script
    assert "url.hash = `${step}-stage`" in script


def test_stream_script_closes_stale_connections_and_does_not_retry_forever_after_network_failure() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")

    assert "let activeStream = null;" in script
    assert "closeActiveStream(current);" in script
    assert "const streamIsCurrent" in script
    assert 'restore("连接中断，未保存不完整内容；请重试。")' in script
    assert "source.readyState === EventSource.CLOSED" not in script


def test_job_polling_treats_cancellation_as_active_and_bypasses_stale_cache() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")

    assert '["QUEUED", "RUNNING", "CANCELLING"].includes(status)' in script
    assert 'cache: "no-store"' in script
    assert 'window.location.assign("/article/list")' in script


def test_one_click_polling_moves_the_workspace_to_the_current_background_stage() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")

    assert "const workflowStepForAction" in script
    assert "if (targetStep && targetStep !== activeStep)" in script
    assert 'await loadWizardStep(targetStep, "replace", backgroundRefreshOptions)' in script


def test_wizard_navigation_replaces_the_workspace_without_a_full_document_reload() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")

    assert "const loadWorkspace = async" in script
    assert 'querySelector("main.page-shell")' in script
    assert "window.history.pushState" in script
    assert 'window.addEventListener("popstate' in script
    assert 'if (status === "SUCCESS") {' in script
    assert 'await loadWorkspace(canonicalDetailUrl(panel, job), "replace", backgroundRefreshOptions)' in script


def test_background_workspace_refresh_is_quiet_and_keeps_the_reader_position() -> None:
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")
    stylesheet = (Path(__file__).parents[1] / "src/ai_article_studio/static/app.css").read_text(encoding="utf-8")

    assert "const backgroundRefreshOptions = Object.freeze({ animate: false, preserveScroll: true, scroll: false });" in script
    assert "const replaceWorkspace = (current, next, { animate = true } = {})" in script
    assert "const scrollPosition = preserveScroll ? { x: window.scrollX, y: window.scrollY } : null;" in script
    assert "window.requestAnimationFrame(() => window.scrollTo(scrollPosition.x, scrollPosition.y));" in script
    assert ".wizard-next-action [data-job-progress][hidden]{display:block!important;visibility:hidden}" in stylesheet


def test_wizard_navigation_uses_stage_anchors_and_reserves_space_for_the_sticky_progress_bar() -> None:
    template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/detail.html").read_text(encoding="utf-8")
    stylesheet = (Path(__file__).parents[1] / "src/ai_article_studio/static/app.css").read_text(encoding="utf-8")

    assert "?step={{ step.key }}#{{ step.key }}-stage" in template
    assert "?step={{ step_id }}#{{ step_id }}-stage" in template
    assert "?step={{ step_order[active_index - 1] }}#{{ step_order[active_index - 1] }}-stage" in template
    assert 'id="images-stage"' in template
    assert ".wizard-stage{scroll-margin-top:" in stylesheet


def test_wizard_assets_have_new_cache_versions() -> None:
    base = (Path(__file__).parents[1] / "src/ai_article_studio/templates/base.html").read_text(encoding="utf-8")

    assert "/static/app.css?v=20260911-task-flow1" in base
    assert "/static/stream.js?v=20260911-task-flow1" in base


def test_reconfirming_a_completed_title_stays_on_the_delivery_card(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "同标题确认", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        response = client.post(
            f"/article/{article['taskId']}/title",
            data={"selected_title": article["selectedTitle"]},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == f"/article/detail/{article['taskId']}?step=delivery#delivery-stage"


def test_invalid_page_step_redirects_to_the_resolved_canonical_step(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "规范 URL", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        response = client.get(f"/article/detail/{article['taskId']}?step=not-a-step", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == f"/article/detail/{article['taskId']}?step=delivery#delivery-stage"

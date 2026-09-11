from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.web import api, pages
from ai_article_studio.web.presentation import article_payload as web_article_payload
from ai_article_studio.web.presentation import log_payload as web_log_payload

PAGE_ROUTE_INVENTORY = {
    ("GET", "/"),
    ("GET", "/article/list"),
    ("POST", "/article/{task_id}/delete"),
    ("POST", "/article/create"),
    ("GET", "/article/detail/{task_id}"),
    ("POST", "/article/{task_id}/continue"),
    ("POST", "/article/{task_id}/one-click"),
    ("POST", "/article/{task_id}/review-revision"),
    ("POST", "/article/{task_id}/title"),
    ("POST", "/article/{task_id}/outline"),
    ("POST", "/article/{task_id}/content"),
    ("POST", "/article/{task_id}/image-plan/reorder"),
    ("POST", "/article/{task_id}/image-plan/{index}"),
    ("POST", "/article/{task_id}/config"),
    ("POST", "/article/{task_id}/image/{index}/retry"),
    ("POST", "/article/{task_id}/images"),
    ("GET", "/article/{task_id}/jobs/status"),
    ("POST", "/article/{task_id}/jobs/next"),
    ("POST", "/article/{task_id}/jobs/retry"),
    ("POST", "/article/{task_id}/jobs/images"),
    ("POST", "/article/{task_id}/jobs/image/{index}/retry"),
    ("POST", "/article/{task_id}/jobs/one-click"),
    ("POST", "/article/{task_id}/jobs/review-revision"),
}

API_ROUTE_INVENTORY = {
    ("POST", "/api/article/create"),
    ("POST", "/api/article/one-click-generate"),
    ("POST", "/api/article/one-click-generate/{task_id}"),
    ("GET", "/api/article/{task_id}"),
    ("POST", "/api/article/list"),
    ("POST", "/api/article/delete/{task_id}"),
    ("POST", "/api/article/generate-titles/{task_id}"),
    ("POST", "/api/article/confirm-title"),
    ("POST", "/api/article/save-title"),
    ("POST", "/api/article/save-outline"),
    ("POST", "/api/article/generate-content/{task_id}"),
    ("GET", "/api/article/real-stream-generate-content/{task_id}"),
    ("POST", "/api/article/save-content"),
    ("POST", "/api/article/generate-image-prompts/{task_id}"),
    ("POST", "/api/article/generate-image-results/{task_id}"),
    ("POST", "/api/article/regenerate-image-result/{task_id}"),
    ("POST", "/api/article/compose/{task_id}"),
    ("POST", "/api/article/continue-workflow/{task_id}"),
    ("POST", "/api/article/apply-review-revision/{task_id}"),
    ("POST", "/api/article/save-model-config"),
    ("GET", "/api/article/agent-logs/{task_id}"),
    ("GET", "/api/article/export-html/{task_id}"),
    ("GET", "/api/article/export-markdown/{task_id}"),
    ("GET", "/api/article/export-zip/{task_id}"),
}


def build_client(tmp_path: Path) -> TestClient:
    from ai_article_studio.main import create_app

    return TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=Settings(app_mode=RuntimeMode.MOCK),
        )
    )


def test_web_routers_preserve_representative_page_and_api_contracts(tmp_path: Path) -> None:
    """Catch route registration drifting back into the composition root."""

    with build_client(tmp_path) as client:
        routers = [getattr(route, "original_router", None) for route in client.app.routes]

        page = client.post("/article/create", data={"topic": "Web route parity"}, follow_redirects=False)
        created = client.post("/api/article/create", json={"topic": "API route parity"})
        invalid = client.post("/api/article/confirm-title", json={"taskId": "missing"})

    assert pages.router in routers
    assert api.router in routers
    assert page.status_code == 303
    assert page.headers["location"].startswith("/article/detail/")
    assert created.status_code == 200
    assert set(created.json()) == {"code", "message", "data"}
    assert isinstance(created.json()["data"], str)
    assert invalid.status_code == 422
    assert invalid.json()["code"] == 422
    assert invalid.json()["message"] == "请求参数校验失败"


def test_main_preserves_serialization_imports_and_full_route_inventory() -> None:
    """Catch compatibility imports or any pre-existing article route being dropped."""

    from ai_article_studio.main import article_payload, log_payload

    page_routes = {(next(iter(route.methods)), route.path) for route in pages.router.routes}
    api_routes = {(next(iter(route.methods)), route.path) for route in api.router.routes}

    assert article_payload is web_article_payload
    assert log_payload is web_log_payload
    assert page_routes == PAGE_ROUTE_INVENTORY
    assert api_routes == API_ROUTE_INVENTORY


def test_web_static_export_and_sse_contracts(tmp_path: Path) -> None:
    """Catch delivery regressions outside normal JSON endpoint payloads."""

    with build_client(tmp_path) as client:
        static = client.get("/static/app.css")
        generated = client.post(
            "/api/article/one-click-generate",
            json={"topic": "Export contract", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        exported = client.get(f"/api/article/export-markdown/{generated['taskId']}")

        stream_task = client.post("/api/article/create", json={"topic": "SSE contract"}).json()["data"]
        titled = client.post(f"/api/article/continue-workflow/{stream_task}").json()["data"]
        selected_title = __import__("json").loads(titled["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": stream_task, "selectedTitle": selected_title})
        client.post(f"/api/article/continue-workflow/{stream_task}")
        stream = client.get(f"/api/article/real-stream-generate-content/{stream_task}")

    assert static.status_code == 200
    assert static.headers["content-type"].startswith("text/css")
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].startswith("attachment; filename=\"article.md\"; filename*=UTF-8''")
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")

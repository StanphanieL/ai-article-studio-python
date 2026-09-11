from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.web import api, pages


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

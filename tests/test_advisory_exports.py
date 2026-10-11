"""A composed draft is exportable regardless of its optional quality review."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.llm import MockArticleAgent
from ai_article_studio.main import create_app
from ai_article_studio.schemas import QualityReviewResult


class BlockingReviewAgent(MockArticleAgent):
    def review(self, article: str) -> QualityReviewResult:
        return QualityReviewResult(
            score=42,
            passed=False,
            summary="作者需要核查事实",
            issues=["有待核查的事实"],
            blocking_issues=["缺少事实来源"],
        )


class FailingReviewAgent(MockArticleAgent):
    def review(self, article: str) -> QualityReviewResult:
        raise RuntimeError("评审模型暂时不可用")


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=Settings(app_mode=RuntimeMode.MOCK),
        ),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize("review_state", ["unreviewed", "blocked", "review_error"])
def test_composed_draft_exports_all_formats_without_passing_review(tmp_path: Path, review_state: str) -> None:
    """A review gate or an export-side write would break this author-owned draft contract."""
    with _client(tmp_path) as client:
        created = client.post(
            "/api/article/one-click-generate",
            json={"topic": "可选评审的文章交付", "config": {"imageProvider": "MOCK"}},
        )
        assert created.status_code == 200
        composed = created.json()["data"]
        task_id = composed["taskId"]
        assert composed["finalMarkdown"]
        assert composed["reviewReport"] is None

        if review_state == "blocked":
            client.app.state.workflow.agent = BlockingReviewAgent()
        elif review_state == "review_error":
            client.app.state.workflow.agent = FailingReviewAgent()

        if review_state != "unreviewed":
            reviewed = client.post(f"/api/article/continue-workflow/{task_id}")
            if review_state == "blocked":
                assert reviewed.status_code == 200
                report = json.loads(reviewed.json()["data"]["reviewReport"])
                assert report["passed"] is False
                assert "缺少事实来源" in report["blockingIssues"]
            else:
                assert reviewed.status_code == 500

        before_export = client.get(f"/api/article/{task_id}").json()["data"]
        markdown = client.get(f"/api/article/export-markdown/{task_id}")
        html = client.get(f"/api/article/export-html/{task_id}")
        archive = client.get(f"/api/article/export-zip/{task_id}")
        after_export = client.get(f"/api/article/{task_id}").json()["data"]

    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "可选评审的文章交付" in markdown.text
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "可选评审的文章交付" in html.text
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/zip")
    with ZipFile(BytesIO(archive.content)) as package:
        assert {"article.md", "article.html", "metadata.json"} <= set(package.namelist())
        assert "可选评审的文章交付" in package.read("article.md").decode()
        assert "可选评审的文章交付" in package.read("article.html").decode()
        assert json.loads(package.read("metadata.json"))["imageCount"] > 0

    assert before_export["finalMarkdown"] == composed["finalMarkdown"]
    for field in ("phase", "status", "content", "finalMarkdown", "reviewReport", "qualityScore", "reviewRound"):
        assert after_export[field] == before_export[field], field
    assert after_export["reviewRound"] == 0
    if review_state == "unreviewed":
        assert after_export["reviewReport"] is None
    elif review_state == "blocked":
        assert json.loads(after_export["reviewReport"])["passed"] is False
    else:
        assert after_export["reviewReport"] is None


@pytest.mark.parametrize("format_name", ["markdown", "html", "zip"])
def test_export_rejects_a_task_without_composed_final_markdown(tmp_path: Path, format_name: str) -> None:
    """An absent final draft, unlike an absent review, must block downloading."""
    with _client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "尚未合成的文章"}).json()["data"]
        response = client.get(f"/api/article/export-{format_name}/{task_id}")

    assert response.status_code == 400
    assert response.json()["code"] == 400
    assert "完成图文稿" in response.json()["message"]

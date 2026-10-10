"""Cross-boundary Mock workflows covering extraction-sensitive checkpoints."""

import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from test_api_workflow import build_client

from ai_article_studio.domain import WorkflowValidationError


def test_stream_checkpoint_restart_composition_export_and_delete(tmp_path):
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = workflow.create("工作流持久化边界")
        titles = workflow.generate_titles(task_id)
        workflow.confirm_title(task_id, json.loads(titles.title_options)[0]["title"])
        workflow.generate_outline(task_id)
        response = client.get(f"/api/article/real-stream-generate-content/{task_id}")
        assert response.status_code == 200
        events = [part.splitlines()[0] for part in response.text.strip().split("\n\n")]
        assert events[0] == "event: progress"
        assert "event: content" in events
        assert events[-1] == "event: done"
        content = workflow.get(task_id).content
        workflow.begin_job(task_id, "GENERATE_IMAGE_PROMPTS", "pending restart")

    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        recovered = workflow.get(task_id)
        assert recovered.status == "FAILED"
        assert recovered.supervisor_action == "GENERATE_IMAGE_PROMPTS"
        assert recovered.content == content
        composed = client.post(f"/api/article/one-click-generate/{task_id}")
        assert composed.status_code == 200
        assert composed.json()["data"]["phase"] == "ARTICLE_COMPOSITION"
        assert composed.json()["data"]["status"] == "SUCCESS"
        assert composed.json()["data"]["finalMarkdown"]
        assert composed.json()["data"]["reviewReport"] is None
        archive = client.get(f"/api/article/export-zip/{task_id}")
        assert archive.status_code == 200
        with ZipFile(BytesIO(archive.content)) as exported:
            assert any(name.endswith(".md") for name in exported.namelist())
            assert any(name.endswith(".svg") for name in exported.namelist())
        assert workflow.logs(task_id)
        assert list((tmp_path / "images" / task_id).glob("*.svg"))
        deleted = client.post(f"/article/{task_id}/delete", headers={"Accept": "application/json"})
        assert deleted.status_code == 200
        assert deleted.json()["data"] is True
        assert workflow.logs(task_id) == []
        assert not (tmp_path / "images" / task_id).exists()
        assert client.get(f"/api/article/{task_id}").status_code == 400


def test_cancelled_stream_removes_existing_assets_and_never_exports_partial_content(tmp_path):
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        completed = client.post(
            "/api/article/one-click-generate",
            json={"topic": "取消后的资产清理", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        task_id = completed["taskId"]
        assert list((tmp_path / "images" / task_id).glob("*.svg"))
        workflow.start_stream_content(task_id)
        stream = workflow.stream_content(task_id)
        assert next(stream).startswith("event: progress")
        assert next(stream).startswith("event: content")
        workflow.request_cancellation(task_id)

        remaining = "".join(stream)

        assert "event: done" not in remaining
        assert "event: fail" in remaining
        assert "任务已取消并永久删除" in remaining
        assert workflow.logs(task_id) == []
        assert not (tmp_path / "images" / task_id).exists()
        for export in (workflow.export_html, workflow.export_markdown, workflow.export_zip):
            with pytest.raises(WorkflowValidationError, match="文章任务不存在"):
                export(task_id)

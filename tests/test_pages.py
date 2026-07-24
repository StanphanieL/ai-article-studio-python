from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import select

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.models import AgentLog, Article
from ai_article_studio.schemas import ImageProvider


def build_client(tmp_path: Path) -> TestClient:
    from ai_article_studio.main import create_app

    return TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=Settings(app_mode=RuntimeMode.MOCK),
        )
    )


def test_task_list_detail_and_static_image_smoke(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "页面冒烟测试"}).json()["data"]
        initial_detail = client.get(f"/article/detail/{task_id}")
        advanced = client.post(f"/article/{task_id}/continue", follow_redirects=False)
        generated = client.post(
            "/api/article/one-click-generate",
            json={"topic": "有图片的页面", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        list_page = client.get("/article/list")
        detail_page = client.get(f"/article/detail/{generated['taskId']}")
        image_url = json.loads(generated["imageResults"])[0]["imageUrl"]
        image = client.get(image_url)

    assert initial_detail.status_code == 200
    assert "导出 Markdown" not in initial_detail.text
    assert "当前为演示模式" not in initial_detail.text
    assert "真实模型已启用" not in initial_detail.text
    assert "Agent 日志" not in detail_page.text
    assert "当前创作设置" not in detail_page.text
    assert advanced.status_code == 303
    assert advanced.headers["location"] == f"/article/detail/{task_id}?step=title#title-stage"
    assert list_page.status_code == 200 and "AI Article Studio" in list_page.text
    assert "<title>任务列表 · AI Article Studio</title>" in list_page.text
    assert "AI Article Studio <span>Python</span>" not in list_page.text
    assert "单用户本地工作台" not in list_page.text
    assert "ARTICLE PRODUCTION WORKBENCH" not in list_page.text
    assert "RECENT TASKS" not in list_page.text
    assert generated["phase"] not in list_page.text
    assert detail_page.status_code == 200
    assert "流式重写正文" not in detail_page.text
    assert "htmx.org" in detail_page.text
    assert image.status_code == 200 and image.headers["content-type"].startswith("image/svg+xml")


def test_detail_page_is_not_cached_and_tolerates_a_legacy_invalid_review_payload(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "历史评审兼容"}).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - simulates a payload written before schema tightening.
            task_id,
            review_report=json.dumps(
                {"score": 80, "passed": False, "summary": "旧记录", "issues": [{"detail": "不是字符串"}]},
                ensure_ascii=False,
            ),
        )
        page = client.get(f"/article/detail/{task_id}")

    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert "历史评审兼容" in page.text


def test_content_generation_cta_uses_sse_by_default_without_a_duplicate_stream_button(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "默认流式正文"}).json()["data"]
        titled = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titled["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        outlined = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        client.post(f"/article/{task_id}/outline", data={"outline_json": outlined["outline"]})
        page = client.get(f"/article/detail/{task_id}")

    assert 'data-stream-url="/api/article/real-stream-generate-content/' + task_id + '"' in page.text
    assert 'data-content-target="#content-editor"' in page.text
    assert 'data-progress-target="#stream-progress"' in page.text
    assert ">生成正文</button>" in page.text
    assert "流式生成正文" not in page.text
    assert "流式重写正文" not in page.text
    assert f'data-job-url="/article/{task_id}/jobs/next"' not in page.text


def test_one_click_outline_to_content_transition_is_not_redirected_back_to_outline(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "一键大纲跳转"}).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - models the persisted handoff between one-click substeps.
            task_id,
            selected_title="一键大纲跳转",
            outline=json.dumps([{"heading": "第一节", "description": "说明"}], ensure_ascii=False),
            phase="OUTLINE_EDITING",
            status="RUNNING",
            execution_mode="ONE_CLICK",
            supervisor_action="GENERATE_CONTENT",
        )
        page = client.get(f"/article/detail/{task_id}?step=content", follow_redirects=False)

    assert page.status_code == 200
    assert 'data-wizard-workspace data-active-step="content"' in page.text


def test_image_plan_form_edit_redirects_to_the_images_card(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "页面配图编排", "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3}},
        ).json()["data"]
        prompt = json.loads(article["imagePrompts"])[1]
        response = client.post(
            f"/article/{article['taskId']}/image-plan/1",
            data={**prompt, "layout": "COMPACT"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == f"/article/detail/{article['taskId']}?step=images#images-stage"


def test_image_stage_defaults_to_a_compact_delivery_view_with_collapsed_advanced_editing(tmp_path: Path) -> None:
    """Prompt internals stay available without dominating the image workspace."""

    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={
                "topic": "配图故事板",
                "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3},
            },
        ).json()["data"]
        page = client.get(f"/article/detail/{article['taskId']}?step=images")

    assert 'data-image-storyboard' in page.text
    assert 'data-image-plan-card' in page.text
    assert 'data-image-plan-form' in page.text
    assert '<details class="image-plan-editor">' in page.text
    assert '<summary>编辑配图</summary>' in page.text
    assert 'class="image-placement-summary"' in page.text
    assert 'class="image-source-summary"' in page.text
    assert 'name="visualRole"' in page.text
    assert 'name="placement"' in page.text
    assert 'name="anchorHeading"' in page.text
    assert 'name="anchorParagraph"' in page.text
    assert 'name="layout"' in page.text
    assert 'name="stockQuery"' in page.text
    assert 'data-image-move="-1"' in page.text
    assert 'data-image-move="1"' in page.text
    assert 'data-image-fixed="true"' in page.text
    assert 'draggable="false"' in page.text
    assert '<details class="image-generation-details">' not in page.text
    assert 'data-image-visual-review' not in page.text


def test_image_storyboard_script_persists_drag_or_keyboard_reordering() -> None:
    script = (Path(__file__).parents[1] / "src" / "ai_article_studio" / "static" / "stream.js").read_text(encoding="utf-8")

    assert 'data-image-plan-form' in script
    assert 'data-image-storyboard' in script
    assert 'dragstart' in script
    assert 'drop' in script
    assert 'data-image-move' in script
    assert 'isFixedCard' in script
    assert 'dataset.reorderUrl' in script


def test_image_storyboard_styles_vary_figure_layouts_and_work_on_small_screens() -> None:
    stylesheet = (Path(__file__).parents[1] / "src" / "ai_article_studio" / "static" / "app.css").read_text(encoding="utf-8")

    assert ".image-storyboard" in stylesheet
    assert ".storyboard-card" in stylesheet
    assert ".article-preview .article-figure.layout-hero" in stylesheet
    assert ".article-preview .article-figure.layout-wide" in stylesheet
    assert ".article-preview .article-figure.layout-compact" in stylesheet


def test_detail_template_defines_a_five_step_card_wizard_shell() -> None:
    """The detail UI has one navigable work card instead of a long phase stack."""

    template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/detail.html").read_text(encoding="utf-8")

    assert 'data-wizard-progress' in template
    assert 'data-step-card="title"' in template
    assert 'data-step-card="outline"' in template
    assert 'data-step-card="content"' in template
    assert 'data-step-card="images"' in template
    assert 'data-step-card="delivery"' in template
    assert 'data-change-title' in template
    assert 'class="workspace-tools"' not in template


def test_wizard_styles_keep_progress_visible_and_fit_small_screens() -> None:
    stylesheet = (Path(__file__).parents[1] / "src/ai_article_studio/static/app.css").read_text(encoding="utf-8")

    assert ".wizard-progress{position:sticky" in stylesheet
    assert ".wizard-progress-step.locked" in stylesheet
    assert ".wizard-card-navigation" in stylesheet
    assert "@media(max-width:520px){.wizard-progress" in stylesheet


def test_creation_page_owns_settings_without_repeating_implementation_metadata_in_detail() -> None:
    list_template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/list.html").read_text(encoding="utf-8")
    detail_template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/detail.html").read_text(encoding="utf-8")
    stylesheet = (Path(__file__).parents[1] / "src/ai_article_studio/static/app.css").read_text(encoding="utf-8")

    assert 'id="new-article"' in list_template
    assert '<details class="create-settings" aria-label="创作设置">' in list_template
    assert '<summary>创作设置</summary>' in list_template
    assert "将随这篇文章保存" not in list_template
    assert 'name="text_model"' in list_template
    assert 'name="content_length_level"' in list_template
    assert 'name="image_provider"' in list_template
    assert 'name="image_count_mode" data-image-count-mode' in list_template
    assert '<details class="create-advanced-settings">' in list_template
    assert '<details class="config-card config-card-readonly">' not in detail_template
    assert "当前创作设置" not in detail_template
    assert 'action="/article/{{ article.task_id }}/config"' not in detail_template
    assert "保存配置并重置" not in detail_template
    assert ".create-settings-grid" in stylesheet
    assert ".create-advanced-settings" in stylesheet


def test_homepage_is_a_create_first_editorial_surface_without_changing_form_contracts() -> None:
    template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/list.html").read_text(encoding="utf-8")
    stylesheet = (Path(__file__).parents[1] / "src/ai_article_studio/static/app.css").read_text(encoding="utf-8")

    assert 'class="home-page"' in template
    assert 'class="home-hero"' in template
    assert '<h1 id="home-title"><span class="home-title-lead">把一个想法</span><span class="home-title-main">写成完整图文稿</span></h1>' in template
    assert '完整图文稿。' not in template
    assert 'class="home-flow" aria-label="创作流程"' in template
    assert 'class="create-card home-create-card"' in template
    assert 'class="task-grid home-task-grid"' in template
    assert 'class="task-card home-task-card"' in template
    assert 'class="task-card-topic"' in template
    assert 'name="mode" value="step"' in template
    assert 'name="mode" value="one_click"' in template
    assert 'data-create-form' in template
    assert 'data-delete-form' in template
    assert ".home-hero" in stylesheet
    assert ".home-title-lead" in stylesheet
    assert ".home-title-main" in stylesheet
    assert ".home-create-card" in stylesheet
    assert ".home-task-card" in stylesheet
    assert ".home-empty" in stylesheet


def test_page_creation_persists_selected_settings_for_step_and_one_click_modes(tmp_path: Path) -> None:
    settings_form = {
        "topic": "首页配置",
        "style": "POPULAR",
        "text_model": "qwen-plus",
        "temperature": "0.4",
        "max_completion_tokens": "4096",
        "content_length_level": "LONG",
        "image_provider": "MOCK",
        "image_count_mode": "CUSTOM",
        "image_count": "4",
    }
    with build_client(tmp_path) as client:
        list_page = client.get("/article/list")
        stepped = client.post("/article/create", data={**settings_form, "mode": "step"}, follow_redirects=False)
        task_id = stepped.headers["location"].rsplit("/", 1)[-1]
        task = client.get(f"/api/article/{task_id}").json()["data"]
        one_click = client.post(
            "/article/create",
            data={**settings_form, "topic": "首页一键配置", "mode": "one_click"},
            follow_redirects=False,
        )
        one_click_id = one_click.headers["location"].rsplit("/", 1)[-1]
        one_click_task = client.get(f"/api/article/{one_click_id}").json()["data"]

    persisted = json.loads(task["modelConfig"])
    one_click_persisted = json.loads(one_click_task["modelConfig"])

    assert '<details class="create-advanced-settings">' in list_page.text
    assert stepped.status_code == 303
    assert task["style"] == "POPULAR"
    assert persisted == {
        "textModel": "qwen-plus",
        "temperature": 0.4,
        "maxCompletionTokens": 4096,
        "contentLengthLevel": "LONG",
        "imageProvider": "MOCK",
        "imageCountMode": "CUSTOM",
        "imageCount": 4,
    }
    assert one_click_task["executionMode"] in {"ONE_CLICK", None}
    assert one_click_persisted == persisted


def test_legacy_page_creation_without_settings_inherits_the_environment_model(tmp_path: Path) -> None:
    from ai_article_studio.main import create_app

    settings = Settings(app_mode=RuntimeMode.MOCK, openai_model="qwen-plus")
    with TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=settings,
        )
    ) as client:
        created = client.post(
            "/article/create",
            data={"topic": "旧页面提交", "style": "TECH", "mode": "step"},
            follow_redirects=False,
        )
        task_id = created.headers["location"].rsplit("/", 1)[-1]
        article = client.get(f"/api/article/{task_id}").json()["data"]

    assert json.loads(article["modelConfig"])["textModel"] == "qwen-plus"


def test_page_delete_permanently_removes_task_logs_and_local_images(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "删除页面任务", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        task_id = article["taskId"]
        image_paths = [
            tmp_path / "images" / image["imageUrl"].removeprefix("/uploads/images/")
            for image in json.loads(article["imageResults"])
        ]
        assert all(path.exists() for path in image_paths)
        deleted = client.post(
            f"/article/{task_id}/delete",
            headers={"Accept": "application/json"},
        )
        list_page = client.get("/article/list")
        api_list = client.post("/api/article/list", json={"pageNo": 1, "pageSize": 10})
        detail = client.get(f"/article/detail/{task_id}")
        logs = client.get(f"/api/article/agent-logs/{task_id}")
        with client.app.state.workflow.sessions() as session:
            persisted_article = session.scalar(select(Article).where(Article.task_id == task_id))
            persisted_logs = list(session.scalars(select(AgentLog).where(AgentLog.task_id == task_id)))

    assert deleted.status_code == 200
    assert deleted.json()["data"] is True
    assert "删除页面任务" not in list_page.text
    assert api_list.json()["data"]["total"] == 0
    assert detail.status_code == 400
    assert logs.json()["data"] == []
    assert persisted_article is None
    assert persisted_logs == []
    assert not any(path.exists() for path in image_paths)


def test_page_delete_keeps_non_javascript_redirect_fallback(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "表单删除任务"}).json()["data"]
        deleted = client.post(f"/article/{task_id}/delete", follow_redirects=False)

    assert deleted.status_code == 303
    assert deleted.headers["location"] == "/article/list"


def test_permanent_delete_keeps_shared_and_outside_local_files_safe(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        first = client.post("/api/article/create", json={"topic": "共享图片一"}).json()["data"]
        second = client.post("/api/article/create", json={"topic": "共享图片二"}).json()["data"]
        shared = tmp_path / "images" / "shared.svg"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text("shared", encoding="utf-8")
        outside = tmp_path / "outside.svg"
        outside.write_text("outside", encoding="utf-8")
        shared_results = json.dumps([{"imageUrl": "/uploads/images/shared.svg"}], ensure_ascii=False)
        workflow._update(first, image_results=shared_results)  # noqa: SLF001 - cleanup ownership fixture.
        workflow._update(second, image_results=shared_results)  # noqa: SLF001 - cleanup ownership fixture.
        workflow._update(  # noqa: SLF001 - path traversal must never become a delete target.
            first,
            image_results=json.dumps(
                [{"imageUrl": "/uploads/images/shared.svg"}, {"imageUrl": "/uploads/images/../../outside.svg"}],
                ensure_ascii=False,
            ),
        )

        deleted = client.post(f"/article/{first}/delete", headers={"Accept": "application/json"})

    assert deleted.status_code == 200
    assert shared.exists()
    assert outside.exists()


def test_startup_purges_legacy_soft_deleted_records_and_their_assets(tmp_path: Path) -> None:
    with build_client(tmp_path) as first_client:
        article = first_client.post(
            "/api/article/one-click-generate",
            json={"topic": "旧软删除记录", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        task_id = article["taskId"]
        asset_dir = tmp_path / "images" / task_id
        assert asset_dir.is_dir()
        with first_client.app.state.workflow.sessions() as session:
            session.scalar(select(Article).where(Article.task_id == task_id)).is_deleted = 1  # type: ignore[union-attr]
            session.commit()

    with build_client(tmp_path) as restarted_client:
        deleted = restarted_client.get(f"/api/article/{task_id}")
        logs = restarted_client.get(f"/api/article/agent-logs/{task_id}")
        with restarted_client.app.state.workflow.sessions() as session:
            record = session.scalar(select(Article).where(Article.task_id == task_id))

    assert deleted.status_code == 400
    assert logs.json()["data"] == []
    assert record is None
    assert not asset_dir.exists()


def test_running_task_offers_irreversible_cancellation_and_permanent_deletion(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "运行中删除"}).json()["data"]
        client.app.state.workflow.begin_job(task_id, "GENERATE_TITLES", "正在生成标题候选")
        list_page = client.get("/article/list")
        deleted = client.post(
            f"/article/{task_id}/delete",
            headers={"Accept": "application/json"},
        )

    assert 'data-task-id="' + task_id + '"' in list_page.text
    assert 'data-delete-form' in list_page.text
    assert 'data-delete-disabled' not in list_page.text
    assert "取消生成并清除文章" in list_page.text
    assert "无法恢复" in list_page.text
    assert deleted.status_code == 202
    assert deleted.json()["data"] is True


def test_task_list_maps_running_completed_waiting_and_attention_states(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        running = client.post("/api/article/create", json={"topic": "进行中"}).json()["data"]
        completed = client.post("/api/article/create", json={"topic": "完成"}).json()["data"]
        waiting = client.post("/api/article/create", json={"topic": "待继续"}).json()["data"]
        failed = client.post("/api/article/create", json={"topic": "失败"}).json()["data"]
        blocked = client.post("/api/article/create", json={"topic": "评审阻断"}).json()["data"]
        malformed_review = client.post("/api/article/create", json={"topic": "坏评审"}).json()["data"]
        workflow = client.app.state.workflow
        workflow._update(running, status="RUNNING")  # noqa: SLF001 - persisted status fixture.
        workflow._update(  # noqa: SLF001 - persisted status fixture.
            completed,
            phase="COMPLETED",
            status="SUCCESS",
            review_report=json.dumps({"score": 91, "passed": True, "summary": "通过"}, ensure_ascii=False),
        )
        workflow._update(failed, status="FAILED")  # noqa: SLF001 - persisted status fixture.
        workflow._update(  # noqa: SLF001 - persisted status fixture.
            blocked,
            phase="COMPLETED",
            status="SUCCESS",
            review_report=json.dumps(
                {"score": 95, "passed": True, "summary": "有阻断", "blockingIssues": ["未闭合围栏"]},
                ensure_ascii=False,
            ),
        )
        workflow._update(malformed_review, phase="COMPLETED", status="SUCCESS", review_report="{bad json")  # noqa: SLF001
        page = client.get("/article/list")

    assert f'data-task-id="{running}" data-task-state="running"' in page.text
    assert f'data-task-id="{completed}" data-task-state="completed"' in page.text
    assert f'data-task-id="{waiting}" data-task-state="waiting"' in page.text
    assert f'data-task-id="{failed}" data-task-state="attention"' in page.text
    assert f'data-task-id="{blocked}" data-task-state="attention"' in page.text
    assert f'data-task-id="{malformed_review}" data-task-state="waiting"' in page.text
    assert "进行中" in page.text
    assert "已完成" in page.text
    assert "待继续" in page.text
    assert "需处理" in page.text


def test_task_list_keeps_delete_form_outside_detail_link_and_delete_script_is_async() -> None:
    template = (Path(__file__).parents[1] / "src/ai_article_studio/templates/list.html").read_text(encoding="utf-8")
    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")
    delete_handler = script[script.index('selectAll("form[data-delete-form]")'):]
    link_start = template.index('<a class="task-card-link"')
    detail_link = template[link_start:template.index("</a>", link_start)]

    assert '<form class="task-card-delete"' not in detail_link
    assert 'data-delete-form' in template
    assert 'data-delete-error' in template
    assert "window.confirm" in delete_handler
    assert 'headers: { Accept: "application/json" }' in delete_handler
    assert "window.location.reload()" in delete_handler
    assert "setDisabled(controls, false);" in delete_handler
    assert delete_handler.index("const formData = new FormData(form);") < delete_handler.index("setDisabled(controls, true);")
    assert "body: formData" in delete_handler


def test_title_choice_has_immediate_feedback_and_outline_save_becomes_dirty_only(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "交互状态"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titles["titleOptions"])[0]["title"]
        title_page = client.get(f"/article/detail/{task_id}")
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        outlined = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        outline_page = client.get(f"/article/detail/{task_id}")
        saved = client.post(
            f"/article/{task_id}/outline",
            data={"outline_json": outlined["outline"]},
            follow_redirects=False,
        )
        after_save = client.get(saved.headers["location"])

    assert 'data-title-confirm-form' in title_page.text
    assert 'aria-live="polite"' in title_page.text
    assert "文章标题" in title_page.text
    assert "选择后将进入大纲步骤" not in title_page.text
    assert "自动生成大纲" not in title_page.text
    assert 'data-title-error' in title_page.text
    assert 'data-outline-confirmation="required"' in outline_page.text
    assert saved.status_code == 303
    assert saved.headers["location"].endswith("?step=content#content-stage")
    assert 'data-step-card="content"' in after_save.text
    assert 'data-outline-editor' not in after_save.text


def test_async_forms_capture_values_before_disabling_controls() -> None:
    """Required form values must survive the busy UI state."""

    script = (Path(__file__).parents[1] / "src/ai_article_studio/static/stream.js").read_text(encoding="utf-8")
    job_handler = script[script.index("const startJob"): script.index('selectAll("form[data-pending-form]")')]
    title_handler = script[
        script.index('selectAll("form[data-title-confirm-form]")'): script.index('selectAll("form[data-job-form]")')
    ]
    create_handler = script[
        script.index('selectAll("form[data-create-form]")'): script.index('selectAll("form[data-delete-form]")')
    ]

    assert job_handler.index("const formData = new FormData(form);") < job_handler.index("setDisabled(jobControls(), true);")
    assert title_handler.index("const formData = new FormData(form);") < title_handler.index("setDisabled(controls, true);")
    assert create_handler.index("const formData = new FormData(form);") < create_handler.index("setDisabled(controls, true);")
    assert "catch (requestError)" in title_handler
    assert "setText(error, requestError.message" in title_handler
    assert "catch (requestError)" in create_handler
    assert "setText(error, requestError.message" in create_handler


def test_page_uses_precise_background_job_ctas_and_explicit_image_job(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "后台页面任务"}).json()["data"]
        initial = client.get(f"/article/detail/{task_id}")
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        title = json.loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": title})
        after_title = client.get(f"/article/detail/{task_id}")
        client.post(f"/api/article/continue-workflow/{task_id}")
        client.post(f"/api/article/continue-workflow/{task_id}")
        client.post(f"/api/article/continue-workflow/{task_id}")
        image_page = client.get(f"/article/detail/{task_id}")

    assert f'data-job-url="/article/{task_id}/jobs/next"' in initial.text
    assert "生成标题候选" in initial.text
    assert "生成大纲" in after_title.text
    assert "继续生成素材" not in after_title.text
    assert f'data-job-url="/article/{task_id}/jobs/images"' in image_page.text
    assert f'data-job-status-url="/article/{task_id}/jobs/status"' in image_page.text
    assert 'data-image-generation-form' in image_page.text


def test_one_click_page_creation_starts_a_server_side_background_job(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        created = client.post(
            "/article/create",
            data={"topic": "异步一键生成", "style": "TECH", "mode": "one_click"},
            follow_redirects=False,
        )
        task_url = created.headers["location"]
        page = client.get(task_url)
        task_id = task_url.rsplit("/", 1)[-1]
        task = client.get(f"/api/article/{task_id}").json()["data"]

    assert created.status_code == 303
    assert "?start=one-click" not in task_url
    assert 'data-auto-start-job="one-click"' not in page.text
    assert task["status"] in {"QUEUED", "RUNNING", "SUCCESS"}
    assert task["executionMode"] in {"ONE_CLICK", None}
    assert task["supervisorAction"]
    assert task["supervisorAction"] != "ONE_CLICK_GENERATE"


def test_one_click_page_creation_returns_a_small_json_handoff_for_the_browser(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/article/create",
            data={"topic": "快速进入一键工作台", "mode": "one_click"},
            headers={"accept": "application/json"},
        )

    assert response.status_code == 201
    assert response.json()["data"]["taskId"]
    assert response.json()["data"]["detailUrl"].startswith("/article/detail/")


def test_title_page_endpoint_returns_a_real_422_for_an_empty_title(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "标题校验"}).json()["data"]
        response = client.post(f"/article/{task_id}/title", data={"selected_title": ""}, follow_redirects=False)

    assert response.status_code == 422
    assert response.json()["message"] == "请求参数校验失败"


def test_completed_task_defaults_to_a_single_delivery_card(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "阶段折叠", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        page = client.get(f"/article/detail/{article['taskId']}")

    assert 'data-step-card="delivery"' in page.text
    assert 'data-step-card="title"' not in page.text
    assert 'data-step-card="outline"' not in page.text
    assert 'data-step-card="content"' not in page.text
    assert 'data-final-preview' in page.text
    assert "查看最终 Markdown 源码" not in page.text


def test_blocked_review_prioritizes_blockers_in_the_repair_card(tmp_path: Path) -> None:
    """A repair action owns the active card even if an older final draft exists."""

    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "阻断交付提示", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - representative persisted review state.
            article["taskId"],
            phase="ARTICLE_COMPOSITION",
            quality_score=94,
            review_report=json.dumps(
                {
                    "score": 94,
                    "passed": False,
                    "summary": "整体方向可用，但尚不能交付。",
                    "issues": ["一条很长的普通建议"],
                    "blockingIssues": ["代码围栏未闭合", "存在裸 HTML"],
                },
                ensure_ascii=False,
            ),
        )
        page = client.get(f"/article/detail/{article['taskId']}")

    assert 'data-review-blockers' in page.text
    assert "阻断交付问题" in page.text
    assert "代码围栏未闭合" in page.text
    assert 'data-step-card="content"' in page.text
    assert 'data-step-card="delivery"' not in page.text
    assert 'data-review-fix-action' in page.text
    assert "重新生成正文" not in page.text
    assert f'data-job-url="/article/{article["taskId"]}/jobs/next"' not in page.text
    assert "跳过确认，一键完成" not in page.text
    assert 'aria-label="文章导出"' not in page.text


def test_current_action_moves_into_the_bottom_step_navigation(tmp_path: Path) -> None:
    """A current-step CTA belongs beside the pager, not in a duplicate top card."""

    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "底部配图操作"}).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - establishes the pre-image-plan step.
            task_id,
            selected_title="底部配图操作",
            content="# 底部配图操作\n\n## 正文\n\n内容。",
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )
        page = client.get(f"/article/detail/{task_id}?step=images")

    markup = page.text
    assert '<p class="eyebrow">下一步操作</p>' not in markup
    assert 'class="wizard-next-action"' in markup
    assert markup.index('data-step-card="images"') < markup.index('id="workflow-panel"')
    navigation_start = markup.index('<nav class="wizard-card-navigation"')
    navigation_end = markup.index("</nav>", navigation_start)
    assert navigation_start < markup.index('id="workflow-panel"') < navigation_end
    assert "生成配图方案" in markup


def test_empty_step_hints_describe_the_available_action_without_layout_jargon(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "底部操作提示"}).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - establish the outline step without an outline artifact.
            task_id,
            selected_title="底部操作提示",
            phase="TITLE_SELECTION",
            status="SUCCESS",
        )
        page = client.get(f"/article/detail/{task_id}?step=outline")

    assert "点击“生成大纲”，确认后即可进入正文。" in page.text
    assert "操作区" not in page.text
    assert "使用上方“生成大纲”" not in page.text


def test_revision_limit_replaces_the_fix_button_with_a_manual_edit_hint(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "修订次数上限", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - representative persisted review state.
            article["taskId"],
            phase="ARTICLE_COMPOSITION",
            review_round=1,
            review_report=json.dumps(
                {
                    "score": 70,
                    "passed": False,
                    "summary": "仍需人工修改。",
                    "revisionAction": "minor",
                },
                ensure_ascii=False,
            ),
        )
        page = client.get(f"/article/detail/{article['taskId']}?step=delivery")

    assert 'data-review-fix-action' not in page.text
    assert 'data-review-manual-action' in page.text


def test_one_click_is_an_optional_utility_instead_of_a_competing_primary_cta(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "精简操作"}).json()["data"]
        client.post(f"/api/article/continue-workflow/{task_id}")
        title = json.loads(client.get(f"/api/article/{task_id}").json()["data"]["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": title})
        page = client.get(f"/article/detail/{task_id}")

    assert 'class="workflow-utilities"' in page.text
    assert "一键完成剩余步骤" in page.text
    assert '<details class="workflow-utilities"' in page.text


def test_image_card_shows_only_a_clickable_source_in_the_default_view(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "图片归因", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        first_image = json.loads(article["imageResults"])[0]
        first_image.update(
            {
                "source": "PEXELS",
                "sourceUrl": "https://www.pexels.com/photo/example/",
                "author": "Ada",
                "authorUrl": "https://www.pexels.com/@ada/",
            }
        )
        client.app.state.workflow._update(  # noqa: SLF001 - page rendering needs a representative persisted card.
            article["taskId"],
            image_results=json.dumps([first_image]),
        )
        page = client.get(f"/article/detail/{article['taskId']}?step=images")

    assert 'href="https://www.pexels.com/photo/example/"' in page.text
    assert "查看来源" in page.text
    assert "作者主页" not in page.text


def test_real_mode_image_choices_hide_provider_fallback_and_model_details() -> None:
    from ai_article_studio.main import _image_choices

    choices = _image_choices(Settings(app_mode=RuntimeMode.REAL))
    auto_choice = choices[0]

    assert auto_choice == {"value": "AUTO", "label": "自动选择"}
    assert all("Picsum" not in choice["label"] for choice in choices)
    assert all("Mock" not in choice["label"] for choice in choices)


def test_content_uses_safe_markdown_preview_and_edit_tabs(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "正文预览", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
    page = client.get(f"/article/detail/{article['taskId']}?step=content")

    assert 'data-content-preview' in page.text
    assert 'data-markdown-tabs="content"' in page.text
    assert "编辑正文" in page.text
    assert "文章预览" in page.text


def test_final_preview_uses_persisted_image_layout_metadata(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "最终预览版式", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        first_image = json.loads(article["imageResults"])[0]
        first_prompt = json.loads(article["imagePrompts"])[0]
        first_image.update({"visualRole": "DIAGRAM", "layout": "WIDE"})
        client.app.state.workflow._update(  # noqa: SLF001 - representative persisted image metadata.
            article["taskId"],
            image_prompts=json.dumps([first_prompt], ensure_ascii=False),
            image_results=json.dumps([first_image], ensure_ascii=False),
        )
        page = client.get(f"/article/detail/{article['taskId']}?step=delivery")

    assert 'class="article-figure layout-wide role-diagram"' in page.text


def test_complete_exports_only_render_in_the_passing_delivery_card(tmp_path: Path) -> None:
    """The final files must not be offered before a passing quality review."""

    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "交付导出位置", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        task_id = article["taskId"]
        workflow = client.app.state.workflow
        workflow._update(  # noqa: SLF001 - representative composed-but-unreviewed state.
            task_id,
            phase="ARTICLE_COMPOSITION",
            status="SUCCESS",
            review_report=None,
        )
        before_review = client.get(f"/article/detail/{task_id}?step=delivery")
        workflow._update(  # noqa: SLF001 - representative passing review state.
            task_id,
            phase="COMPLETED",
            status="SUCCESS",
            review_report=json.dumps(
                {"score": 91, "passed": True, "summary": "可以交付"}, ensure_ascii=False
            ),
        )
        after_review = client.get(f"/article/detail/{task_id}?step=delivery")

    assert 'data-final-export' not in before_review.text
    assert 'aria-label="文章导出"' not in before_review.text
    assert 'data-final-export' in after_review.text
    assert f'/api/article/export-markdown/{task_id}' in after_review.text
    assert f'/api/article/export-html/{task_id}' in after_review.text
    assert f'/api/article/export-zip/{task_id}' in after_review.text
    detail_head_start = after_review.text.index('<section class="detail-head">')
    detail_head = after_review.text[detail_head_start:after_review.text.index("</section>", detail_head_start)]
    assert 'data-final-export' not in detail_head


def test_outline_and_content_expose_accessible_preview_and_edit_tabs(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "分栏预览"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        outline = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]["outline"]
        outline_page = client.get(f"/article/detail/{task_id}?step=outline")
        client.post(f"/article/{task_id}/outline", data={"outline_json": outline})
        client.post(f"/api/article/continue-workflow/{task_id}")
        content_page = client.get(f"/article/detail/{task_id}?step=content")

    assert 'data-markdown-tabs="outline"' in outline_page.text
    assert 'aria-label="大纲查看方式"' in outline_page.text
    assert 'id="outline-preview-tab"' in outline_page.text
    assert 'id="outline-edit-tab"' in outline_page.text
    assert 'data-outline-preview' in outline_page.text
    assert 'data-markdown-tabs="content"' in content_page.text
    assert 'aria-label="正文查看方式"' in content_page.text
    assert 'id="content-preview-tab"' in content_page.text
    assert 'id="content-edit-tab"' in content_page.text
    assert 'data-content-preview' in content_page.text


def test_outline_preview_offers_direct_confirmation_and_content_uses_neutral_status(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "预览确认大纲"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        outline = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]["outline"]
        outline_page = client.get(f"/article/detail/{task_id}?step=outline")
        confirmed = client.post(
            f"/article/{task_id}/outline",
            data={"outline_json": outline},
            follow_redirects=False,
        )
        content_page = client.get(confirmed.headers["location"])

    assert 'data-outline-confirm-form' in outline_page.text
    assert '确认大纲并进入正文' in outline_page.text
    assert 'name="outline_json"' in outline_page.text
    assert confirmed.status_code == 303
    assert confirmed.headers["location"].endswith("?step=content#content-stage")
    assert '开始流式生成正文' not in content_page.text
    assert '默认实时生成' not in content_page.text


def test_streaming_preview_accepts_a_contract_checked_content_replacement() -> None:
    script = (Path(__file__).parents[1] / "src" / "ai_article_studio" / "static" / "stream.js").read_text(encoding="utf-8")

    assert 'addEventListener("content_replace"' in script
    assert "target.value = JSON.parse(event.data).content" in script


def test_image_stage_exposes_an_explicit_generation_choice_and_keeps_failures_in_page(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "图片选择"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        client.post(f"/api/article/continue-workflow/{task_id}")
        client.post(f"/api/article/continue-workflow/{task_id}")
        client.post(f"/api/article/continue-workflow/{task_id}")
        before_images = client.get(f"/article/detail/{task_id}")

        class FailingImageProvider:
            def generate(self, *_args):
                request = httpx.Request("POST", "https://images.example.test/generate")
                response = httpx.Response(429, request=request)
                raise httpx.HTTPStatusError("test image provider failure", request=request, response=response)

        client.app.state.workflow.images.providers[ImageProvider.SILICONFLOW] = FailingImageProvider()
        failed = client.post(
            f"/article/{task_id}/images",
            data={"provider": "SILICONFLOW"},
            follow_redirects=False,
        )
        for _ in range(50):
            if client.get(f"/article/{task_id}/jobs/status").json()["data"]["status"] == "FAILED":
                break
            time.sleep(0.01)
        after_failure = client.get(f"/article/detail/{task_id}")

    assert 'action="/article/' in before_images.text
    assert "图片来源" in before_images.text
    assert "图片来源与模型" not in before_images.text
    assert 'data-image-generation-form' in before_images.text
    assert failed.status_code == 303
    assert failed.headers["location"].endswith("?step=images#images-stage")
    assert "test image provider failure" in after_failure.text

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.llm import MockArticleAgent
from ai_article_studio.schemas import ImagePlacement, ImagePromptOption, ImageProvider, ImageVisualRole, QualityReviewResult
from ai_article_studio.supervisor import SupervisorAction
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


def test_create_then_continue_generates_title_candidates(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        created = client.post("/api/article/create", json={"topic": "AI 写作如何保持可控"})
        task_id = created.json()["data"]

        advanced = client.post(f"/api/article/continue-workflow/{task_id}")

    article = advanced.json()["data"]
    assert created.json()["code"] == 0
    assert article["phase"] == "TITLE_SELECTION"
    assert len(__import__("json").loads(article["titleOptions"])) == 3


def test_step_create_api_accepts_an_optional_per_task_config(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        created = client.post(
            "/api/article/create",
            json={
                "topic": "接口配置创建",
                "style": "BUSINESS",
                "config": {
                    "textModel": "qwen-plus",
                    "contentLengthLevel": "SHORT",
                    "imageProvider": "MOCK",
                    "imageCountMode": "CUSTOM",
                    "imageCount": 3,
                },
            },
        )
        article = client.get(f"/api/article/{created.json()['data']}").json()["data"]

    config = json.loads(article["modelConfig"])
    assert article["style"] == "BUSINESS"
    assert config["textModel"] == "qwen-plus"
    assert config["contentLengthLevel"] == "SHORT"
    assert config["imageCount"] == 3


def test_mock_one_click_workflow_composes_images_and_exports_before_review(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/article/one-click-generate",
            json={
                "topic": "用确定性状态机控制 AI 内容工作流",
                "style": "TECH",
                "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3},
            },
        )
        article = response.json()["data"]
        markdown = client.get(f"/api/article/export-markdown/{article['taskId']}")
        html = client.get(f"/api/article/export-html/{article['taskId']}")
        archive = client.get(f"/api/article/export-zip/{article['taskId']}")
        logs = client.get(f"/api/article/agent-logs/{article['taskId']}")

    assert response.json()["code"] == 0
    assert article["phase"] == "ARTICLE_COMPOSITION"
    assert article["status"] == "SUCCESS"
    assert article["finalMarkdown"]
    assert article["qualityScore"] is None
    assert article["reviewReport"] is None
    assert len(__import__("json").loads(article["imageResults"])) == 3
    assert markdown.status_code == 200 and markdown.headers["content-type"].startswith("text/markdown")
    assert markdown.content.startswith(b"#")
    assert html.status_code == 200 and b"data:image/svg+xml;base64" in html.content
    assert archive.status_code == 200 and archive.headers["content-type"].startswith("application/zip")
    assert any(log["agentName"] == "ImageResultAgent" for log in logs.json()["data"])


def test_one_click_api_without_config_inherits_the_configured_default_text_model(tmp_path: Path) -> None:
    from ai_article_studio.main import create_app

    settings = Settings(app_mode=RuntimeMode.MOCK, openai_model="qwen-plus")
    with TestClient(
        create_app(
            database_url=f"sqlite:///{tmp_path / 'articles.db'}",
            upload_dir=tmp_path / "images",
            settings=settings,
        )
    ) as client:
        response = client.post("/api/article/one-click-generate", json={"topic": "继承默认模型"})

    assert response.status_code == 200
    assert json.loads(response.json()["data"]["modelConfig"])["textModel"] == "qwen-plus"


def test_one_click_api_failure_returns_recoverable_task_payload(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow

        def fail_content(*_args: object) -> str:
            raise RuntimeError("content provider temporarily unavailable")

        workflow.agent.content = fail_content  # type: ignore[method-assign]
        response = client.post(
            "/api/article/one-click-generate",
            json={"topic": "一键 API 失败可恢复", "config": {"imageProvider": "MOCK"}},
        )

    payload = response.json()
    assert response.status_code >= 400
    assert payload["code"] != 0
    assert payload["data"]["taskId"]
    assert payload["data"]["supervisorAction"] == "GENERATE_CONTENT"
    assert payload["data"]["errorMessage"]


def test_existing_one_click_api_resumes_the_persisted_failed_substep(tmp_path: Path) -> None:
    """The public existing-task route must continue orchestration from its checkpoint."""

    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        original_content = workflow.agent.content

        def fail_content(*_args: object) -> str:
            raise RuntimeError("content provider temporarily unavailable")

        workflow.agent.content = fail_content  # type: ignore[method-assign]
        failed = client.post(
            "/api/article/one-click-generate",
            json={"topic": "已有任务一键续跑", "config": {"imageProvider": "MOCK"}},
        )
        task_id = failed.json()["data"]["taskId"]
        workflow.agent.content = original_content  # type: ignore[method-assign]
        resumed = client.post(f"/api/article/one-click-generate/{task_id}")

    assert failed.status_code == 500
    assert failed.json()["data"]["supervisorAction"] == "GENERATE_CONTENT"
    assert resumed.status_code == 200
    assert resumed.json()["data"]["phase"] == "ARTICLE_COMPOSITION"
    assert resumed.json()["data"]["status"] == "SUCCESS"
    assert resumed.json()["data"]["reviewReport"] is None


def test_image_generation_persists_visual_review_and_logs_it_separately(tmp_path: Path) -> None:
    from ai_article_studio.schemas import ImageVisualReview, ImageVisualReviewStatus

    class PassingVisionReviewer:
        def review(self, _image, _prompt):  # type: ignore[no-untyped-def]
            return ImageVisualReview(status=ImageVisualReviewStatus.PASSED, score=88, issues=[])

    with build_client(tmp_path) as client:
        client.app.state.workflow.vision_reviewer = PassingVisionReviewer()
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "视觉质检记录", "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3}},
        ).json()["data"]
        logs = client.get(f"/api/article/agent-logs/{article['taskId']}").json()["data"]

    results = json.loads(article["imageResults"])
    assert [item["visualReview"]["status"] for item in results] == ["PASSED", "PASSED", "PASSED"]
    assert all(item["visualReview"]["score"] == 88 for item in results)
    assert any(log["agentName"] == "ImageVisualReviewAgent" and log["status"] == "SUCCESS" for log in logs)


def test_image_plan_layout_edit_keeps_generated_asset_and_returns_json(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "配图编排元数据", "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3}},
        ).json()["data"]
        task_id = article["taskId"]
        prompts = json.loads(article["imagePrompts"])
        results = json.loads(article["imageResults"])
        edited = {**prompts[1], "layout": "COMPACT"}

        response = client.post(
            f"/article/{task_id}/image-plan/1",
            json=edited,
            headers={"accept": "application/json"},
        )
        logs = client.get(f"/api/article/agent-logs/{task_id}").json()["data"]

    assert response.status_code == 200
    saved = response.json()["data"]
    saved_results = json.loads(saved["imageResults"])
    assert saved_results[1]["imageUrl"] == results[1]["imageUrl"]
    assert saved_results[1]["layout"] == "COMPACT"
    assert saved["finalMarkdown"] is None
    assert saved["reviewReport"] is None
    assert any(log["agentName"] == "ImagePlanEditorAgent" and log["status"] == "SUCCESS" for log in logs)


def test_image_plan_edit_rejects_invalid_heading_and_paragraph_without_failing_task(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "配图锚点校验", "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3}},
        ).json()["data"]
        task_id = article["taskId"]
        prompt = json.loads(article["imagePrompts"])[1]
        bad_heading = client.post(
            f"/article/{task_id}/image-plan/1",
            json={**prompt, "anchorHeading": "不存在的章节"},
            headers={"accept": "application/json"},
        )
        bad_paragraph = client.post(
            f"/article/{task_id}/image-plan/1",
            json={**prompt, "anchorParagraph": 99},
            headers={"accept": "application/json"},
        )
        refreshed = client.get(f"/api/article/{task_id}").json()["data"]

    assert bad_heading.status_code == 400
    assert "未找到锚点标题" in bad_heading.json()["message"]
    assert bad_paragraph.status_code == 400
    assert "锚点段落无效" in bad_paragraph.json()["message"]
    assert refreshed["status"] == "SUCCESS"
    assert refreshed["finalMarkdown"] is not None


def test_image_plan_layout_edit_before_generation_keeps_image_generation_as_next_action(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "生成前调整配图"}).json()["data"]
        titled = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = json.loads(titled["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        outlined = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        client.post(
            f"/article/{task_id}/outline",
            data={"outline_markdown": outline_preview_markdown(json.loads(outlined["outline"]))},
        )
        client.post(f"/api/article/continue-workflow/{task_id}")
        planned = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        prompt = json.loads(planned["imagePrompts"])[1]
        response = client.post(
            f"/article/{task_id}/image-plan/1",
            json={**prompt, "layout": "COMPACT"},
            headers={"accept": "application/json"},
        )

    assert response.status_code == 200
    assert response.json()["data"]["imageResults"] is None
    assert response.json()["data"]["supervisorAction"] == "GENERATE_IMAGE_RESULTS"


def test_image_plan_visual_edit_invalidates_results_and_reorder_preserves_pairs(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "配图重排", "config": {"imageProvider": "MOCK", "imageCountMode": "CUSTOM", "imageCount": 3}},
        ).json()["data"]
        task_id = article["taskId"]
        prompts = json.loads(article["imagePrompts"])
        results = json.loads(article["imageResults"])
        reordered = client.post(
            f"/article/{task_id}/image-plan/reorder",
            json={"indices": [0, 2, 1]},
            headers={"accept": "application/json"},
        )
        assert reordered.status_code == 200
        reordered_data = reordered.json()["data"]
        reordered_prompts = json.loads(reordered_data["imagePrompts"])
        reordered_results = json.loads(reordered_data["imageResults"])
        visual_edit = client.post(
            f"/article/{task_id}/image-plan/1",
            json={**reordered_prompts[1], "visualRole": "SCENE"},
            headers={"accept": "application/json"},
        )

    assert [item["imageTitle"] for item in reordered_prompts] == [prompts[0]["imageTitle"], prompts[2]["imageTitle"], prompts[1]["imageTitle"]]
    assert [item["imageUrl"] for item in reordered_results] == [results[0]["imageUrl"], results[2]["imageUrl"], results[1]["imageUrl"]]
    assert visual_edit.status_code == 200
    invalidated = visual_edit.json()["data"]
    assert json.loads(invalidated["imagePrompts"])[1]["visualRole"] == "SCENE"
    assert invalidated["imageResults"] is None
    assert invalidated["finalMarkdown"] is None
    assert invalidated["reviewReport"] is None


def test_agent_logs_redact_auth_and_structured_secret_values_before_persistence(tmp_path: Path) -> None:
    leaked_values = ("bearer-secret", "basic-secret", "json-api-secret", "json-token-secret", "kv-access-secret")
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "日志脱敏"}).json()["data"]

        def fail_titles(*_args: object):  # type: ignore[no-untyped-def]
            raise RuntimeError(
                "Authorization: Bearer bearer-secret; Proxy-Authorization: Basic basic-secret; "
                '{"api_key":"json-api-secret","token":"json-token-secret"}; access_token=kv-access-secret'
            )

        client.app.state.workflow.agent.titles = fail_titles  # type: ignore[method-assign]
        response = client.post(f"/api/article/generate-titles/{task_id}")
        logs = client.get(f"/api/article/agent-logs/{task_id}").json()["data"]

    assert response.status_code == 500
    assert len(logs) == 1
    persisted = " ".join(str(log[key] or "") for log in logs for key in ("inputText", "outputText", "errorMessage"))
    assert "[REDACTED]" in persisted
    assert not any(secret in persisted for secret in leaked_values)


def test_generated_image_plan_repairs_a_stale_model_anchor_before_persistence(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "生成方案锚点"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish the content that generated anchors must match.
            task_id,
            selected_title="生成方案锚点",
            content="# 生成方案锚点\n\n## 已存在章节\n\n唯一段落。",
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )
        workflow.agent.image_prompts = lambda *_args: [  # type: ignore[method-assign]
            ImagePromptOption(
                image_title="封面",
                usage_scene="文章封面",
                prompt_zh="封面",
                prompt_en="cover",
                visual_role=ImageVisualRole.COVER,
                placement=ImagePlacement.COVER,
            ),
            ImagePromptOption(
                image_title="错误锚点",
                usage_scene="正文图",
                prompt_zh="图解",
                prompt_en="diagram",
                visual_role=ImageVisualRole.DIAGRAM,
                placement=ImagePlacement.AFTER_PARAGRAPH,
                anchor_heading="不存在章节",
                anchor_paragraph=1,
            ),
        ]
        response = client.post(f"/api/article/generate-image-prompts/{task_id}")
        saved = client.get(f"/api/article/{task_id}").json()["data"]

    assert response.status_code == 200
    repaired = json.loads(saved["imagePrompts"])[1]
    assert repaired["anchorHeading"] == "已存在章节"
    assert repaired["placement"] == "SECTION_END"
    assert repaired["anchorParagraph"] == 0
    assert repaired["anchorExcerpt"] == "唯一段落。"
    assert saved["status"] == "SUCCESS"


def test_generated_image_plan_clamps_a_model_anchor_to_the_last_real_paragraph(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "段落锚点修复"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish one real paragraph.
            task_id,
            selected_title="段落锚点修复",
            content="# 段落锚点修复\n\n## 行为锚点\n\n只有一段正文。",
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )
        workflow.agent.image_prompts = lambda *_args: [  # type: ignore[method-assign]
            ImagePromptOption(
                image_title="封面",
                usage_scene="文章封面",
                prompt_zh="封面",
                prompt_en="cover",
                visual_role=ImageVisualRole.COVER,
                placement=ImagePlacement.COVER,
            ),
            ImagePromptOption(
                image_title="正文图",
                usage_scene="正文图",
                prompt_zh="图解",
                prompt_en="diagram",
                visual_role=ImageVisualRole.DIAGRAM,
                placement=ImagePlacement.AFTER_PARAGRAPH,
                anchor_heading="行为锚点",
                anchor_paragraph=2,
            ),
        ]
        response = client.post(f"/api/article/generate-image-prompts/{task_id}")
        saved = client.get(f"/api/article/{task_id}").json()["data"]

    assert response.status_code == 200
    repaired = json.loads(saved["imagePrompts"])[1]
    assert repaired["anchorHeading"] == "行为锚点"
    assert repaired["placement"] == "AFTER_PARAGRAPH"
    assert repaired["anchorParagraph"] == 1
    assert repaired["anchorExcerpt"] == "只有一段正文。"


def test_existing_image_plan_is_lazily_bound_to_its_article_excerpt_before_generation(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "旧方案补全证据"}).json()["data"]
        legacy_prompt = ImagePromptOption(
            image_title="流程偏离对照",
            usage_scene="解释重复流程为何会分叉",
            prompt_zh="重复流程的两条路径对照图",
            prompt_en="two repeated workflow paths",
            visual_role=ImageVisualRole.CONCEPT,
            placement=ImagePlacement.AFTER_PARAGRAPH,
            anchor_heading="重复执行",
            anchor_paragraph=1,
        )
        workflow._update(  # noqa: SLF001 - simulate a record created before anchor excerpts existed.
            task_id,
            content="# 旧方案补全证据\n\n## 重复执行\n\n同一流程多次运行后，第二次开始出现偏离。",
            image_prompts=json.dumps([legacy_prompt.model_dump(by_alias=True)], ensure_ascii=False),
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )

        workflow.generate_image_results(task_id, ImageProvider.MOCK)
        saved = client.get(f"/api/article/{task_id}").json()["data"]

    prompt = json.loads(saved["imagePrompts"])[0]
    result = json.loads(saved["imageResults"])[0]
    assert prompt["anchorExcerpt"] == "同一流程多次运行后，第二次开始出现偏离。"
    assert result["anchorExcerpt"] == prompt["anchorExcerpt"]


def test_generated_image_plan_enforces_cover_role_invariants(tmp_path: Path) -> None:
    def cover() -> ImagePromptOption:
        return ImagePromptOption(
            image_title="封面",
            usage_scene="文章封面",
            prompt_zh="封面",
            prompt_en="cover",
            visual_role=ImageVisualRole.COVER,
            placement=ImagePlacement.COVER,
        )

    def section(role: ImageVisualRole, heading: str) -> ImagePromptOption:
        return ImagePromptOption(
            image_title=f"{heading}配图",
            usage_scene="正文图",
            prompt_zh="图解",
            prompt_en="diagram",
            visual_role=role,
            placement=ImagePlacement.AFTER_PARAGRAPH,
            anchor_heading=heading,
            anchor_paragraph=1,
        )

    invalid_plans = [([section(ImageVisualRole.CONCEPT, "第一节"), cover()], "封面图必须位于配图方案第 1 项")]
    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        for index, (plans, expected_error) in enumerate(invalid_plans):
            task_id = client.post("/api/article/create", json={"topic": f"方案不变量{index}"}).json()["data"]
            workflow._update(  # noqa: SLF001 - establish valid Markdown anchors for each invalid storyboard.
                task_id,
                selected_title=f"方案不变量{index}",
                content="# 方案不变量\n\n## 第一节\n\n第一段。\n\n## 第二节\n\n第二段。",
                phase="CONTENT_GENERATION",
                status="SUCCESS",
            )
            workflow.agent.image_prompts = lambda *_args, plans=plans: plans  # type: ignore[method-assign]
            response = client.post(f"/api/article/generate-image-prompts/{task_id}")
            saved = client.get(f"/api/article/{task_id}").json()["data"]

            assert response.status_code == 400
            assert expected_error in response.json()["message"]
            assert saved["status"] == "FAILED"
            assert saved["imagePrompts"] is None


def test_generated_image_plan_deduplicates_colliding_anchor_before_persistence(tmp_path: Path) -> None:
    content = "# 标题\n\n## 第一节\n\n唯一段落。\n\n## 第二节\n\n另一段落。"
    plans = [
        ImagePromptOption(
            image_title="封面", usage_scene="封面", prompt_zh="封面", prompt_en="cover", visual_role=ImageVisualRole.COVER, placement=ImagePlacement.COVER
        ),
        ImagePromptOption(
            image_title="保留图",
            usage_scene="第一节",
            prompt_zh="图解",
            prompt_en="diagram",
            visual_role=ImageVisualRole.DIAGRAM,
            placement=ImagePlacement.AFTER_PARAGRAPH,
            anchor_heading="第一节",
            anchor_paragraph=1,
        ),
        ImagePromptOption(
            image_title="冲突图",
            usage_scene="第一节",
            prompt_zh="图解",
            prompt_en="diagram",
            visual_role=ImageVisualRole.DIAGRAM,
            placement=ImagePlacement.SECTION_END,
            anchor_heading="第一节",
        ),
    ]

    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "锚点冲突"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish the generated plan's real anchor catalogue.
            task_id,
            selected_title="锚点冲突",
            content=content,
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )
        workflow.agent.image_prompts = lambda *_args: plans  # type: ignore[method-assign]
        generated = client.post(f"/api/article/generate-image-prompts/{task_id}")
        saved = client.get(f"/api/article/{task_id}").json()["data"]

    assert generated.status_code == 200
    assert saved["status"] == "SUCCESS"
    assert [prompt["imageTitle"] for prompt in json.loads(saved["imagePrompts"])] == ["封面", "保留图"]


def test_generated_image_plan_preserves_duplicate_non_cover_roles_and_manual_position_edit_remains_strict(tmp_path: Path) -> None:
    def cover() -> ImagePromptOption:
        return ImagePromptOption(
            image_title="封面",
            usage_scene="文章封面",
            prompt_zh="封面",
            prompt_en="cover",
            visual_role=ImageVisualRole.COVER,
            placement=ImagePlacement.COVER,
        )

    def section(title: str, heading: str) -> ImagePromptOption:
        return ImagePromptOption(
            image_title=title,
            usage_scene="正文图",
            prompt_zh="流程图解",
            prompt_en="workflow diagram",
            visual_role=ImageVisualRole.CONCEPT,
            placement=ImagePlacement.AFTER_PARAGRAPH,
            anchor_heading=heading,
            anchor_paragraph=1,
        )

    with build_client(tmp_path) as client:
        workflow = client.app.state.workflow
        task_id = client.post("/api/article/create", json={"topic": "重复视觉角色"}).json()["data"]
        workflow._update(  # noqa: SLF001 - establish real anchors for model-created storyboard cards.
            task_id,
            selected_title="重复视觉角色",
            content="# 重复视觉角色\n\n## 第一节\n\n第一段。\n\n## 第二节\n\n第二段。",
            phase="CONTENT_GENERATION",
            status="SUCCESS",
        )
        workflow.agent.image_prompts = lambda *_args: [  # type: ignore[method-assign]
            cover(),
            section("第一节流程", "第一节"),
            section("第二节流程", "第二节"),
        ]
        generated = client.post(f"/api/article/generate-image-prompts/{task_id}")
        persisted = generated.json()["data"]
        prompts = json.loads(persisted["imagePrompts"])
        manual_duplicate = client.post(
            f"/article/{task_id}/image-plan/2",
            json={**prompts[2], "visualRole": prompts[1]["visualRole"]},
            headers={"accept": "application/json"},
        )
        manual_collision = client.post(
            f"/article/{task_id}/image-plan/2",
            json={**prompts[2], "placement": "SECTION_END", "anchorHeading": "第一节", "anchorParagraph": 0},
            headers={"accept": "application/json"},
        )

    assert generated.status_code == 200
    assert [prompt["visualRole"] for prompt in prompts] == ["COVER", "CONCEPT", "CONCEPT"]
    assert manual_duplicate.status_code == 200
    assert manual_collision.status_code == 400
    assert "相邻配图不能使用同一锚点" in manual_collision.json()["message"]


def test_streaming_content_saves_article_and_emits_content_then_done(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "流式生成"}).json()["data"]
        article = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = __import__("json").loads(article["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        client.post(f"/api/article/continue-workflow/{task_id}")
        stream = client.get(f"/api/article/real-stream-generate-content/{task_id}")
        saved = client.get(f"/api/article/{task_id}")

    assert "event: content" in stream.text
    assert "event: done" in stream.text
    assert "开始流式生成正文" not in stream.text
    assert "正在生成正文" in stream.text
    assert saved.json()["data"]["content"].startswith("#")


def test_streaming_content_clears_previous_image_and_review_artifacts(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "流式覆盖旧产物", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        stream = client.get(f"/api/article/real-stream-generate-content/{article['taskId']}")
        refreshed = client.get(f"/api/article/{article['taskId']}").json()["data"]

    assert "event: done" in stream.text
    assert refreshed["imagePrompts"] is None
    assert refreshed["imageResults"] is None
    assert refreshed["finalMarkdown"] is None
    assert refreshed["reviewReport"] is None


def test_generated_content_preserves_model_text_for_later_explicit_review(tmp_path: Path) -> None:
    class ContractRepairingAgent(MockArticleAgent):
        repair_calls = 0

        def content(self, topic: str, title: str, outline, style: str, length: str) -> str:  # type: ignore[no-untyped-def]
            return "# 标题\n\n## 第一步\n\n扫码领取动作图解库，372 位体验者都说亲测有效。"

        def repair_content(self, *_args: object) -> str:
            self.repair_calls += 1
            return "# 标题\n\n## 第一步\n\n从低强度动作开始，并根据自身感受逐步调整。"

    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "如何从零开始健身", "style": "POPULAR"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = __import__("json").loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        client.post(f"/api/article/continue-workflow/{task_id}")
        agent = ContractRepairingAgent()
        client.app.state.workflow.agent = agent
        generated = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]

    assert agent.repair_calls == 0
    assert "扫码领取动作图解库" in generated["content"]
    assert "372 位体验者" in generated["content"]


def test_streaming_content_saves_model_text_without_contract_repair(tmp_path: Path) -> None:
    class ContractRepairingStreamAgent(MockArticleAgent):
        repair_calls = 0

        def stream_content(self, *_args: object):  # type: ignore[no-untyped-def]
            yield '# 标题\n\n## 第一步\n\n```json\n{"resource": "扫码领取"}\n```'

        def repair_content(self, *_args: object) -> str:
            self.repair_calls += 1
            return "# 标题\n\n## 第一步\n\n把训练安排写成可自行执行的清单。"

    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "如何从零开始健身", "style": "POPULAR"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        selected_title = __import__("json").loads(titles["titleOptions"])[0]["title"]
        client.post("/api/article/confirm-title", json={"taskId": task_id, "selectedTitle": selected_title})
        client.post(f"/api/article/continue-workflow/{task_id}")
        agent = ContractRepairingStreamAgent()
        client.app.state.workflow.agent = agent
        stream = client.get(f"/api/article/real-stream-generate-content/{task_id}")
        saved = client.get(f"/api/article/{task_id}").json()["data"]

    assert agent.repair_calls == 0
    assert "event: content_replace" not in stream.text
    assert saved["content"] == '# 标题\n\n## 第一步\n\n```json\n{"resource": "扫码领取"}\n```'


def test_saving_model_config_resets_downstream_artifacts(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "重置下游结果", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        response = client.post(
            "/api/article/save-model-config",
            json={
                "taskId": article["taskId"],
                "style": "BUSINESS",
                "config": {"imageProvider": "MOCK", "contentLengthLevel": "SHORT"},
            },
        )

    reset = response.json()["data"]
    assert reset["phase"] == "CREATED"
    assert reset["content"] is None
    assert reset["imageResults"] is None
    assert reset["finalMarkdown"] is None


def test_saving_an_identical_model_config_does_not_reset_composed_work(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "幂等配置保存", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        saved = client.post(
            "/api/article/save-model-config",
            json={"taskId": article["taskId"], "style": article["style"], "config": json.loads(article["modelConfig"])},
        ).json()["data"]

    assert saved["phase"] == "ARTICLE_COMPOSITION"
    assert saved["content"] == article["content"]
    assert saved["finalMarkdown"] == article["finalMarkdown"]


def test_failed_explicit_review_requires_an_explicit_revision_and_fresh_review(tmp_path: Path) -> None:
    class AlwaysFailReviewAgent(MockArticleAgent):
        def review(self, article: str) -> QualityReviewResult:
            return QualityReviewResult(
                score=68,
                passed=False,
                summary="需要重新生成图片",
                revision_action=SupervisorAction.REGENERATE_IMAGE_RESULTS,
                revision_reason="配图与文章不匹配",
            )

    with build_client(tmp_path) as client:
        client.app.state.workflow.agent = AlwaysFailReviewAgent()
        response = client.post(
            "/api/article/one-click-generate",
            json={"topic": "自动评审修订", "config": {"imageProvider": "MOCK"}},
        )
        composed = response.json()["data"]
        reviewed = client.post(f"/api/article/continue-workflow/{composed['taskId']}").json()["data"]
        revised = client.post(f"/api/article/apply-review-revision/{composed['taskId']}").json()["data"]
        repeated = client.post(f"/api/article/apply-review-revision/{composed['taskId']}").json()["data"]

    assert composed["reviewRound"] == 0
    assert composed["reviewReport"] is None
    assert reviewed["reviewRound"] == 0
    assert reviewed["reviewReport"] is not None
    assert reviewed["phase"] == "ARTICLE_COMPOSITION"
    assert revised["reviewRound"] == 1
    assert revised["phase"] == "ARTICLE_COMPOSITION"
    assert revised["reviewReport"] is None
    assert revised["finalMarkdown"]
    assert repeated["reviewRound"] == 1
    assert repeated["reviewReport"] is None


def test_invalid_generated_markdown_is_preserved_until_explicit_review_blocks_it(tmp_path: Path) -> None:
    class InvalidMarkdownAgent(MockArticleAgent):
        def content(self, topic: str, title: str, outline, style: str, length: str) -> str:  # type: ignore[no-untyped-def]
            return "这段正文故意没有二级标题。"

        def review(self, article: str) -> QualityReviewResult:
            return QualityReviewResult(score=94, passed=True, summary="模型误判为可交付")

    with build_client(tmp_path) as client:
        client.app.state.workflow.agent = InvalidMarkdownAgent()
        task_id = client.post("/api/article/create", json={"topic": "高分阻断问题"}).json()["data"]
        titles = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        client.post(
            "/api/article/confirm-title",
            json={"taskId": task_id, "selectedTitle": json.loads(titles["titleOptions"])[0]["title"]},
        )
        client.post(f"/api/article/continue-workflow/{task_id}")
        generated = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]
        client.app.state.workflow._update(  # noqa: SLF001 - model-only markdown is a deliberate review fixture.
            task_id,
            final_markdown=generated["content"],
            phase="ARTICLE_COMPOSITION",
            status="SUCCESS",
        )
        reviewed = client.post(f"/api/article/continue-workflow/{task_id}").json()["data"]

    review = json.loads(reviewed["reviewReport"])
    assert generated["content"] == "这段正文故意没有二级标题。"
    assert generated["reviewReport"] is None
    assert reviewed["qualityScore"] == 94
    assert reviewed["phase"] == "ARTICLE_COMPOSITION"
    assert review["passed"] is False
    assert review["blockingIssues"]


def test_review_cannot_complete_when_a_high_score_ignores_an_unsourced_percentage(tmp_path: Path) -> None:
    class OverlyPermissiveReviewAgent(MockArticleAgent):
        def review(self, article: str) -> QualityReviewResult:
            return QualityReviewResult(score=96, passed=True, summary="模型误判为可交付")

    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "量化事实校验", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        workflow = client.app.state.workflow
        workflow.agent = OverlyPermissiveReviewAgent()
        workflow._update(  # noqa: SLF001 - create a persisted final-delivery review fixture.
            article["taskId"],
            phase="ARTICLE_COMPOSITION",
            status="SUCCESS",
            final_markdown="# 量化事实校验\n\n## 结论\n\n90% 的团队都会遇到这个问题。",
            review_report=None,
            quality_score=None,
            review_round=0,
        )
        reviewed = client.post(f"/api/article/continue-workflow/{article['taskId']}").json()["data"]

    review = __import__("json").loads(reviewed["reviewReport"])
    assert reviewed["phase"] == "ARTICLE_COMPOSITION"
    assert review["passed"] is False
    assert any("无来源的比例或量化结论" in issue for issue in review["blockingIssues"])


def test_validation_errors_use_the_base_response_shape(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/article/one-click-generate",
            json={
                "topic": "非法图片数量",
                "config": {"imageCountMode": "CUSTOM", "imageCount": 2},
            },
        )

    assert response.status_code == 422
    assert response.json()["code"] == 422
    assert response.json()["message"] == "请求参数校验失败"
    assert isinstance(response.json()["data"], list)


def test_export_is_rejected_until_a_final_article_exists(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        task_id = client.post("/api/article/create", json={"topic": "未完成导出"}).json()["data"]
        response = client.get(f"/api/article/export-markdown/{task_id}")

    assert response.status_code == 400
    assert response.json()["code"] == 400
    assert "完成图文稿" in response.json()["message"]


def test_retrying_one_image_returns_article_to_composition_flow(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "单图重试", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        retried = client.post(f"/api/article/regenerate-image-result/{article['taskId']}?index=0&provider=MOCK").json()["data"]
        composed = client.post(f"/api/article/continue-workflow/{article['taskId']}").json()["data"]
        completed = client.post(f"/api/article/continue-workflow/{article['taskId']}").json()["data"]

    assert retried["phase"] == "CONTENT_GENERATION"
    assert retried["finalMarkdown"] is None
    assert composed["phase"] == "ARTICLE_COMPOSITION"
    assert completed["phase"] == "COMPLETED"


def test_saving_an_edited_outline_resets_downstream_content_and_enters_content_stage(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        article = client.post(
            "/api/article/one-click-generate",
            json={"topic": "大纲变更", "config": {"imageProvider": "MOCK"}},
        ).json()["data"]
        saved = client.post(
            "/api/article/save-outline",
            json={
                "taskId": article["taskId"],
                "outline": [{"heading": "重新设计", "description": "大纲已调整"}],
            },
        ).json()["data"]

    assert saved["phase"] == "CONTENT_GENERATION"
    assert saved["content"] is None
    assert saved["imagePrompts"] is None
    assert saved["imageResults"] is None
    assert saved["finalMarkdown"] is None
    assert saved["reviewReport"] is None

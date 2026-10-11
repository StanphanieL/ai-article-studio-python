"""Image planning keeps valid work and delegates layout to deterministic code."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from ai_article_studio.config import Settings
from ai_article_studio.domain import AdapterResponseValidationError
from ai_article_studio.llm import OpenAICompatibleArticleAgent
from ai_article_studio.schemas import ArticleModelConfig, ImageProvider

BODY = "# 知识库\n\n整理知识并明确检索需求。\n\n## 收集\n\n将笔记按主题放入文件夹。\n\n## 检索\n\n输入关键词，找到相关笔记。"


def item(slot_id: str, title: str, role: str = "DIAGRAM") -> dict[str, str]:
    return {
        "slotId": slot_id,
        "imageTitle": title,
        "usageScene": "说明笔记与检索的关系",
        "promptZh": "笔记卡片沿箭头汇入文件夹，简洁图解，无文字无水印",
        "promptEn": "Note cards flow along arrows into folders, clean diagram, no text or watermark",
        "visualRole": role,
    }


@contextmanager
def planner(replies: list[object]) -> Iterator[tuple[OpenAICompatibleArticleAgent, list[dict]]]:
    requests: list[dict] = []
    remaining = iter(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        reply = next(remaining)
        if isinstance(reply, httpx.HTTPError):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 180},
            },
            request=request,
        )

    agent = OpenAICompatibleArticleAgent(
        Settings(_env_file=None, openai_api_key="diagnostic-secret"),
        ArticleModelConfig(text_model="qwen-plus", image_provider=ImageProvider.SILICONFLOW),
    )
    agent.client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        agent.client = client
        yield agent, requests


def test_compact_plan_assigns_server_anchors_and_preserves_repeated_roles() -> None:
    with planner([[item("cover", "封面", "COVER"), item("s1p1", "收集笔记"), item("s2p1", "检索笔记")]]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert [(p.visual_role.value, p.anchor_heading, p.anchor_paragraph) for p in plans] == [
        ("COVER", "", 0),
        ("DIAGRAM", "收集", 1),
        ("DIAGRAM", "检索", 1),
    ]
    assert plans[0].placement == "COVER" and plans[0].layout == "HERO"
    assert len(calls) == 1
    assert calls[0]["model"] == "qwen-plus"
    assert 1024 <= calls[0]["max_completion_tokens"] < 4096
    assert agent.model_config.max_completion_tokens == 8192


def test_repair_only_requests_bad_slots_and_keeps_successful_descriptions() -> None:
    good_cover = item("cover", "唯一封面", "COVER")
    good_body = item("s1p1", "保留这个图解")
    bad = item("s2p1", "错误图解")
    bad["promptEn"] = ""
    with planner([[good_cover, good_body, bad], [item("s2p1", "修复后的检索图")]]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert [p.image_title for p in plans] == ["唯一封面", "保留这个图解", "修复后的检索图"]
    assert len(calls) == 2
    repair_prompt = calls[1]["messages"][0]["content"]
    assert "s2p1" in repair_prompt and "输入关键词，找到相关笔记" in repair_prompt
    assert "保留这个图解" not in repair_prompt
    assert "唯一封面" not in repair_prompt


def test_truncated_json_preserves_complete_items_and_repairs_only_missing_slots() -> None:
    first = "[" + json.dumps(item("cover", "保留封面", "COVER"), ensure_ascii=False) + ', {"slotId":"s1p1"'
    with planner([first, [item("s1p1", "收集图"), item("s2p1", "检索图")]]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert [p.image_title for p in plans] == ["保留封面", "收集图", "检索图"]
    assert len(calls) == 2
    assert "保留封面" not in calls[1]["messages"][0]["content"]


def test_failed_optional_repair_keeps_a_usable_plan_instead_of_discarding_all_items() -> None:
    with planner([[item("cover", "封面", "COVER"), item("s1p1", "有效图解")], httpx.ReadTimeout("diagnostic-secret")]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert [p.image_title for p in plans] == ["封面", "有效图解"]
    assert len(calls) == 2
    assert "2/3" in agent.image_plan_diagnostics
    assert "diagnostic-secret" not in agent.image_plan_diagnostics


def test_plan_cannot_succeed_without_a_valid_cover_after_bounded_repair() -> None:
    with planner([[item("s1p1", "正文图")], []]) as (agent, calls):
        with pytest.raises(AdapterResponseValidationError, match="封面"):
            agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert len(calls) == 2


def test_invalid_roles_are_repaired_without_inventing_a_provider_routing_choice() -> None:
    cover = item("cover", "封面", "DIAGRAM")
    body = item("s1p1", "正文图")
    del body["visualRole"]
    with planner([[cover, body, item("s2p1", "有效图")], [item("cover", "封面", "COVER"), item("s1p1", "正文图")]]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert len(calls) == 2
    assert [p.visual_role.value for p in plans] == ["COVER", "DIAGRAM", "DIAGRAM"]
    assert "有效图" not in calls[1]["messages"][0]["content"]


def test_repair_does_not_attach_ambiguous_legacy_items_to_a_different_section() -> None:
    legacy_cover = item("cover", "重复封面", "COVER")
    legacy_body = item("s1p1", "不属于检索章节的图片")
    del legacy_cover["slotId"], legacy_body["slotId"]
    with planner([[item("cover", "保留封面", "COVER")], [legacy_cover, legacy_body]]) as (agent, calls):
        with pytest.raises(AdapterResponseValidationError, match="正文配图"):
            agent.image_prompts("知识库", "知识库", BODY, "TECH", 3)

    assert len(calls) == 2
    assert "1/3" in agent.image_plan_diagnostics


def test_plan_diagnostics_include_call_usage_and_do_not_include_content_or_secrets() -> None:
    with planner([[item("cover", "不应进入日志的标题", "COVER")]]) as (agent, _calls):
        agent.image_prompts("知识库", "知识库", BODY, "TECH", 1)

    diagnostics = agent.image_plan_diagnostics
    assert "首次生成" in diagnostics and "400" in diagnostics and "180" in diagnostics
    assert "stop" in diagnostics
    assert "diagnostic-secret" not in diagnostics and "不应进入日志的标题" not in diagnostics


def test_plan_request_does_not_repeat_the_full_body_or_require_presentation_metadata() -> None:
    content = "# 知识库\n\n简介。\n\n## 收集\n\n" + "按主题整理笔记。" * 800
    with planner([[item("cover", "封面", "COVER"), item("s1p1", "正文图")]]) as (agent, calls):
        plans = agent.image_prompts("知识库", "知识库", content, "TECH", 3)

    assert len(plans) == 2  # Only two distinct, meaningful insertion positions exist.
    assert len(calls) == 1
    prompt = calls[0]["messages"][0]["content"]
    assert len(prompt) < 2500
    assert content not in prompt
    assert "anchorParagraph" not in prompt and "stockQuery" not in prompt


@pytest.mark.parametrize("fails", [False, True])
def test_workflow_persists_safe_image_plan_diagnostics_on_success_and_failure(tmp_path, fails: bool) -> None:
    from ai_article_studio.db import create_session_factory
    from ai_article_studio.providers import ImageGenerationService
    from ai_article_studio.workflow import ArticleWorkflow

    replies = [[], []] if fails else [[item("cover", "秘密标题", "COVER"), item("s1p1", "收集图"), item("s2p1", "检索图")]]
    sessions = create_session_factory(f"sqlite:///{tmp_path / 'diagnostics.db'}")
    with planner(replies) as (agent, _calls):
        workflow = ArticleWorkflow(sessions, agent.settings, agent, ImageGenerationService({}))
        task_id = workflow.create("知识库", config=agent.model_config)
        workflow.confirm_title(task_id, "知识库")
        workflow.save_content(task_id, BODY)
        if fails:
            with pytest.raises(AdapterResponseValidationError):
                workflow.generate_image_prompts(task_id)
        else:
            workflow.generate_image_prompts(task_id)
        log = workflow.logs(task_id)[0]

    assert log.agent_name == "ImagePromptAgent"
    assert log.status == ("FAILED" if fails else "SUCCESS")
    assert "首次生成" in (log.output_text or "")
    assert "400" in log.output_text and "180" in log.output_text
    assert ("局部修复" in log.output_text) is fails
    assert "diagnostic-secret" not in log.output_text and "秘密标题" not in log.output_text

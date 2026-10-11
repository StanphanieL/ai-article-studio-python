from __future__ import annotations

import httpx
import pytest

from ai_article_studio.config import Settings
from ai_article_studio.llm import MockArticleAgent, OpenAICompatibleArticleAgent
from ai_article_studio.schemas import ArticleModelConfig, ImagePromptOption, ImageVisualRole, OutlineItem


def test_structured_output_uses_json_repair_after_parse_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = '{"broken":' if calls == 1 else '{"fixed": true}'
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    settings = Settings(openai_api_key="test-key", _env_file=None)
    agent = OpenAICompatibleArticleAgent(settings)
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert agent._json("返回 JSON") == {"fixed": True}
    assert calls == 2


def test_structured_output_accepts_a_complete_json_value_with_trailing_prose() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '[{"title":"标题","reason":"原因"}]\n以上是结果。'}}]},
            request=request,
        )

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert agent.titles("选题", "TECH")[0].title == "标题"
    assert calls == 1


def test_real_stream_content_reads_openai_compatible_sse_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.content and b'"stream":true' in request.content
        return httpx.Response(
            200,
            content=(b'data: {"choices":[{"delta":{"content":"# hello"}}]}\n\ndata: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n'),
            request=request,
        )

    settings = Settings(openai_api_key="test-key", _env_file=None)
    agent = OpenAICompatibleArticleAgent(settings)
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert "".join(agent.stream_content("topic", "title", [], "TECH", "SHORT")) == "# hello world"


def test_openai_compatible_base_url_already_ending_in_v1_is_not_duplicated() -> None:
    requested_path = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_path
        requested_path = request.url.path
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=request,
        )

    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        _env_file=None,
    )
    agent = OpenAICompatibleArticleAgent(settings)
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    assert agent._chat("hello") == "ok"
    assert requested_path == "/compatible-mode/v1/chat/completions"


def test_content_prompts_require_strict_markdown_for_standard_and_streaming_generation() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        if payload.get("stream"):
            return httpx.Response(
                200,
                content=b'data: {"choices":[{"delta":{"content":"# title"}}]}\n\ndata: [DONE]\n\n',
                request=request,
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "# title"}}]}, request=request)

    settings = Settings(openai_api_key="test-key", _env_file=None)
    agent = OpenAICompatibleArticleAgent(settings)
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))
    outline = [OutlineItem(heading="步骤", description="说明")]

    assert agent.content("选题", "标题", outline, "TECH", "SHORT") == "# title"
    assert "".join(agent.stream_content("选题", "标题", outline, "TECH", "SHORT")) == "# title"

    assert len(prompts) == 2
    for prompt in prompts:
        assert "唯一的一级标题" in prompt
        assert "二级标题" in prompt
        assert "围栏代码块" in prompt
        assert "裸 HTML" in prompt


def test_generation_prompts_share_the_no_invented_resources_contract() -> None:
    prompts: list[str] = []
    replies = iter(
        [
            '[{"title": "标题", "reason": "原因"}]',
            '[{"heading": "步骤", "description": "说明"}]',
            "# 标题\n\n## 步骤\n\n正文。",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        prompts.append(payload["messages"][0]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    agent.titles("如何从零开始健身", "POPULAR")
    agent.outline("如何从零开始健身", "7 天启动计划", "POPULAR")
    agent.content("如何从零开始健身", "7 天启动计划", [OutlineItem(heading="步骤", description="说明")], "POPULAR", "SHORT")

    assert len(prompts) == 3
    assert all("不得虚构" in prompt for prompt in prompts)
    assert all("二维码" in prompt for prompt in prompts)
    assert "JSON 代码块" in prompts[-1]


def test_review_repairs_a_parseable_payload_that_breaks_the_required_string_array_schema() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = (
            '{"score": 70, "passed": false, "summary": "需修订", '
            '"blockingIssues": [{"type": "事实", "detail": "无来源背书"}], '
            '"issues": [{"detail": "结构可改进"}], "suggestions": []}'
            if calls == 1
            else '{"score": 70, "passed": false, "summary": "需修订", '
            '"revisionAction": "REGENERATE_CONTENT", "revisionReason": "删除无来源背书", '
            '"blockingIssues": ["无来源背书"], "issues": ["结构可改进"], "suggestions": []}'
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    review = agent.review("# 标题\n\n## 小节\n\n正文")

    assert calls == 2
    assert review.blocking_issues == ["无来源背书"]
    assert review.issues == ["结构可改进"]


def test_real_requests_use_the_article_model_configuration() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))
    configured = agent.with_model_config(ArticleModelConfig(text_model="qwen-plus", temperature=0.2, max_completion_tokens=1024))

    assert configured._chat("hello") == "ok"
    assert captured["model"] == "qwen-plus"
    assert captured["temperature"] == 0.2
    assert captured["max_completion_tokens"] == 1024


def test_legacy_image_prompt_json_receives_semantic_plan_defaults() -> None:
    prompt = ImagePromptOption.model_validate(
        {
            "imageTitle": "旧封面",
            "usageScene": "文章封面",
            "promptZh": "简洁的文章封面",
            "promptEn": "clean article cover",
        }
    )

    assert prompt.visual_role == "CONCEPT"
    assert prompt.placement == "SECTION_END"
    assert prompt.anchor_heading == ""
    assert prompt.anchor_paragraph == 0
    assert prompt.layout == "STANDARD"
    assert prompt.stock_query == ""


def test_mock_image_plans_assign_distinct_roles_anchors_and_layouts() -> None:
    content = "# Design systems\n\nIntroduction.\n\n## Principles\n\nExplanation.\n\n## Workflows\n\nSteps."

    plans = MockArticleAgent().image_prompts("design systems", "Design systems", content, "TECH", 3)

    assert [plan.visual_role for plan in plans] == ["COVER", "CONCEPT", "CLOSING"]
    assert [plan.placement for plan in plans] == ["COVER", "AFTER_PARAGRAPH", "SECTION_END"]
    assert [plan.anchor_heading for plan in plans] == ["", "Principles", "Workflows"]
    assert [plan.anchor_paragraph for plan in plans] == [0, 1, 0]
    assert [plan.layout for plan in plans] == ["HERO", "WIDE", "COMPACT"]
    assert all(2 <= len(plan.stock_query.split()) <= 5 for plan in plans)


def test_mock_image_plans_cover_the_full_eight_image_limit_without_reusing_roles() -> None:
    content = "# Eight images\n\n" + "\n\n".join(f"## Section {index}\n\nParagraph {index}." for index in range(1, 8))

    plans = MockArticleAgent().image_prompts("topic", "title", content, "TECH", 8)

    assert len(plans) == 8
    assert len({plan.visual_role for plan in plans}) == 8
    assert ImageVisualRole.DETAIL in {plan.visual_role for plan in plans}
    assert ImageVisualRole.DATA in {plan.visual_role for plan in plans}


def test_real_image_plan_prompt_uses_compact_grounded_briefs_and_server_placement() -> None:
    captured: list[str] = []
    response = (
        '[{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover",'
        '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
        '"layout":"HERO","stockQuery":"design workspace"}]'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(__import__("json").loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems\n\n## Foundations\n\nOne paragraph.", "TECH", 1)

    assert plans[0].visual_role == "COVER"
    assert plans[0].placement == "COVER"
    assert plans[0].stock_query == "design workspace"
    assert "visualRole" in captured[0]
    assert "非封面角色可以重复" in captured[0]
    assert "stockQuery" in captured[0]
    assert "slotId" in captured[0]
    assert "anchorParagraph" not in captured[0]
    assert "正文摘录" in captured[0]
    assert "One paragraph." in captured[0]
    assert "不得凭空加入实验室、XRD/衍射" in captured[0]
    assert "Foundations" in captured[0]


def test_real_image_plan_repairs_a_specific_scene_that_the_article_never_mentions() -> None:
    calls: list[str] = []
    responses = iter(
        [
            (
                '[{"imageTitle":"实验设备","usageScene":"表现可重复性失效",'
                '"promptZh":"实验室内三台 XRD 设备并排",'
                '"promptEn":"three XRD machines in a laboratory, no text",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                '"layout":"HERO","stockQuery":"laboratory xrd equipment"}]'
            ),
            (
                '[{"slotId":"cover","imageTitle":"重复流程的偏离","usageScene":"让读者看懂同一流程重复运行后出现偏离",'
                '"promptZh":"俯视视角，同一套简洁流程卡片连续重复三次，第三次路径向右分叉偏离，留白背景，no text，no logo，no watermark",'
                '"promptEn":"top down repeated process cards with the third path branching right, clean editorial diagram, no text, no logo, no watermark",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                '"layout":"HERO","stockQuery":"repeated process branching"}]'
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(__import__("json").loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts(
        "如何处理可重复性失效",
        "可重复性失效时，我们还在坚持什么？",
        "# 可重复性\n\n## 偏离如何出现\n\n同一流程重复运行后出现偏离，需要记录偏离的位置。",
        "TECH",
        1,
    )

    assert len(calls) == 2
    assert "XRD/衍射设备" in calls[1]
    assert "XRD" not in plans[0].prompt_zh
    assert plans[0].stock_query == "repeated process branching"


def test_real_image_plan_drops_extra_items_without_a_full_array_repair() -> None:
    calls = 0
    cover = (
        '{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover",'
        '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
        '"layout":"HERO","stockQuery":"design workspace"}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        response = f"[{cover},{cover}]" if calls == 1 else f"[{cover}]"
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert len(plans) == 1
    assert calls == 1


def test_real_image_plan_fills_legacy_cover_metadata_without_an_extra_model_call() -> None:
    calls: list[str] = []
    responses = iter(
        [
            '[{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover"}]',
            (
                '[{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                '"layout":"HERO","stockQuery":"design workspace"}]'
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(__import__("json").loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert len(calls) == 1
    assert plans[0].visual_role == "COVER"
    assert plans[0].placement == "COVER" and plans[0].layout == "HERO"


def test_real_image_plan_uses_server_cover_placement_without_an_extra_model_call() -> None:
    calls = 0
    responses = iter(
        [
            (
                '[{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover",'
                '"visualRole":"COVER","placement":"SECTION_END","anchorHeading":"结尾","anchorParagraph":1,'
                '"layout":"HERO","stockQuery":"design workspace"}]'
            ),
            (
                '[{"imageTitle":"封面","usageScene":"文章封面","promptZh":"封面","promptEn":"cover",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                '"layout":"HERO","stockQuery":"design workspace"}]'
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert calls == 1
    assert plans[0].placement == "COVER"
    assert plans[0].anchor_heading == "" and plans[0].anchor_paragraph == 0


@pytest.mark.parametrize(
    ("invalid_stock_query", "expected_stock_query"),
    [
        ("设计师 工作区", "editorial article cover"),
        ("designer", "designer concept"),
        ("designer workspace creative studio editorial photo", "designer workspace creative studio editorial"),
    ],
)
def test_real_image_plan_normalizes_nonconforming_stock_queries_without_an_extra_model_call(
    invalid_stock_query: str,
    expected_stock_query: str,
) -> None:
    calls = 0
    responses = iter(
        [
            (
                '[{"imageTitle":"概念图","usageScene":"导语后","promptZh":"概念图","promptEn":"concept",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                f'"layout":"STANDARD","stockQuery":"{invalid_stock_query}"}}]'
            ),
            (
                '[{"imageTitle":"概念图","usageScene":"导语后","promptZh":"概念图","promptEn":"concept",'
                '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
                '"layout":"STANDARD","stockQuery":"designer workspace"}]'
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": next(responses)}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert calls == 1
    assert plans[0].stock_query == expected_stock_query


def test_real_image_plan_normalizes_a_non_ascii_stock_query_from_the_english_prompt() -> None:
    """A valid semantic plan should not fail solely because its stock query is Chinese."""

    calls = 0
    response = (
        '[{"imageTitle":"概念图","usageScene":"导语后","promptZh":"概念图","promptEn":"abstract concept art",'
        '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
        '"layout":"STANDARD","stockQuery":"抽象 概念 艺术"}]'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert calls == 1
    assert plans[0].stock_query == "abstract concept art"


def test_real_image_plan_uses_a_role_fallback_when_no_english_stock_terms_are_available() -> None:
    """An all-Chinese model output still produces a usable Pexels query."""

    response = (
        '[{"imageTitle":"概念图","usageScene":"导语后","promptZh":"概念图","promptEn":"概念插画",'
        '"visualRole":"COVER","placement":"COVER","anchorHeading":"","anchorParagraph":0,'
        '"layout":"STANDARD","stockQuery":"抽象 概念 艺术"}]'
    )

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": response}}]}, request=request))
    )

    plans = agent.image_prompts("design systems", "Design systems", "# Design systems", "TECH", 1)

    assert plans[0].stock_query == "editorial article cover"

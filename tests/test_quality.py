from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_article_studio.quality import content_contract_issues, markdown_blocking_issues
from ai_article_studio.schemas import QualityReviewResult
from ai_article_studio.supervisor import ArticlePhase, ArticleStatus, SupervisorAction, decide_next_action


def test_quality_gate_accepts_structured_markdown_and_valid_json_code() -> None:
    article = """# 可交付文章

开场说明。

## 实践步骤

```json
{"enabled": true}
```

```python
items = ["design", "verify"]
print(items)
```
"""

    assert markdown_blocking_issues(article) == []


def test_content_contract_blocks_invented_resources_social_proof_and_json_in_a_non_technical_article() -> None:
    article = """# 从零开始健身

## 第一步

扫码领取动作图解库；372 位首发体验者都说亲测有效。

```json
{"knee_collapse": {"fix": "弹力带套膝上"}}
```
"""

    issues = content_contract_issues(
        article,
        topic="如何从零开始健身",
        title="从沙发到深蹲：零基础健身计划",
        style="POPULAR",
    )

    assert any("虚构或未提供的外部资源" in issue for issue in issues)
    assert any("无来源的用户数、亲测或权威背书" in issue for issue in issues)
    assert any("JSON 代码块" in issue for issue in issues)


@pytest.mark.parametrize("claim", ["90% 的笔记工具", "超过 70％ 的用户", "百分之九十的团队"])
def test_content_contract_blocks_unsourced_quantitative_claims(claim: str) -> None:
    article = f"""# 知识库设计

## 常见误区

{claim} 都会遇到检索问题。
"""

    issues = content_contract_issues(
        article,
        topic="如何构建个人知识库",
        title="个人知识库的设计边界",
        style="TECH",
    )

    assert any("无来源的比例或量化结论" in issue for issue in issues)


def test_content_contract_keeps_versions_steps_and_dates_available() -> None:
    article = """# FastAPI 实践

## 第 2 步

使用 Python 3.13，并在 2026 年 7 月完成验证。
"""

    assert not any(
        "无来源的比例或量化结论" in issue
        for issue in content_contract_issues(
            article,
            topic="FastAPI 教程",
            title="FastAPI 实践",
            style="TECH",
        )
    )


def test_content_contract_keeps_fenced_code_available_for_a_clearly_technical_topic() -> None:
    article = """# FastAPI 流式接口

## SSE 示例

```python
yield "event: done"
```
"""

    issues = content_contract_issues(
        article,
        topic="用 Python 和 FastAPI 实现 SSE 流式接口",
        title="FastAPI SSE 实战",
        style="TECH",
    )

    assert not any("代码围栏" in issue for issue in issues)


def test_quality_gate_accepts_standard_markdown_autolinks() -> None:
    article = """# 可交付文章

详细说明见 <https://example.com/docs>。

## 实践步骤

按链接继续阅读。
"""

    assert markdown_blocking_issues(article) == []


def test_quality_gate_does_not_mistake_inline_code_examples_for_bare_html() -> None:
    article = """# 可交付文章

使用 `<div className=\"grid\"><motion.div /></div>` 描述 JSX 结构。

## 实践步骤

将同样的结构保留在行内代码中。
"""

    assert markdown_blocking_issues(article) == []


def test_quality_gate_rejects_malformed_fence_openers_without_hiding_later_html() -> None:
    article = """# 可交付文章

## 样式示例

```css .grid {
  display: grid;
}

<div>后续的裸 HTML 仍应被检查</div>
"""

    issues = markdown_blocking_issues(article)

    assert "代码围栏开头格式不正确。" in issues
    assert "正文包含裸 HTML；请改用标准 Markdown。" in issues


def test_quality_gate_accepts_a_language_free_fence_and_its_closing_marker() -> None:
    article = """# 可交付文章

## 摘录

```
plain text example
```
"""

    assert markdown_blocking_issues(article) == []


@pytest.mark.parametrize(
    ("article", "expected"),
    [
        ("没有一级标题\n\n## 一个小节", "一级标题"),
        ("# 标题一\n\n# 标题二\n\n## 一个小节", "只能包含一个一级标题"),
        ("# 标题\n\n只有正文，没有小节", "二级标题"),
        ("# 标题\n\n<div>裸 HTML</div>\n\n## 小节", "裸 HTML"),
        ("# 标题\n\n## 小节\n\n```python\nprint('unfinished')", "未闭合"),
        ("# 标题\n\n## 小节\n\n```json\n{broken}\n```", "JSON"),
        ("# 标题\n\n## 小节\n\n```javascript\nrepeat(...))\n```", "括号"),
    ],
)
def test_quality_gate_reports_blocking_markdown_defects(article: str, expected: str) -> None:
    assert any(expected in issue for issue in markdown_blocking_issues(article))


def test_quality_review_supports_legacy_payload_and_serializes_blocking_issues() -> None:
    legacy = QualityReviewResult.model_validate_json('{"score": 94, "passed": true, "summary": "旧数据"}')

    assert legacy.blocking_issues == []
    assert QualityReviewResult(score=94, passed=True, summary="有问题", blocking_issues=["缺少 H1"]).model_dump(by_alias=True)["blockingIssues"] == ["缺少 H1"]


def test_blocking_review_cannot_complete_even_with_a_high_score() -> None:
    state = SimpleNamespace(
        phase=ArticlePhase.ARTICLE_COMPOSITION,
        status=ArticleStatus.SUCCESS,
        selected_title="标题",
        outline="[]",
        content="# 标题\n\n## 小节",
        image_prompts="[]",
        image_results="[]",
        final_markdown="# 标题\n\n## 小节",
        review_round=0,
    )
    review = QualityReviewResult(
        score=94,
        passed=True,
        summary="模型误判为可交付",
        blocking_issues=["正文缺少有效内容"],
    )

    decision = decide_next_action(state, review, max_auto_review_rounds=1)

    assert decision.action is SupervisorAction.REGENERATE_CONTENT
    assert decision.auto_revision is True
    assert decision.increment_review_round is True

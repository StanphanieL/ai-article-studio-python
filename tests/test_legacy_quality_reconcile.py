from __future__ import annotations

import json
from pathlib import Path

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.schemas import QualityReviewResult
from ai_article_studio.supervisor import ArticlePhase, ArticleStatus, SupervisorAction


def _app(tmp_path: Path):
    from ai_article_studio.main import create_app

    return create_app(
        database_url=f"sqlite:///{tmp_path / 'articles.db'}",
        upload_dir=tmp_path / "images",
        settings=Settings(app_mode=RuntimeMode.MOCK),
    )


def _write_legacy_completed_article(app, markdown: str, review: dict[str, object], review_round: int = 0) -> str:  # type: ignore[no-untyped-def]
    workflow = app.state.workflow
    task_id = workflow.create("历史质量校正")
    workflow._update(  # noqa: SLF001 - prepares a persisted pre-quality-gate record.
        task_id,
        final_markdown=markdown,
        review_report=json.dumps(review, ensure_ascii=False),
        quality_score=review["score"],
        review_round=review_round,
        phase=ArticlePhase.COMPLETED.value,
        status=ArticleStatus.SUCCESS.value,
        supervisor_action=SupervisorAction.COMPLETE.value,
        supervisor_reason="质量评审通过，文章可以交付",
    )
    return task_id


def test_startup_reconciles_bad_legacy_completed_review_without_deleting_delivery_artifacts(tmp_path: Path) -> None:
    first_app = _app(tmp_path)
    legacy_markdown = "# 历史文章\n\n这里只有正文，没有二级标题。"
    legacy_report = {"score": 94, "passed": True, "summary": "旧评审错误地判定可交付"}
    task_id = _write_legacy_completed_article(first_app, legacy_markdown, legacy_report)

    restarted_app = _app(tmp_path)
    article = restarted_app.state.workflow.get(task_id)
    review = QualityReviewResult.model_validate_json(article.review_report or "")

    assert article.final_markdown == legacy_markdown
    assert article.phase == ArticlePhase.ARTICLE_COMPOSITION
    assert article.status == ArticleStatus.SUCCESS
    assert article.quality_score == 94
    assert review.passed is False
    assert any("二级标题" in issue for issue in review.blocking_issues)
    assert article.supervisor_action == SupervisorAction.REGENERATE_CONTENT
    assert article.supervisor_reason


def test_startup_keeps_valid_legacy_completed_review_unchanged(tmp_path: Path) -> None:
    first_app = _app(tmp_path)
    legacy_markdown = "# 合格历史文章\n\n## 正文\n\n内容结构完整。"
    legacy_report = {"score": 94, "passed": True, "summary": "可交付"}
    task_id = _write_legacy_completed_article(first_app, legacy_markdown, legacy_report)

    restarted_app = _app(tmp_path)
    article = restarted_app.state.workflow.get(task_id)

    assert article.final_markdown == legacy_markdown
    assert article.review_report == json.dumps(legacy_report, ensure_ascii=False)
    assert article.phase == ArticlePhase.COMPLETED
    assert article.status == ArticleStatus.SUCCESS
    assert article.supervisor_action == SupervisorAction.COMPLETE


def test_startup_removes_a_stale_inline_code_html_gate_finding_but_keeps_real_defects(tmp_path: Path) -> None:
    first_app = _app(tmp_path)
    markdown = """# 历史文章

## 代码示例

行内 JSX：`<div className=\"grid\"><motion.div /></div>`。

```css
.grid { grid-template-columns: repeat(2, 1fr)); }
```
"""
    task_id = _write_legacy_completed_article(
        first_app,
        markdown,
        {
            "score": 94,
            "passed": False,
            "summary": "历史检测结果",
            "blockingIssues": ["正文包含裸 HTML；请改用标准 Markdown。", "代码样式内容存在明显的括号不平衡。"],
        },
    )

    restarted_app = _app(tmp_path)
    article = restarted_app.state.workflow.get(task_id)
    review = QualityReviewResult.model_validate_json(article.review_report or "")

    assert review.blocking_issues == [
        "代码样式内容存在明显的括号不平衡。",
        "当前选题不需要代码围栏；请改用自然语言、列表或 Markdown 表格。",
    ]
    assert article.phase == ArticlePhase.ARTICLE_COMPOSITION
    assert article.supervisor_action == SupervisorAction.REGENERATE_CONTENT


def test_startup_leaves_unparseable_legacy_review_untouched(tmp_path: Path) -> None:
    first_app = _app(tmp_path)
    workflow = first_app.state.workflow
    task_id = workflow.create("无法解析的历史评审")
    bad_report = "{not valid json"
    workflow._update(  # noqa: SLF001 - persists an old malformed record for startup recovery.
        task_id,
        final_markdown="# 历史文章\n\n正文缺少二级标题。",
        review_report=bad_report,
        quality_score=94,
        phase=ArticlePhase.COMPLETED.value,
        status=ArticleStatus.SUCCESS.value,
        supervisor_action=SupervisorAction.COMPLETE.value,
        supervisor_reason="质量评审通过，文章可以交付",
    )

    restarted_app = _app(tmp_path)
    article = restarted_app.state.workflow.get(task_id)

    assert article.review_report == bad_report
    assert article.phase == ArticlePhase.COMPLETED
    assert article.status == ArticleStatus.SUCCESS
    assert article.supervisor_action == SupervisorAction.COMPLETE


def test_startup_never_reconciles_a_persisted_running_job(tmp_path: Path) -> None:
    first_app = _app(tmp_path)
    legacy_markdown = "# 正在执行\n\n正文缺少二级标题。"
    legacy_report = {"score": 94, "passed": True, "summary": "旧评审错误"}
    task_id = _write_legacy_completed_article(first_app, legacy_markdown, legacy_report)
    first_app.state.workflow._update(  # noqa: SLF001 - simulates an interrupted long-running action.
        task_id,
        status=ArticleStatus.RUNNING.value,
        supervisor_action=SupervisorAction.GENERATE_CONTENT.value,
        supervisor_reason="任务已开始：生成正文",
    )

    restarted_app = _app(tmp_path)
    article = restarted_app.state.workflow.get(task_id)

    assert article.status == ArticleStatus.FAILED
    assert article.phase == ArticlePhase.COMPLETED
    assert article.supervisor_action == SupervisorAction.GENERATE_CONTENT
    assert article.review_report == json.dumps(legacy_report, ensure_ascii=False)

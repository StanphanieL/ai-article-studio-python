"""Drafts remain editable until the author explicitly requests delivery review."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.db import create_session_factory
from ai_article_studio.domain import WorkflowValidationError
from ai_article_studio.llm import MockArticleAgent
from ai_article_studio.providers import create_image_service
from ai_article_studio.schemas import ArticleModelConfig, ImageCountMode, ImageProvider, OutlineItem, QualityReviewResult
from ai_article_studio.workflow import ArticleWorkflow

UNREVIEWED_DRAFT = "# 知识库设计\n\n## 常见问题\n\n90% 的用户都会遇到检索问题。"
CORRECTED_DRAFT = "# 知识库设计\n\n## 常见问题\n\n先整理检索需求，再逐步验证方案。"


class DraftAgent(MockArticleAgent):
    def __init__(self) -> None:
        self.draft = UNREVIEWED_DRAFT
        self.repair_calls = 0
        self.review_calls = 0

    def content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str:
        return self.draft

    def repair_content(self, *args: object) -> str:
        self.repair_calls += 1
        return CORRECTED_DRAFT

    def review(self, article: str) -> QualityReviewResult:
        self.review_calls += 1
        return QualityReviewResult(score=95, passed=True, summary="模型认为稿件可以交付")


@pytest.fixture
def workflow(tmp_path: Path) -> Iterator[ArticleWorkflow]:
    settings = Settings(_env_file=None, app_mode=RuntimeMode.MOCK, upload_dir=tmp_path / "images", vision_model="")
    sessions = create_session_factory(f"sqlite:///{tmp_path / 'articles.db'}")
    yield ArticleWorkflow(sessions, settings, DraftAgent(), create_image_service(settings))
    sessions.kw["bind"].dispose()


def prepare_content_task(workflow: ArticleWorkflow) -> str:
    task_id = workflow.create(
        "如何构建个人知识库",
        config=ArticleModelConfig(image_provider=ImageProvider.MOCK, image_count_mode=ImageCountMode.CUSTOM, image_count=3),
    )
    workflow.confirm_title(task_id, "知识库设计")
    workflow.save_outline(task_id, [OutlineItem(heading="常见问题", description="分析检索需求与实现步骤")])
    return task_id


@pytest.mark.parametrize("mode", ["normal", "stream", "edit"])
def test_body_paths_save_unreviewed_text_without_quality_repair(workflow: ArticleWorkflow, mode: str) -> None:
    task_id = prepare_content_task(workflow)
    if mode == "normal":
        workflow.generate_content(task_id)
    elif mode == "stream":
        workflow.start_stream_content(task_id)
        events = list(workflow.stream_content(task_id))
        assert events[0].startswith("event: progress")
        assert events[-1].startswith("event: done")
        assert not any("event: content_replace" in event or "event: fail" in event for event in events)
    else:
        workflow.save_content(task_id, UNREVIEWED_DRAFT)

    saved = workflow.get(task_id)
    assert saved.content == UNREVIEWED_DRAFT
    assert saved.status == "SUCCESS"
    assert saved.review_report is None
    assert saved.quality_score is None
    assert workflow.agent.repair_calls == 0
    assert workflow.agent.review_calls == 0


@pytest.mark.parametrize("mode", ["normal", "stream", "edit"])
def test_empty_body_still_fails_without_invoking_quality_repair(workflow: ArticleWorkflow, mode: str) -> None:
    task_id = prepare_content_task(workflow)
    workflow.agent.draft = " \n "
    if mode == "stream":
        workflow.start_stream_content(task_id)
        events = list(workflow.stream_content(task_id))
        assert events[-1].startswith("event: fail")
        assert not any("event: done" in event for event in events)
    else:
        with pytest.raises(WorkflowValidationError, match="正文不能为空"):
            if mode == "normal":
                workflow.generate_content(task_id)
            else:
                workflow.save_content(task_id, " \n ")
    assert workflow.get(task_id).content is None
    assert workflow.agent.repair_calls == 0


def test_one_click_waits_for_explicit_review_and_keeps_quality_blockers(workflow: ArticleWorkflow) -> None:
    task_id = prepare_content_task(workflow)
    draft = workflow.one_click(task_id)

    assert draft.content == UNREVIEWED_DRAFT
    assert draft.final_markdown
    assert draft.phase == "ARTICLE_COMPOSITION"
    assert draft.status == "SUCCESS"
    assert draft.review_report is None
    assert draft.supervisor_action == "REVIEW_ARTICLE"
    assert workflow.agent.review_calls == 0
    assert not any(log.agent_name == "QualityReviewAgent" for log in workflow.logs(task_id))

    repeated = workflow.one_click(task_id)
    assert repeated.final_markdown == draft.final_markdown
    assert repeated.review_report is None
    assert workflow.agent.review_calls == 0

    reviewed = workflow.review(task_id)
    review = QualityReviewResult.model_validate_json(reviewed.review_report)
    assert workflow.agent.review_calls == 1
    assert review.passed is False
    assert any("无来源的比例" in issue for issue in review.blocking_issues)
    assert reviewed.phase == "ARTICLE_COMPOSITION"
    assert reviewed.content == UNREVIEWED_DRAFT
    assert reviewed.review_round == 0


def test_revision_recomposes_but_waits_for_a_fresh_explicit_review(workflow: ArticleWorkflow) -> None:
    task_id = prepare_content_task(workflow)
    workflow.one_click(task_id)
    workflow.review(task_id)
    workflow.agent.draft = CORRECTED_DRAFT

    revised = workflow.apply_review_revision(task_id)

    assert revised.content == CORRECTED_DRAFT
    assert revised.final_markdown
    assert revised.phase == "ARTICLE_COMPOSITION"
    assert revised.review_report is None
    assert revised.review_round == 1
    assert workflow.agent.review_calls == 1

    reviewed = workflow.review(task_id)
    assert reviewed.phase == "COMPLETED"
    assert QualityReviewResult.model_validate_json(reviewed.review_report).passed is True
    assert workflow.agent.review_calls == 2


def test_one_click_does_not_implicitly_revise_an_already_reviewed_draft(workflow: ArticleWorkflow) -> None:
    task_id = prepare_content_task(workflow)
    workflow.one_click(task_id)
    reviewed = workflow.review(task_id)

    unchanged = workflow.one_click(task_id)

    assert unchanged.phase == "ARTICLE_COMPOSITION"
    assert unchanged.review_report == reviewed.review_report
    assert unchanged.review_round == 0
    assert workflow.agent.review_calls == 1


def test_background_revision_leaves_job_completion_to_the_runner(workflow: ArticleWorkflow) -> None:
    task_id = prepare_content_task(workflow)
    workflow.one_click(task_id)
    workflow.review(task_id)
    workflow.agent.draft = CORRECTED_DRAFT
    workflow.begin_job(task_id, "APPLY_REVIEW_REVISION", "正在按评审修订")
    workflow.mark_job_running(task_id)

    with workflow.background_job_execution():
        revised = workflow.apply_review_revision(task_id)

    assert revised.final_markdown
    assert revised.review_report is None
    assert revised.status == "RUNNING"
    assert revised.execution_mode == "STEP"
    finished = workflow.finish_job(task_id)
    assert finished.status == "SUCCESS"
    assert finished.execution_mode is None

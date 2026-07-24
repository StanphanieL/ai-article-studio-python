from __future__ import annotations

from types import SimpleNamespace

from ai_article_studio.schemas import QualityReviewResult
from ai_article_studio.supervisor import (
    ArticlePhase,
    ArticleStatus,
    SupervisorAction,
    decide_next_action,
)


def article_state(**overrides: object) -> SimpleNamespace:
    state = {
        "phase": ArticlePhase.CREATED,
        "status": ArticleStatus.INIT,
        "selected_title": None,
        "outline": None,
        "content": None,
        "image_prompts": None,
        "image_results": None,
        "final_markdown": None,
        "review_round": 0,
    }
    state.update(overrides)
    return SimpleNamespace(**state)


def test_created_task_generates_title_candidates_first() -> None:
    decision = decide_next_action(article_state())

    assert decision.action is SupervisorAction.GENERATE_TITLES
    assert decision.auto_revision is False


def test_title_selection_waits_until_a_user_confirms_a_title() -> None:
    decision = decide_next_action(article_state(phase=ArticlePhase.TITLE_SELECTION))

    assert decision.action is SupervisorAction.WAIT_FOR_TITLE


def test_confirmed_title_advances_to_outline_generation() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.TITLE_SELECTION,
            selected_title="可控的 AI 写作工作流",
        )
    )

    assert decision.action is SupervisorAction.GENERATE_OUTLINE


def test_outline_editing_without_a_generated_outline_generates_outline() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.OUTLINE_EDITING,
            selected_title="可控的 AI 写作工作流",
        )
    )

    assert decision.action is SupervisorAction.GENERATE_OUTLINE


def test_content_phase_prioritizes_missing_content_before_images() -> None:
    decision = decide_next_action(article_state(phase=ArticlePhase.CONTENT_GENERATION))

    assert decision.action is SupervisorAction.GENERATE_CONTENT


def test_composed_article_without_a_review_is_sent_for_review() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
        )
    )

    assert decision.action is SupervisorAction.REVIEW_ARTICLE
    assert decision.auto_revision is False


def test_failed_review_marks_only_regeneration_as_an_automatic_revision() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
        ),
        QualityReviewResult(
            passed=False,
            score=70,
            summary="配图不匹配",
            revision_action=SupervisorAction.REGENERATE_IMAGE_RESULTS,
        ),
    )

    assert decision.action is SupervisorAction.REGENERATE_IMAGE_RESULTS
    assert decision.auto_revision is True
    assert decision.increment_review_round is True


def test_failed_review_with_an_unrecognized_model_action_defaults_to_content_revision() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
        ),
        QualityReviewResult(
            passed=False,
            score=94,
            summary="需要细微修订",
            revision_action="minor",
        ),
    )

    assert decision.action is SupervisorAction.REGENERATE_CONTENT
    assert decision.auto_revision is True
    assert decision.increment_review_round is True


def test_passing_review_marks_article_complete() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
        ),
        QualityReviewResult(passed=True, score=92, summary="可交付"),
    )

    assert decision.action is SupervisorAction.COMPLETE


def test_failed_review_regenerates_requested_artifact_once() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
        ),
        QualityReviewResult(
            passed=False,
            score=70,
            summary="配图不匹配",
            revision_action=SupervisorAction.REGENERATE_IMAGE_RESULTS,
        ),
        max_auto_review_rounds=1,
    )

    assert decision.action is SupervisorAction.REGENERATE_IMAGE_RESULTS
    assert decision.auto_revision is True


def test_failed_review_at_revision_limit_requires_manual_review() -> None:
    decision = decide_next_action(
        article_state(
            phase=ArticlePhase.ARTICLE_COMPOSITION,
            final_markdown="# 完整图文稿",
            review_round=1,
        ),
        QualityReviewResult(
            passed=False,
            score=70,
            summary="配图不匹配",
            revision_action=SupervisorAction.REGENERATE_IMAGE_RESULTS,
        ),
        max_auto_review_rounds=1,
    )

    assert decision.action is SupervisorAction.MANUAL_REVIEW


def test_failed_task_stops_until_the_error_is_handled() -> None:
    decision = decide_next_action(article_state(status=ArticleStatus.FAILED))

    assert decision.action is SupervisorAction.STOP_FAILED

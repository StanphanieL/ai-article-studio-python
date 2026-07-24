"""Deterministic workflow routing; LLMs never decide process control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .schemas import QualityReviewResult


class ArticlePhase(StrEnum):
    CREATED = "CREATED"
    TITLE_SELECTION = "TITLE_SELECTION"
    OUTLINE_EDITING = "OUTLINE_EDITING"
    CONTENT_GENERATION = "CONTENT_GENERATION"
    ARTICLE_COMPOSITION = "ARTICLE_COMPOSITION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ArticleStatus(StrEnum):
    INIT = "INIT"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class SupervisorAction(StrEnum):
    GENERATE_TITLES = "GENERATE_TITLES"
    WAIT_FOR_TITLE = "WAIT_FOR_TITLE"
    GENERATE_OUTLINE = "GENERATE_OUTLINE"
    GENERATE_CONTENT = "GENERATE_CONTENT"
    GENERATE_IMAGE_PROMPTS = "GENERATE_IMAGE_PROMPTS"
    GENERATE_IMAGE_RESULTS = "GENERATE_IMAGE_RESULTS"
    COMPOSE_ARTICLE = "COMPOSE_ARTICLE"
    REVIEW_ARTICLE = "REVIEW_ARTICLE"
    REGENERATE_CONTENT = "REGENERATE_CONTENT"
    REGENERATE_IMAGE_PROMPTS = "REGENERATE_IMAGE_PROMPTS"
    REGENERATE_IMAGE_RESULTS = "REGENERATE_IMAGE_RESULTS"
    COMPLETE = "COMPLETE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    STOP_FAILED = "STOP_FAILED"


class ArticleState(Protocol):
    phase: ArticlePhase | str
    status: ArticleStatus | str
    selected_title: str | None
    outline: str | None
    content: str | None
    image_prompts: str | None
    image_results: str | None
    final_markdown: str | None
    review_round: int | None


@dataclass(frozen=True)
class SupervisorDecision:
    action: SupervisorAction
    reason: str
    auto_revision: bool = False
    review_round: int = 0
    increment_review_round: bool = False


def _decision(
    action: SupervisorAction,
    reason: str,
    auto_revision: bool = False,
    review_round: int = 0,
    increment_review_round: bool = False,
) -> SupervisorDecision:
    return SupervisorDecision(
        action,
        reason,
        auto_revision,
        review_round,
        increment_review_round,
    )


def decide_next_action(
    article: ArticleState,
    review_result: QualityReviewResult | None = None,
    max_auto_review_rounds: int = 1,
) -> SupervisorDecision:
    """Map persisted article state to the only safe next workflow operation."""

    phase = ArticlePhase(article.phase)
    status = ArticleStatus(article.status)
    review_round = max(article.review_round or 0, 0)
    if status is ArticleStatus.CANCELLING:
        return _decision(
            SupervisorAction.STOP_FAILED,
            "任务正在取消并永久删除",
            review_round=review_round,
        )
    if status is ArticleStatus.FAILED:
        return _decision(
            SupervisorAction.STOP_FAILED,
            "当前任务处于失败状态，请先处理失败原因",
            review_round=review_round,
        )
    if phase is ArticlePhase.CREATED:
        return _decision(
            SupervisorAction.GENERATE_TITLES,
            "任务刚创建，下一步生成标题候选",
            review_round=review_round,
        )
    if phase is ArticlePhase.TITLE_SELECTION:
        if not article.selected_title:
            return _decision(
                SupervisorAction.WAIT_FOR_TITLE,
                "等待用户选择标题",
                review_round=review_round,
            )
        return _decision(
            SupervisorAction.GENERATE_OUTLINE,
            "标题已确认，下一步生成文章大纲",
            review_round=review_round,
        )
    if phase is ArticlePhase.OUTLINE_EDITING:
        if not article.outline:
            return _decision(
                SupervisorAction.GENERATE_OUTLINE,
                "标题已确认，下一步生成文章大纲",
                review_round=review_round,
            )
        return _decision(
            SupervisorAction.GENERATE_CONTENT,
            "大纲已准备好，下一步生成正文",
            review_round=review_round,
        )
    if phase is ArticlePhase.CONTENT_GENERATION:
        if not article.content:
            return _decision(
                SupervisorAction.GENERATE_CONTENT,
                "正文缺失，下一步生成正文",
                review_round=review_round,
            )
        if not article.image_prompts:
            return _decision(
                SupervisorAction.GENERATE_IMAGE_PROMPTS,
                "正文已生成，下一步生成配图提示词",
                review_round=review_round,
            )
        if not article.image_results:
            return _decision(
                SupervisorAction.GENERATE_IMAGE_RESULTS,
                "配图提示词已生成，下一步生成图片结果",
                review_round=review_round,
            )
        if not article.final_markdown:
            return _decision(
                SupervisorAction.COMPOSE_ARTICLE,
                "素材已准备好，下一步合成图文稿",
                review_round=review_round,
            )
    if phase is ArticlePhase.ARTICLE_COMPOSITION:
        if not article.final_markdown:
            return _decision(
                SupervisorAction.COMPOSE_ARTICLE,
                "最终图文稿缺失，下一步重新合成",
                review_round=review_round,
            )
        if review_result is None:
            return _decision(
                SupervisorAction.REVIEW_ARTICLE,
                "最终图文稿已生成，下一步质量评审",
                review_round=review_round,
            )
        if review_result.passed and not review_result.blocking_issues:
            return _decision(
                SupervisorAction.COMPLETE,
                "质量评审通过，文章可以交付",
                review_round=review_round,
            )
        if review_round >= max_auto_review_rounds:
            return _decision(
                SupervisorAction.MANUAL_REVIEW,
                "已达到自动修订轮次上限",
                review_round=review_round,
            )
        try:
            action = SupervisorAction(review_result.revision_action or SupervisorAction.REGENERATE_CONTENT.value)
        except ValueError:
            # Review content is model output.  A vague value such as "minor"
            # must still produce the one permitted repair attempt rather than a
            # successful-looking no-op in the page job.
            action = SupervisorAction.REGENERATE_CONTENT
        if action not in {
            SupervisorAction.REGENERATE_CONTENT,
            SupervisorAction.REGENERATE_IMAGE_PROMPTS,
            SupervisorAction.REGENERATE_IMAGE_RESULTS,
        }:
            action = SupervisorAction.REGENERATE_CONTENT
        return _decision(
            action,
            review_result.revision_reason or "质量评审未通过",
            action is not SupervisorAction.MANUAL_REVIEW,
            review_round,
            action is not SupervisorAction.MANUAL_REVIEW,
        )
    if phase is ArticlePhase.COMPLETED:
        return _decision(
            SupervisorAction.COMPLETE,
            "文章已完成",
            review_round=review_round,
        )
    return _decision(
        SupervisorAction.MANUAL_REVIEW,
        f"当前阶段暂不支持自动推进：{phase}",
        review_round=review_round,
    )

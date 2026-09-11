from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_article_studio.main import resolve_page_navigation


def article(**overrides: object) -> SimpleNamespace:
    """Small persisted-article stand-in for the page-only navigation rules."""

    fields = {
        "selected_title": None,
        "outline": None,
        "content": None,
        "image_prompts": None,
        "image_results": None,
        "final_markdown": None,
        "phase": "CREATED",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("state", "next_action", "expected_default"),
    [
        (article(), "GENERATE_TITLES", "title"),
        (article(selected_title="标题"), "GENERATE_OUTLINE", "title"),
        (article(selected_title="标题", outline="[]"), "GENERATE_CONTENT", "content"),
        (
            article(selected_title="标题", outline="[]", content="# 正文"),
            "GENERATE_IMAGE_PROMPTS",
            "images",
        ),
        (
            article(selected_title="标题", outline="[]", content="# 正文", image_results="[]"),
            "COMPOSE_ARTICLE",
            "images",
        ),
        (
            article(selected_title="标题", outline="[]", content="# 正文", image_results="[]", final_markdown="# 成稿"),
            "REVIEW_ARTICLE",
            "delivery",
        ),
    ],
)
def test_navigation_derives_the_workflow_default_from_persisted_artifacts(
    state: SimpleNamespace,
    next_action: str,
    expected_default: str,
) -> None:
    navigation = resolve_page_navigation(state, next_action, requested_step=None)

    assert navigation["default_step"] == expected_default
    assert navigation["current_step"] == expected_default
    assert navigation["current_position"] == ["title", "outline", "content", "images", "delivery"].index(expected_default)


def test_navigation_keeps_completed_steps_reachable_but_falls_back_from_future_or_invalid_steps() -> None:
    state = article(selected_title="标题", outline="[]", content="# 正文")

    previous = resolve_page_navigation(state, "GENERATE_IMAGE_PROMPTS", requested_step="outline")
    future = resolve_page_navigation(state, "GENERATE_IMAGE_PROMPTS", requested_step="delivery")
    invalid = resolve_page_navigation(state, "GENERATE_IMAGE_PROMPTS", requested_step="unknown")

    assert previous["current_step"] == "outline"
    assert future["current_step"] == "images"
    assert invalid["current_step"] == "images"
    assert [step["key"] for step in future["workflow_steps"]] == ["title", "outline", "content", "images", "delivery"]
    assert [step["complete"] for step in future["workflow_steps"]] == [True, True, True, False, False]
    assert [step["reachable"] for step in future["workflow_steps"]] == [True, True, True, True, False]
    assert [step["locked"] for step in future["workflow_steps"]] == [False, False, False, False, True]


def test_navigation_uses_the_repair_action_over_stale_later_artifacts() -> None:
    state = article(
        selected_title="标题",
        outline="[]",
        content="# 正文",
        image_prompts="[]",
        image_results="[]",
        final_markdown="# 成稿",
    )

    navigation = resolve_page_navigation(state, "REGENERATE_CONTENT", requested_step="delivery")

    assert navigation["default_step"] == "content"
    assert navigation["current_step"] == "content"
    assert navigation["workflow_steps"][3]["locked"] is True


def test_generated_outline_stays_on_the_outline_card_until_the_user_saves_it() -> None:
    state = article(selected_title="标题", outline="[]", phase="OUTLINE_EDITING", execution_mode="STEP")

    navigation = resolve_page_navigation(state, "GENERATE_CONTENT", requested_step=None)

    assert navigation["default_step"] == "outline"
    assert navigation["current_step"] == "outline"


def test_one_click_body_generation_can_advance_past_the_outline_confirmation_card() -> None:
    state = article(
        selected_title="标题",
        outline="[]",
        phase="OUTLINE_EDITING",
        execution_mode="ONE_CLICK",
        status="RUNNING",
    )

    navigation = resolve_page_navigation(state, "GENERATE_CONTENT", requested_step="content")

    assert navigation["default_step"] == "content"
    assert navigation["current_step"] == "content"


def test_completed_composition_opens_the_delivery_card_instead_of_returning_to_images() -> None:
    state = article(
        selected_title="标题",
        outline="[]",
        content="# 正文",
        image_prompts="[]",
        image_results="[]",
        final_markdown="# 图文稿",
        phase="ARTICLE_COMPOSITION",
    )

    navigation = resolve_page_navigation(state, "COMPOSE_ARTICLE", requested_step=None)

    assert navigation["default_step"] == "delivery"
    assert navigation["current_step"] == "delivery"

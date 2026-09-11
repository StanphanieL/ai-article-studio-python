"""Behavioral tests for workflow policy without a database or providers."""

import json

import pytest

from ai_article_studio.domain import WorkflowValidationError
from ai_article_studio.schemas import ImagePlacement, ImagePromptOption, ImageVisualRole


def plan(**values):
    return ImagePromptOption(
        image_title="配图", usage_scene="正文", prompt_zh="图", prompt_en="image", **values
    )


def test_generated_plan_repairs_anchor_and_role_without_mutating_input():
    from ai_article_studio._workflow_image_plan import normalize_generated_image_plan, validate_image_plan_contract

    content = "# 标题\n\n导语。\n\n## 验证\n\n第一段。\n\n第二段。"
    plans = [
        plan(visual_role=ImageVisualRole.COVER, placement=ImagePlacement.COVER),
        plan(visual_role=ImageVisualRole.CONCEPT, placement=ImagePlacement.AFTER_INTRO),
        plan(
            visual_role=ImageVisualRole.CONCEPT,
            placement=ImagePlacement.AFTER_PARAGRAPH,
            anchor_heading="验证",
            anchor_paragraph=99,
        ),
    ]
    originals = [item.model_dump() for item in plans]

    normalized, adjusted = normalize_generated_image_plan(content, plans)

    assert [item.model_dump() for item in plans] == originals
    assert adjusted == 3
    assert normalized[2].anchor_paragraph == 2
    assert normalized[2].anchor_excerpt == "第二段。"
    assert normalized[2].visual_role is ImageVisualRole.DIAGRAM
    validate_image_plan_contract(content, normalized)
    assert normalize_generated_image_plan(content, normalized) == (normalized, 0)


@pytest.mark.parametrize(
    ("placement", "heading", "paragraph", "message"),
    [
        (ImagePlacement.COVER, "验证", 0, "不能指定章节"),
        (ImagePlacement.AFTER_PARAGRAPH, "验证", 2, "锚点段落无效"),
        (ImagePlacement.SECTION_END, "不存在", 0, "未找到锚点标题"),
        (ImagePlacement.SECTION_END, "验证", 1, "段落序号必须为 0"),
    ],
)
def test_editor_anchor_validation_is_strict(placement, heading, paragraph, message):
    from ai_article_studio._workflow_image_plan import validate_image_plan_anchor

    with pytest.raises(WorkflowValidationError, match=message):
        validate_image_plan_anchor(
            "## 验证\n\n唯一一段。",
            plan(placement=placement, anchor_heading=heading, anchor_paragraph=paragraph),
        )


def test_generated_normalization_does_not_invent_a_missing_cover():
    from ai_article_studio._workflow_image_plan import normalize_generated_image_plan, validate_image_plan_contract

    content = "## 验证\n\n正文。"
    normalized, _ = normalize_generated_image_plan(content, [plan(placement=ImagePlacement.AFTER_INTRO)])

    with pytest.raises(WorkflowValidationError, match="只能包含一张封面图"):
        validate_image_plan_contract(content, normalized)


@pytest.mark.parametrize("raw", [None, "broken json", "{}", '[null, 1, "text"]'])
def test_asset_path_parser_ignores_malformed_legacy_records(tmp_path, raw):
    from ai_article_studio._workflow_persistence import local_image_paths

    assert local_image_paths(raw, tmp_path) == set()


def test_asset_path_parser_only_returns_contained_local_assets(tmp_path):
    from ai_article_studio._workflow_persistence import local_image_paths

    raw = json.dumps([
        {"imageUrl": "/uploads/images/task/image.svg"},
        {"imageUrl": "/uploads/images/../outside.svg"},
        {"imageUrl": "https://supplier.test/image.png"},
        {"imageUrl": "/uploads/images/"},
        {"imageUrl": 3},
    ])

    assert local_image_paths(raw, tmp_path) == {tmp_path / "task/image.svg"}

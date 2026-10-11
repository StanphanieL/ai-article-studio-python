"""Generated image plans should retain meaning while using real insertion points."""

from ai_article_studio._workflow_image_plan import normalize_generated_image_plan, validate_image_plan_contract
from ai_article_studio.schemas import ImagePlacement, ImagePromptOption, ImageVisualRole


def _image(title: str, role: ImageVisualRole, placement: ImagePlacement, heading: str = "", paragraph: int = 0) -> ImagePromptOption:
    return ImagePromptOption(
        image_title=title,
        usage_scene=f"解释{title}",
        prompt_zh=f"绘制{title}",
        prompt_en=f"Illustrate {title}",
        visual_role=role,
        placement=placement,
        anchor_heading=heading,
        anchor_paragraph=paragraph,
    )


def _cover() -> ImagePromptOption:
    return _image("封面", ImageVisualRole.COVER, ImagePlacement.COVER)


def test_duplicate_non_cover_roles_remain_semantically_unchanged() -> None:
    content = "## 章节\n\n第一段。\n\n第二段。"
    plans = [
        _cover(),
        _image("结构图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "章节", 1),
        _image("流程图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "章节", 2),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert [plan.visual_role for plan in normalized] == [ImageVisualRole.COVER, ImageVisualRole.DIAGRAM, ImageVisualRole.DIAGRAM]
    assert [plan.prompt_zh for plan in normalized] == ["绘制封面", "绘制结构图", "绘制流程图"]
    validate_image_plan_contract(content, normalized)


def test_clamped_collision_moves_only_the_later_plan_within_its_section() -> None:
    content = "## 目标章节\n\n第一段。\n\n第二段。\n\n## 无关章节\n\n其他段落。"
    plans = [
        _cover(),
        _image("原有图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "目标章节", 2),
        _image("溢出图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "目标章节", 99),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert [(plan.placement, plan.anchor_heading, plan.anchor_paragraph) for plan in normalized[1:]] == [
        (ImagePlacement.AFTER_PARAGRAPH, "目标章节", 2),
        (ImagePlacement.AFTER_PARAGRAPH, "目标章节", 1),
    ]
    assert normalized[2].anchor_excerpt == "第一段。"
    assert normalized[2].prompt_zh == "绘制溢出图"
    validate_image_plan_contract(content, normalized)


def test_section_end_and_last_paragraph_are_one_visual_insertion_point() -> None:
    content = "## 章节\n\n第一段。\n\n第二段。"
    plans = [
        _cover(),
        _image("段尾图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "章节", 2),
        _image("章尾图", ImageVisualRole.DIAGRAM, ImagePlacement.SECTION_END, "章节"),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert normalized[2].placement is ImagePlacement.AFTER_PARAGRAPH
    assert normalized[2].anchor_heading == "章节"
    assert normalized[2].anchor_paragraph == 1
    validate_image_plan_contract(content, normalized)


def test_excess_generated_plans_are_reduced_without_moving_to_unrelated_sections() -> None:
    content = "## 目标章节\n\n唯一段落。\n\n## 无关章节\n\n另一段落。"
    plans = [
        _cover(),
        _image("保留图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_PARAGRAPH, "目标章节", 1),
        _image("重复图", ImageVisualRole.DIAGRAM, ImagePlacement.SECTION_END, "目标章节"),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert [plan.image_title for plan in normalized] == ["封面", "保留图"]
    validate_image_plan_contract(content, normalized)


def test_no_h2_article_keeps_cover_and_one_intro_image() -> None:
    content = "# 标题\n\n一段很短的导语。"
    plans = [
        _cover(),
        _image("导语图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_INTRO),
        _image("过量图", ImageVisualRole.DIAGRAM, ImagePlacement.AFTER_INTRO),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert [plan.image_title for plan in normalized] == ["封面", "导语图"]
    validate_image_plan_contract(content, normalized)


def test_duplicate_intro_images_do_not_move_into_unrelated_sections() -> None:
    content = "# 标题\n\n导语。\n\n## 正文\n\n正文段落。"
    plans = [
        _cover(),
        _image("导语图", ImageVisualRole.SCENE, ImagePlacement.AFTER_INTRO),
        _image("第二张导语图", ImageVisualRole.SCENE, ImagePlacement.AFTER_INTRO),
    ]

    normalized, _ = normalize_generated_image_plan(content, plans)

    assert [plan.image_title for plan in normalized] == ["封面", "导语图"]
    validate_image_plan_contract(content, normalized)


def test_legacy_role_replacement_hook_does_not_change_meaning():
    from ai_article_studio.schemas import ImagePromptOption, ImageVisualRole
    from ai_article_studio.workflow import ArticleWorkflow

    prompt = ImagePromptOption(image_title="图解", usage_scene="说明流程", prompt_zh="流程", prompt_en="workflow", visual_role=ImageVisualRole.DIAGRAM)
    assert ArticleWorkflow._replacement_visual_role(prompt, {ImageVisualRole.DIAGRAM}) is None

from ai_article_studio.anchors import image_anchor_excerpt, image_visual_brief_catalog, markdown_anchor_sections
from ai_article_studio.composition import compose_article
from ai_article_studio.schemas import ImagePlacement, ImageResultOption


def test_anchor_catalog_ignores_h2_text_inside_a_fenced_code_block() -> None:
    content = "# 标题\n\n## 正文章节\n\n第一段。\n\n```md\n## 代码里的标题\n```\n\n第二段。"

    sections = markdown_anchor_sections(content)

    assert [(section.heading, len(section.paragraphs)) for section in sections] == [("正文章节", 3)]


def test_composer_does_not_treat_a_code_heading_as_a_real_image_section() -> None:
    content = "# 标题\n\n## 正文章节\n\n第一段。\n\n```md\n## 代码里的标题\n```\n\n第二段。"
    image = ImageResultOption(
        image_title="配图",
        usage_scene="正文配图",
        prompt_zh="图",
        prompt_en="image",
        placement=ImagePlacement.SECTION_END,
        anchor_heading="正文章节",
        image_url="/uploads/images/example.svg",
        source="MOCK",
    )

    composed = compose_article("标题", content, [image])

    assert composed.index("```md") < composed.index("![配图]")
    assert composed.count("## 代码里的标题") == 1


def test_page_anchor_options_expose_only_real_headings_and_real_paragraph_numbers() -> None:
    from ai_article_studio.main import _image_anchor_options

    headings, paragraphs = _image_anchor_options("## 第一节\n\n一。\n\n二。\n\n```md\n## 伪标题\n```")

    assert headings == ["第一节"]
    assert paragraphs == {"第一节": [1, 2, 3]}


def test_visual_brief_catalog_keeps_anchor_text_and_excludes_fenced_headings() -> None:
    content = "## 可重复性\n\n同一流程重复运行后出现偏离。\n\n```md\n## 代码里的标题\n```\n\n需要记录偏离发生的位置。"

    catalog = image_visual_brief_catalog(content)

    assert '"可重复性"' in catalog
    assert "第 1 段：同一流程重复运行后出现偏离。" in catalog
    assert "第 3 段：需要记录偏离发生的位置。" in catalog
    assert "代码里的标题" not in catalog


def test_image_anchor_excerpt_uses_the_exact_canonical_paragraph_or_section_end() -> None:
    content = "# 标题\n\n引言说明文章关注的行为变化。\n\n## 行为锚点\n\n先记录触发条件。\n\n重复执行同一流程后，记录偏离发生的位置与原因。"

    paragraph_excerpt = image_anchor_excerpt(content, "AFTER_PARAGRAPH", "行为锚点", 2)
    section_end_excerpt = image_anchor_excerpt(content, "SECTION_END", "行为锚点", 0)
    cover_excerpt = image_anchor_excerpt(content, "COVER")

    assert paragraph_excerpt == "重复执行同一流程后，记录偏离发生的位置与原因。"
    assert section_end_excerpt == paragraph_excerpt
    assert cover_excerpt == "引言说明文章关注的行为变化。"

from __future__ import annotations

from ai_article_studio.schemas import ImageResultOption


def image(title: str, scene: str, url: str, **plan: object) -> ImageResultOption:
    values = {
        "image_title": title,
        "usage_scene": scene,
        "prompt_zh": "中文提示词",
        "prompt_en": "English prompt",
        "image_url": url,
        "source": "MOCK",
        "author": "AI Article Studio",
        "source_url": "https://example.test/source",
        "author_url": "https://example.test/author",
    }
    values.update(plan)
    return ImageResultOption(**values)


def test_compose_places_cover_before_introduction_and_images_after_sections() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        "可控工作流",
        "导语\n\n## 第一节\n\n第一节正文\n\n## 第二节\n\n第二节正文",
        [
            image("封面", "文章封面", "/uploads/images/cover.svg"),
            image("图一", "第一节正文配图", "/uploads/images/one.svg"),
            image("图二", "第二节正文配图", "/uploads/images/two.svg"),
        ],
    )

    assert markdown.index("# 可控工作流") < markdown.index("cover.svg") < markdown.index("导语")
    assert markdown.index("第一节正文") < markdown.index("one.svg") < markdown.index("## 第二节")
    assert markdown.index("第二节正文") < markdown.index("two.svg")
    expected_caption = "> 来源：[示例图片](https://example.test/source) ｜ 作者：[AI Article Studio](https://example.test/author)"
    assert expected_caption in markdown
    assert "第一节正文配图" not in markdown
    assert "图片来源：https://" not in markdown
    assert "作者主页：https://" not in markdown


def test_compose_places_semantic_images_at_their_requested_anchors() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        "可控工作流",
        "导语第一段。\n\n导语第二段。\n\n## 第一节\n\n第一节第一段。\n\n第一节第二段。\n\n## 第二节\n\n第二节正文。",
        [
            image("封面", "文章封面", "/uploads/images/cover.svg", visual_role="COVER", placement="COVER", layout="HERO"),
            image(
                "导语图",
                "导语解释图",
                "/uploads/images/intro.svg",
                visual_role="CONCEPT",
                placement="AFTER_INTRO",
                layout="WIDE",
            ),
            image(
                "第一节图",
                "第一节第二段后的图解",
                "/uploads/images/first.svg",
                visual_role="DIAGRAM",
                placement="AFTER_PARAGRAPH",
                anchor_heading="第一节",
                anchor_paragraph=2,
                layout="WIDE",
            ),
            image(
                "第二节图",
                "第二节收束图",
                "/uploads/images/second.svg",
                visual_role="CLOSING",
                placement="SECTION_END",
                anchor_heading="第二节",
                layout="COMPACT",
            ),
        ],
    )

    assert markdown.index("cover.svg") < markdown.index("导语第一段")
    assert markdown.index("导语第二段") < markdown.index("intro.svg") < markdown.index("## 第一节")
    assert markdown.index("第一节第二段") < markdown.index("first.svg") < markdown.index("## 第二节")
    assert markdown.index("第二节正文") < markdown.index("second.svg")


def test_compose_falls_back_to_section_end_without_duplicate_or_adjacent_images() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        "配图回退",
        "导语。\n\n## 第一节\n\n第一节正文。\n\n## 第二节\n\n第二节正文。",
        [
            image(
                "无效锚点",
                "第一节解释图",
                "/uploads/images/fallback.svg",
                placement="AFTER_PARAGRAPH",
                anchor_heading="不存在的小节",
                anchor_paragraph=9,
            ),
            image(
                "重复图片",
                "重复资源",
                "/uploads/images/fallback.svg",
                placement="SECTION_END",
                anchor_heading="第二节",
            ),
            image(
                "第二节图",
                "第二节收束图",
                "/uploads/images/second.svg",
                placement="SECTION_END",
                anchor_heading="第二节",
            ),
        ],
    )

    assert markdown.count("fallback.svg") == 1
    assert markdown.index("第一节正文") < markdown.index("fallback.svg") < markdown.index("## 第二节")
    assert markdown.index("第二节正文") < markdown.index("second.svg")
    assert ")]\n\n> 重复资源" not in markdown


def test_compose_preserves_every_unique_image_when_one_section_anchor_overflows() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        "单节配图",
        "## 唯一章节\n\n唯一段落。",
        [
            image("图一", "第一张", "/uploads/images/one.svg", placement="AFTER_PARAGRAPH", anchor_heading="唯一章节", anchor_paragraph=1),
            image("图二", "第二张", "/uploads/images/two.svg", placement="AFTER_PARAGRAPH", anchor_heading="唯一章节", anchor_paragraph=1),
            image("图三", "第三张", "/uploads/images/three.svg", placement="AFTER_PARAGRAPH", anchor_heading="唯一章节", anchor_paragraph=1),
        ],
    )

    assert markdown.index("唯一段落") < markdown.index("one.svg") < markdown.index("two.svg") < markdown.index("three.svg")
    assert all(markdown.count(image_url) == 1 for image_url in ("one.svg", "two.svg", "three.svg"))


def test_compose_keeps_later_section_anchor_collisions_after_that_section() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        "章节顺序",
        "## 第一节\n\n第一节正文。\n\n## 第二节\n\n第二节正文。",
        [
            image("后一图", "第二节图一", "/uploads/images/later-one.svg", placement="SECTION_END", anchor_heading="第二节"),
            image("后二图", "第二节图二", "/uploads/images/later-two.svg", placement="SECTION_END", anchor_heading="第二节"),
        ],
    )

    assert markdown.index("## 第二节") < markdown.index("第二节正文") < markdown.index("later-one.svg") < markdown.index("later-two.svg")
    assert markdown.index("later-one.svg") > markdown.index("第一节正文")


def test_compose_escapes_hostile_image_metadata_before_emitting_markdown() -> None:
    from ai_article_studio.composition import compose_article

    markdown = compose_article(
        '<img class="title-owned" src="x">',
        "正文。",
        [
            image(
                '<img class="caption-owned" src="x">',
                '</blockquote><script class="scene-owned">boom</script>',
                "/uploads/images/safe.svg",
                source='<span class="source-owned">来源</span>',
                source_url='https://example.test/")<script class="url-owned">',
                author='<img class="author-owned" src="x">',
                author_url="javascript:alert(1)",
                placement="AFTER_INTRO",
            )
        ],
    )

    assert "<" not in markdown
    assert "class=\"" not in markdown
    assert "javascript:" not in markdown
    assert "&lt;" in markdown


def test_reader_facing_markdown_hides_provider_codes_in_legacy_attribution() -> None:
    from ai_article_studio.composition import reader_facing_markdown

    markdown = (
        "> 示例说明 ｜ 来源：FALLBACK ｜ 作者：AI Article Studio placeholder\n\n"
        "> 另一条说明 ｜ 来源：PEXELS ｜ 原图：[查看原图](https://example.test/source)"
    )

    assert reader_facing_markdown(markdown) == (
        "> 来源：示例图片 ｜ 作者：AI Article Studio\n\n"
        "> 来源：[Pexels 图库](https://example.test/source)"
    )

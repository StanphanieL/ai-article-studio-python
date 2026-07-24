from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZipFile

import httpx
import pytest

from ai_article_studio.schemas import ImageResultOption


def image(
    url: str,
    *,
    source: str = "MOCK",
    source_url: str = "",
    author: str = "",
    author_url: str = "",
    **plan: object,
) -> ImageResultOption:
    return ImageResultOption(
        image_title="封面",
        usage_scene="文章封面",
        prompt_zh="中文提示词",
        prompt_en="English prompt",
        image_url=url,
        source=source,
        source_url=source_url,
        author=author,
        author_url=author_url,
        **plan,
    )


def test_html_and_zip_exports_embed_and_package_local_images(tmp_path) -> None:
    from ai_article_studio.exports import ArticleExporter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cover.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>',
        encoding="utf-8",
    )
    url = "/uploads/images/cover.svg"
    markdown = "# 测试文章\n\n![封面](/uploads/images/cover.svg)"
    exporter = ArticleExporter(image_dir)

    html_file = exporter.export_html("测试文章", markdown, [image(url)])
    zip_file = exporter.export_zip("测试文章", markdown, [image(url), image(url)])

    assert b"data:image/svg+xml;base64," in html_file.content
    with ZipFile(BytesIO(zip_file.content)) as archive:
        assert set(archive.namelist()) == {
            "article.md",
            "article.html",
            "metadata.json",
            "images/image-01.svg",
        }


def test_zip_metadata_keeps_full_attribution_when_duplicate_files_are_deduplicated(tmp_path) -> None:
    from ai_article_studio.exports import ArticleExporter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "cover.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>',
        encoding="utf-8",
    )
    url = "/uploads/images/cover.svg"
    images = [
        image(
            url,
            source="PEXELS",
            source_url="https://www.pexels.com/photo/one/",
            author="Ada",
            author_url="https://www.pexels.com/@ada/",
        ),
        image(
            url,
            source="PEXELS",
            source_url="https://www.pexels.com/photo/two/",
            author="Lin",
            author_url="https://www.pexels.com/@lin/",
        ),
    ]

    zip_file = ArticleExporter(image_dir).export_zip("测试文章", f"# 测试文章\n\n![封面]({url})", images)

    with ZipFile(BytesIO(zip_file.content)) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert [name for name in archive.namelist() if name.startswith("images/")] == ["images/image-01.svg"]
        assert metadata == {
            "title": "测试文章",
            "imageCount": 2,
            "imageFileCount": 1,
            "images": [
                {
                    "imageUrl": url,
                    "source": "PEXELS",
                    "sourceUrl": "https://www.pexels.com/photo/one/",
                    "author": "Ada",
                    "authorUrl": "https://www.pexels.com/@ada/",
                    "exportPath": "images/image-01.svg",
                },
                {
                    "imageUrl": url,
                    "source": "PEXELS",
                    "sourceUrl": "https://www.pexels.com/photo/two/",
                    "author": "Lin",
                    "authorUrl": "https://www.pexels.com/@lin/",
                    "exportPath": "images/image-01.svg",
                },
            ],
        }


def test_export_rejects_untrusted_remote_image_urls(tmp_path) -> None:
    from ai_article_studio.exports import ArticleExporter, UnsafeImageUrlError

    exporter = ArticleExporter(tmp_path)

    try:
        exporter.load_image("https://example.invalid/secret.png")
    except UnsafeImageUrlError:
        pass
    else:
        raise AssertionError("untrusted remote image should be rejected")


def test_export_rejects_a_redirect_before_requesting_an_untrusted_host(tmp_path) -> None:
    from ai_article_studio.exports import ArticleExporter, UnsafeImageUrlError

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "cdn.pexels.com":
            return httpx.Response(
                302,
                headers={"location": "http://169.254.169.254/latest/meta-data"},
                request=request,
            )
        raise AssertionError("untrusted host must never be requested")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exporter = ArticleExporter(tmp_path, client=client)

    with pytest.raises(UnsafeImageUrlError):
        exporter.load_image("https://cdn.pexels.com/photo.jpg")

    assert requested == ["https://cdn.pexels.com/photo.jpg"]


def test_markdown_renderer_discards_bare_html_but_keeps_markdown_tables_and_html_code_examples() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown(
        """# 标题

第一句<br/>第二句。

<hr/>

<table>
  <tr><td>模型误输出的 HTML 表格</td></tr>
</table>

## 合法表格

| 技术 | 用途 |
| --- | --- |
| Python | 工作流 |

```html
<div class="demo">这是代码示例</div>
```
"""
    )

    assert "&lt;br" not in rendered
    assert "&lt;hr" not in rendered
    assert "模型误输出的 HTML 表格" not in rendered
    assert "<table>" in rendered
    assert "<th>技术</th>" in rendered
    assert "<td>Python</td>" in rendered
    assert "&lt;div class=&quot;demo&quot;&gt;这是代码示例&lt;/div&gt;" in rendered


def test_markdown_renderer_preserves_standard_autolinks_while_removing_html_tags() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown("访问 <https://example.com/docs>，不要使用 <span>裸 HTML</span>。")

    assert '<a href="https://example.com/docs">https://example.com/docs</a>' in rendered
    assert "<span>" not in rendered
    assert "裸 HTML" in rendered


def test_markdown_renderer_preserves_inline_jsx_code_while_removing_real_html() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown("使用 `<div className=\"grid\"><motion.div /></div>`。\n\n<span>真实裸 HTML</span>")

    assert '<code>&lt;div className="grid"&gt;&lt;motion.div /&gt;&lt;/div&gt;</code>' in rendered
    assert "<span>" not in rendered
    assert "真实裸 HTML" in rendered


def test_markdown_renderer_supports_fenced_code_nested_in_a_list() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown(
        """# 标题

## 示例

1. **触控反馈**
   ```tsx
   <motion.button whileTap={{ scale: 0.98 }} />
   ```
2. **下一步**
"""
    )

    assert '<ol>' in rendered
    assert '<pre><code>' in rendered
    assert '&lt;motion.button whileTap={{ scale: 0.98 }} /&gt;' in rendered
    assert '<li><strong>下一步</strong></li>' in rendered


def test_markdown_renderer_does_not_discard_following_markdown_after_a_self_closing_html_tag() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown("# 标题\n\n<div/>\n\n## 仍应显示\n\n这里是后续正文。")

    assert "<div" not in rendered
    assert "<h2>仍应显示</h2>" in rendered
    assert "这里是后续正文。" in rendered


def test_markdown_renderer_uses_only_controlled_figure_classes_for_image_metadata() -> None:
    from ai_article_studio.exports import render_markdown

    portable_markdown = """# 图文稿

![宽图](/uploads/images/wide.svg)

![小图](/uploads/images/compact.svg)
"""
    markdown_text = portable_markdown + """

<figure class="layout-evil">模型输出的裸 HTML</figure>
"""
    rendered = render_markdown(
        markdown_text,
        images=[
            image("/uploads/images/wide.svg", visual_role="DIAGRAM", layout="WIDE"),
            image("/uploads/images/compact.svg", visual_role="CLOSING", layout="COMPACT"),
        ],
    )

    assert '<figure class="article-figure layout-wide role-diagram">' in rendered
    assert '<figure class="article-figure layout-compact role-closing">' in rendered
    assert 'src="/uploads/images/wide.svg"' in rendered
    assert "layout-evil" not in rendered
    assert "<figure" not in portable_markdown
    assert "layout-wide" not in portable_markdown


def test_exports_apply_layout_to_html_but_keep_markdown_portable(tmp_path) -> None:
    from ai_article_studio.exports import ArticleExporter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "wide.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>',
        encoding="utf-8",
    )
    url = "/uploads/images/wide.svg"
    source_markdown = "# 测试文章\n\n![封面](/uploads/images/wide.svg)"
    images = [image(url, visual_role="DIAGRAM", layout="WIDE")]
    exporter = ArticleExporter(image_dir)

    html_file = exporter.export_html("测试文章", source_markdown, images)
    zip_file = exporter.export_zip("测试文章", source_markdown, images)

    assert b'<figure class="article-figure layout-wide role-diagram">' in html_file.content
    assert b"layout-wide" not in exporter.export_markdown("测试文章", source_markdown).content
    with ZipFile(BytesIO(zip_file.content)) as archive:
        assert b'<figure class="article-figure layout-wide role-diagram">' in archive.read("article.html")
        assert b"layout-wide" not in archive.read("article.md")


def test_markdown_renderer_uses_first_metadata_for_duplicate_image_url() -> None:
    from ai_article_studio.exports import render_markdown

    rendered = render_markdown(
        "![重复图](/uploads/images/duplicate.svg)",
        images=[
            image("/uploads/images/duplicate.svg", visual_role="DIAGRAM", layout="WIDE"),
            image("/uploads/images/duplicate.svg", visual_role="CLOSING", layout="COMPACT"),
        ],
    )

    assert 'class="article-figure layout-wide role-diagram"' in rendered
    assert "layout-compact" not in rendered


def test_exports_do_not_reintroduce_hostile_image_metadata_into_html_or_zip(tmp_path) -> None:
    from ai_article_studio.composition import compose_article
    from ai_article_studio.exports import ArticleExporter

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "safe.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="2" height="2"/></svg>',
        encoding="utf-8",
    )
    hostile = image(
        "/uploads/images/safe.svg",
        source='<script class="source-owned">boom</script>',
        source_url='https://example.test/")<script class="url-owned">',
        author='<img class="author-owned" src="x">',
        author_url="javascript:alert(1)",
    ).model_copy(
        update={
            "image_title": '<img class="caption-owned" src="x">',
            "usage_scene": '</blockquote><script class="scene-owned">boom</script>',
        }
    )
    final_markdown = compose_article('<img class="title-owned" src="x">', "正文。", [hostile])
    exporter = ArticleExporter(image_dir)

    html_file = exporter.export_html("导出标题", final_markdown, [hostile])
    zip_file = exporter.export_zip("导出标题", final_markdown, [hostile])

    assert b"<script" not in html_file.content
    assert b'class="caption-owned"' not in html_file.content
    with ZipFile(BytesIO(zip_file.content)) as archive:
        assert b"<" not in archive.read("article.md")
        assert b"<script" not in archive.read("article.html")
        assert b'class="caption-owned"' not in archive.read("article.html")

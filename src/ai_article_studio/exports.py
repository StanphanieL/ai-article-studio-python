"""Safe, portable HTML and ZIP exports for finalized image-rich Markdown."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape, unescape
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import bleach
import httpx
import markdown

from .composition import reader_facing_markdown
from .markdown_syntax import mask_inline_code_spans, split_inline_code_spans
from .safe_fetch import UnsafeImageUrlError, fetch_trusted_image
from .schemas import ImageLayout, ImageResultOption, ImageVisualRole

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_RAW_HTML = re.compile(r"<!--.*?-->|</?[a-z][a-z0-9-]*(?:\s+[^<>]*?)?\s*/?>", re.IGNORECASE)
_RAW_BREAK = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
_RAW_RULE = re.compile(r"^\s*<hr\s*/?\s*>\s*$", re.IGNORECASE)
_RAW_BLOCK_TAGS = "|".join(
    (
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "center",
        "colgroup",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "dd",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "head",
        "header",
        "html",
        "iframe",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "script",
        "section",
        "style",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    )
)
_RAW_BLOCK_OPEN = re.compile(rf"<(?P<tag>{_RAW_BLOCK_TAGS})\b[^>]*>", re.IGNORECASE)
_IMAGE_PARAGRAPH = re.compile(
    r'<p>(?P<image><img\b[^>]*\bsrc="(?P<src>[^"]+)"[^>]*>)</p>',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExportFile:
    file_name: str
    content_type: str
    content: bytes


@dataclass(frozen=True)
class ImageBinary:
    content: bytes
    content_type: str
    extension: str


class ArticleExporter:
    def __init__(self, image_dir: Path, client: httpx.Client | None = None) -> None:
        self.image_dir = image_dir.resolve()
        self.client = client or httpx.Client(timeout=30, follow_redirects=True)

    def load_image(self, image_url: str) -> ImageBinary:
        if image_url.startswith("/uploads/images/"):
            relative_path = image_url.removeprefix("/uploads/images/")
            file_path = (self.image_dir / relative_path).resolve()
            if not file_path.is_relative_to(self.image_dir):
                raise UnsafeImageUrlError("非法本地图片路径")
            if not file_path.is_file():
                raise FileNotFoundError(f"本地图片不存在：{relative_path}")
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            return ImageBinary(file_path.read_bytes(), content_type, file_path.suffix or ".bin")
        downloaded = fetch_trusted_image(self.client, image_url)
        extension = mimetypes.guess_extension(downloaded.content_type) or ".png"
        return ImageBinary(downloaded.content, downloaded.content_type, extension)

    def export_html(self, title: str, final_markdown: str, images: list[ImageResultOption]) -> ExportFile:
        replacements = self._embedded_urls(images)
        reader_markdown = reader_facing_markdown(final_markdown)
        rendered = render_markdown(
            _replace_urls(reader_markdown, replacements),
            images=_replace_image_urls(images, replacements),
        )
        html = _document_html(title, rendered)
        return ExportFile(
            f"{_safe_file_name(title)}.html",
            "text/html; charset=utf-8",
            html.encode(),
        )

    def export_markdown(self, title: str, final_markdown: str) -> ExportFile:
        return ExportFile(
            f"{_safe_file_name(title)}.md",
            "text/markdown; charset=utf-8",
            reader_facing_markdown(final_markdown).encode(),
        )

    def export_zip(self, title: str, final_markdown: str, images: list[ImageResultOption]) -> ExportFile:
        prepared = self._prepared_images(images)
        replacements = {original: relative for original, relative, _ in prepared}
        exported_paths = {original: relative for original, relative, _ in prepared}
        reader_markdown = reader_facing_markdown(final_markdown)
        html = _document_html(
            title,
            render_markdown(
                _replace_urls(reader_markdown, replacements),
                images=_replace_image_urls(images, replacements),
            ),
        )
        payload = BytesIO()
        with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
            archive.writestr("article.md", _replace_urls(reader_markdown, replacements))
            archive.writestr("article.html", html)
            archive.writestr(
                "metadata.json",
                json.dumps(
                    {
                        "title": title,
                        "imageCount": len(images),
                        "imageFileCount": len(prepared),
                        "images": [
                            {
                                "imageUrl": image.image_url,
                                "source": image.source,
                                "sourceUrl": image.source_url,
                                "author": image.author,
                                "authorUrl": image.author_url,
                                "exportPath": exported_paths[image.image_url],
                            }
                            for image in images
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            for _, relative, binary in prepared:
                archive.writestr(relative, binary.content)
        return ExportFile(f"{_safe_file_name(title)}.zip", "application/zip", payload.getvalue())

    def _embedded_urls(self, images: list[ImageResultOption]) -> dict[str, str]:
        replacements: dict[str, str] = {}
        for image in images:
            if image.image_url not in replacements:
                binary = self.load_image(image.image_url)
                encoded = base64.b64encode(binary.content).decode("ascii")
                replacements[image.image_url] = f"data:{binary.content_type};base64,{encoded}"
        return replacements

    def _prepared_images(self, images: list[ImageResultOption]) -> list[tuple[str, str, ImageBinary]]:
        prepared: list[tuple[str, str, ImageBinary]] = []
        seen: set[str] = set()
        for image in images:
            if image.image_url in seen:
                continue
            seen.add(image.image_url)
            binary = self.load_image(image.image_url)
            relative = f"images/image-{len(prepared) + 1:02d}{binary.extension}"
            prepared.append((image.image_url, relative, binary))
        return prepared


def _replace_urls(markdown_text: str, replacements: dict[str, str]) -> str:
    result = markdown_text
    for original, replacement in replacements.items():
        result = result.replace(original, replacement)
    return result


def render_markdown(markdown_text: str, images: Sequence[ImageResultOption] = ()) -> str:
    """Render Markdown for exports and in-app previews with the same sanitization policy."""

    raw_html = markdown.markdown(
        _strip_bare_html(markdown_text),
        extensions=["extra", "sane_lists", "pymdownx.superfences"],
    )
    allowed_tags = set(bleach.sanitizer.ALLOWED_TAGS) | {
        "h1",
        "h2",
        "h3",
        "hr",
        "img",
        "p",
        "pre",
        "code",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
    }
    sanitized = bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes={"img": ["src", "alt"], "a": ["href", "title"]},
        protocols={"http", "https", "data"},
        strip=True,
    )
    return _decorate_image_figures(sanitized, images)


def _replace_image_urls(
    images: Sequence[ImageResultOption], replacements: dict[str, str]
) -> list[ImageResultOption]:
    """Keep trusted layout metadata aligned with an embedded/exported image URL."""

    return [
        image.model_copy(update={"image_url": replacements.get(image.image_url, image.image_url)})
        for image in images
    ]


def _decorate_image_figures(rendered_html: str, images: Sequence[ImageResultOption]) -> str:
    """Wrap only renderer-produced image paragraphs in fixed, validated classes.

    Article Markdown is sanitized before this step.  Classes are assembled from
    Pydantic enums rather than model output, so generated Markdown can never
    inject HTML or arbitrary CSS into the preview/export.
    """

    figure_classes: dict[str, str] = {}
    for image in images:
        if image.image_url:
            # Composition keeps the first result when a provider returns the
            # same asset twice, so preview/export must use the same metadata.
            figure_classes.setdefault(image.image_url, _figure_class(image))
    if not figure_classes:
        return rendered_html

    def replace(match: re.Match[str]) -> str:
        figure_class = figure_classes.get(unescape(match.group("src")))
        if figure_class is None:
            return match.group(0)
        return f'<figure class="{figure_class}">{match.group("image")}</figure>'

    return _IMAGE_PARAGRAPH.sub(replace, rendered_html)


def _figure_class(image: ImageResultOption) -> str:
    layout = image.layout if isinstance(image.layout, ImageLayout) else ImageLayout.STANDARD
    role = image.visual_role if isinstance(image.visual_role, ImageVisualRole) else ImageVisualRole.CONCEPT
    return f"article-figure layout-{layout.value.lower()} role-{role.value.lower()}"


def _strip_bare_html(markdown_text: str) -> str:
    """Remove model-emitted HTML before parsing Markdown, while preserving fenced examples.

    Generated articles are deliberately Markdown-only: the quality gate rejects raw
    HTML.  Rendering must honour that contract too; otherwise Bleach safely escapes
    unsupported tags but turns ``<br/>`` and ``<table>`` into distracting visible
    text.  Block HTML is discarded as an invalid generated fragment, whereas inline
    tags are stripped and their surrounding prose remains.
    """

    output: list[str] = []
    active_fence: str | None = None
    discarded_block: str | None = None
    normalized = markdown_text.replace("\r\n", "\n").replace("\r", "\n")

    for line in normalized.split("\n"):
        fence = _FENCE.match(line)
        if active_fence is not None:
            output.append(line)
            if fence and _closes_fence(fence.group(1), active_fence):
                active_fence = None
            continue
        if fence:
            active_fence = fence.group(1)
            output.append(line)
            continue

        if discarded_block is not None:
            closing = re.search(rf"</{re.escape(discarded_block)}\s*>", line, re.IGNORECASE)
            if closing:
                discarded_block = None
                output.extend(_strip_inline_html(line[closing.end() :]))
            continue

        masked_line = mask_inline_code_spans(line)
        opening = _RAW_BLOCK_OPEN.search(masked_line)
        if opening:
            tag = opening.group("tag")
            remaining = line[opening.end() :]
            closing = re.search(
                rf"</{re.escape(tag)}\s*>",
                mask_inline_code_spans(remaining),
                re.IGNORECASE,
            )
            output.extend(_strip_inline_html(line[: opening.start()]))
            if opening.group().rstrip().endswith("/>"):
                output.extend(_strip_inline_html(line[opening.end() :]))
            elif closing:
                closing_end = opening.end() + closing.end()
                output.extend(_strip_inline_html(line[closing_end:]))
            else:
                discarded_block = tag
            continue

        output.extend(_strip_inline_html(line))

    return "\n".join(output)


def _strip_inline_html(line: str) -> list[str]:
    if _RAW_RULE.fullmatch(line):
        return ["---"]
    stripped: list[str] = []
    for is_code, part in split_inline_code_spans(line):
        if is_code:
            stripped.append(part)
            continue
        without_breaks = _RAW_BREAK.sub("  ", part)
        stripped.append(_RAW_HTML.sub("", without_breaks))
    return ["".join(stripped)]


def _closes_fence(marker: str, opening_marker: str) -> bool:
    return marker[0] == opening_marker[0] and len(marker) >= len(opening_marker)


def _document_html(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        "<style>body{max-width:860px;margin:40px auto;font:17px/1.75 system-ui;}"
        "img{max-width:100%;border-radius:12px;}blockquote{color:#52606d;}"
        "figure.article-figure{margin:2rem auto;max-width:760px;}"
        "figure.article-figure img{display:block;width:100%;}"
        "figure.layout-hero,figure.layout-wide{max-width:100%;}"
        "figure.layout-standard{max-width:760px;}figure.layout-compact{max-width:480px;}"
        "@media(max-width:640px){body{margin:20px auto;padding:0 16px;}"
        "figure.article-figure,figure.layout-compact{max-width:100%;margin:1.4rem auto;}}</style>"
        f"</head><body>{body}</body></html>"
    )


def _safe_file_name(title: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", title or "ai-article").strip(" .")
    return (cleaned or "ai-article")[:60]

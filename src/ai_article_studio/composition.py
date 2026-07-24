"""Deterministic assembly of article Markdown and ordered image assets."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from urllib.parse import quote, urlsplit

from .anchors import mask_fenced_code_for_heading_scan
from .schemas import ImagePlacement, ImageResultOption, ImageVisualRole

SECTION_PATTERN = re.compile(r"(?m)^##(?!#)\s+.*$")

_SOURCE_LABELS = {
    "PEXELS": "Pexels 图库",
    "SILICONFLOW": "SiliconFlow 生成",
    "GOOGLE_AI": "Google AI 生成",
    "MOCK": "示例图片",
    "FALLBACK": "示例图片",
}
_READER_LEGACY_CAPTION = re.compile(r"(?m)^>\s*[^\n]*?\s*｜\s*(?=来源：)")
_READER_SOURCE = re.compile(
    r"来源：(?P<source>PEXELS|SILICONFLOW|GOOGLE_AI|MOCK|FALLBACK)"
    r"(?:[ \t]*｜[ \t]*原图：\[查看原图]\((?P<url>https?://[^)\s]+)\))?",
    re.MULTILINE,
)
_LEGACY_FALLBACK_AUTHOR = re.compile(r"(?P<prefix>作者：)(?:AI Article Studio placeholder|Picsum Placeholder)(?=[ \t]*(?:｜|$))", re.MULTILINE)


@dataclass(frozen=True)
class _Section:
    heading: str
    heading_key: str
    blocks: tuple[str, ...]


def compose_article(title: str, content: str, image_results: Sequence[ImageResultOption]) -> str:
    """Insert every distinct image once at its planned, safe Markdown anchor.

    Planning metadata is intentionally advisory: a stale heading or paragraph
    must not break delivery.  It falls back to a deterministic section end,
    while an occupied anchor uses another text boundary rather than emitting
    adjacent figures.
    """

    if not content.strip():
        raise ValueError("文章正文为空，无法进行图文合成")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    resolved_title, body = _resolve_title_and_body(title, normalized)
    introduction, sections = _parse_structure(body)
    images = _deduplicate_images(image_results)
    cover = _take_cover(images)
    if cover is not None:
        images = [image for image in images if image.image_url != cover.image_url]

    slots = _resolve_image_slots(images, introduction, sections)
    output: list[str] = [f"# {_safe_markdown_text(resolved_title)}"] if resolved_title else []
    if cover is not None:
        output.append(_image_markdown(cover))

    for index, block in enumerate(introduction, start=1):
        output.append(block)
        _append_slot_images(output, slots, _intro_paragraph_slot(index))
    _append_slot_images(output, slots, _intro_end_slot())

    for section_index, section in enumerate(sections):
        output.append(section.heading)
        for paragraph_index, block in enumerate(section.blocks, start=1):
            output.append(block)
            _append_slot_images(output, slots, _paragraph_slot(section_index, paragraph_index))
        _append_slot_images(output, slots, _section_end_slot(section_index))

    return "\n\n".join(part for part in output if part).strip()


def _resolve_title_and_body(title: str, content: str) -> tuple[str, str]:
    if title.strip():
        resolved_title = title.strip()
    elif content.startswith("# "):
        resolved_title = content.split("\n", 1)[0][2:].strip()
    else:
        resolved_title = ""
    if content.startswith("# "):
        return resolved_title, content.partition("\n")[2].strip()
    return resolved_title, content


def _parse_structure(body: str) -> tuple[tuple[str, ...], list[_Section]]:
    # The regex keeps its familiar, offset-based implementation, but headings
    # written inside a fenced code sample are never article sections.
    matches = list(SECTION_PATTERN.finditer(mask_fenced_code_for_heading_scan(body)))
    if not matches:
        return _split_blocks(body), []
    introduction = _split_blocks(body[: matches[0].start()])
    sections: list[_Section] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        heading = match.group().strip()
        sections.append(
            _Section(
                heading=heading,
                heading_key=_heading_key(heading),
                blocks=_split_blocks(body[match.end() : end]),
            )
        )
    return introduction, sections


def _split_blocks(markdown_text: str) -> tuple[str, ...]:
    return tuple(block.strip() for block in re.split(r"\n{2,}", markdown_text.strip()) if block.strip())


def _deduplicate_images(images: Sequence[ImageResultOption]) -> list[ImageResultOption]:
    result: list[ImageResultOption] = []
    seen_urls: set[str] = set()
    for image in images:
        if not image.image_url or image.image_url in seen_urls:
            continue
        seen_urls.add(image.image_url)
        result.append(image)
    return result


def _take_cover(images: Sequence[ImageResultOption]) -> ImageResultOption | None:
    for image in images:
        if (
            image.placement is ImagePlacement.COVER
            or image.visual_role is ImageVisualRole.COVER
            or "封面" in image.usage_scene
            or "cover" in image.usage_scene.casefold()
        ):
            return image
    return None


def _resolve_image_slots(
    images: Sequence[ImageResultOption], introduction: Sequence[str], sections: Sequence[_Section]
) -> dict[str, list[ImageResultOption]]:
    """Keep every unique image at its intended boundary in input order.

    A plan may legitimately contain more images than paragraph boundaries.  In
    that case they form an ordered overflow sequence at the requested boundary
    rather than disappearing or being moved to an earlier section.
    """

    placements: dict[str, list[ImageResultOption]] = {}
    fallback_cursor = 0

    for image in images:
        preferred, fallback_cursor = _preferred_slot(
            image,
            introduction,
            sections,
            fallback_cursor,
        )
        if preferred is None:
            continue
        placements.setdefault(preferred, []).append(image)
    return placements


def _append_slot_images(
    output: list[str], slots: dict[str, list[ImageResultOption]], slot: str
) -> None:
    output.extend(_image_markdown(image) for image in slots.get(slot, []))


def _preferred_slot(
    image: ImageResultOption,
    introduction: Sequence[str],
    sections: Sequence[_Section],
    fallback_cursor: int,
) -> tuple[str | None, int]:
    matching_section = _matching_section_index(image, sections)

    if image.placement is ImagePlacement.AFTER_INTRO:
        if introduction:
            return _intro_end_slot(), fallback_cursor
        return _fallback_section_slot(sections, fallback_cursor)

    if image.placement is ImagePlacement.AFTER_PARAGRAPH and matching_section is not None:
        section = sections[matching_section]
        if 1 <= image.anchor_paragraph <= len(section.blocks):
            return _paragraph_slot(matching_section, image.anchor_paragraph), fallback_cursor
        return _section_end_slot(matching_section), fallback_cursor

    if matching_section is not None:
        return _section_end_slot(matching_section), fallback_cursor
    return _fallback_section_slot(sections, fallback_cursor, introduction)


def _fallback_section_slot(
    sections: Sequence[_Section],
    fallback_cursor: int,
    introduction: Sequence[str] = (),
) -> tuple[str | None, int]:
    if sections:
        section_index = fallback_cursor % len(sections)
        return _section_end_slot(section_index), fallback_cursor + 1
    if introduction:
        return _intro_end_slot(), fallback_cursor
    return None, fallback_cursor


def _matching_section_index(image: ImageResultOption, sections: Sequence[_Section]) -> int | None:
    anchor_key = _heading_key(image.anchor_heading)
    if anchor_key:
        for index, section in enumerate(sections):
            if section.heading_key == anchor_key:
                return index

    legacy_context = f"{image.usage_scene} {image.image_title}".casefold()
    for index, section in enumerate(sections):
        if section.heading_key and section.heading_key in legacy_context:
            return index
    return None


def _heading_key(value: str) -> str:
    stripped = re.sub(r"^#{1,6}\s*", "", value).strip()
    return re.sub(r"\s+", " ", stripped).casefold()


def _intro_end_slot() -> str:
    return "intro:end"


def _intro_paragraph_slot(paragraph_index: int) -> str:
    return f"intro:paragraph:{paragraph_index}"


def _section_end_slot(section_index: int) -> str:
    return f"section:{section_index}:end"


def _paragraph_slot(section_index: int, paragraph_index: int) -> str:
    return f"section:{section_index}:paragraph:{paragraph_index}"


def _image_markdown(image: ImageResultOption) -> str:
    image_url = _safe_markdown_url(image.image_url, allow_local_path=True)
    if not image_url:
        return "> 图片资源地址无效，未插入图片。"
    title = _safe_markdown_text(image.image_title) or "文章配图"
    caption: list[str] = []
    source_label = _display_source(image.source)
    source_url = _safe_markdown_url(image.source_url)
    if image.source:
        caption.append(f"来源：[{source_label}]({source_url})" if source_url else f"来源：{source_label}")
    author_url = _safe_markdown_url(image.author_url)
    author = _display_author(image.source, image.author)
    if author and author_url:
        caption.append(f"作者：[{author}]({author_url})")
    elif author:
        caption.append(f"作者：{author}")
    elif author_url:
        caption.append(f"作者主页：[查看作者]({author_url})")
    return f"![{title}]({image_url})\n\n> {' ｜ '.join(part for part in caption if part)}"


def _display_source(source: str) -> str:
    """Keep attribution useful without leaking provider implementation labels."""

    normalized = (source or "").strip().upper()
    return _safe_markdown_text(_SOURCE_LABELS.get(normalized, source))


def _display_author(source: str, author: str) -> str:
    """Normalize legacy fallback attribution for reader-facing captions."""

    if (source or "").strip().upper() == "FALLBACK":
        return "AI Article Studio"
    return _safe_markdown_text(author)


def reader_facing_markdown(markdown: str) -> str:
    """Normalize legacy attribution labels before readers preview or export a draft.

    The persisted provider code remains unchanged for retry and export metadata,
    while a reader never has to see implementation terms such as ``FALLBACK``.
    """

    def replace_source(match: re.Match[str]) -> str:
        label = _display_source(match.group("source"))
        source_url = match.group("url")
        return f"来源：[{label}]({source_url})" if source_url else f"来源：{label}"

    normalized = _READER_LEGACY_CAPTION.sub("> ", markdown or "")
    normalized = _READER_SOURCE.sub(replace_source, normalized)
    return _LEGACY_FALLBACK_AUTHOR.sub(r"\g<prefix>AI Article Studio", normalized)


def _safe_markdown_text(value: str) -> str:
    """Normalize model metadata so it remains text, never Markdown/HTML syntax."""

    normalized = re.sub(r"\s+", " ", value or "").strip()
    escaped = escape(normalized, quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+!|])", r"\\\1", escaped)


def _safe_markdown_url(value: str, *, allow_local_path: bool = False) -> str:
    """Allow only ordinary HTTP(S) attribution URLs and our local image path."""

    candidate = re.sub(r"\s+", "", value or "")
    if not candidate or any(character in candidate for character in '<>"\\'):
        return ""
    if allow_local_path and candidate.startswith("/uploads/images/") and ".." not in candidate:
        return quote(candidate, safe="/%._-")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return quote(candidate, safe=":/?#[]@!$&'*+,;=%")

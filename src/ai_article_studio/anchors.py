"""One canonical Markdown anchor catalogue for article image placement.

The LLM, page editor, workflow validator, and composer must agree on what a
section and a paragraph are.  In particular, headings inside fenced code
blocks are content, not article sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SECTION_HEADING = re.compile(r"^##(?!#)\s+(.+?)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class MarkdownAnchorSection:
    heading: str
    paragraphs: tuple[str, ...]

    @property
    def key(self) -> str:
        return heading_key(self.heading)


def heading_key(value: str) -> str:
    """Return a forgiving comparison key without accepting Markdown syntax."""

    stripped = re.sub(r"^#{1,6}\s*", "", value).strip()
    return re.sub(r"\s+", " ", stripped).casefold()


def markdown_anchor_sections(content: str) -> list[MarkdownAnchorSection]:
    """Parse real H2 sections and paragraph blocks outside fenced code blocks."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    fence: str | None = None
    for line in normalized.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            if current is not None:
                # Preserve code verbatim as a normal paragraph block while
                # ensuring its internal ``##`` lines never become headings.
                current.append(line)
            continue
        if fence is not None:
            if current is not None:
                current.append(line)
            continue
        heading_match = _SECTION_HEADING.match(line)
        if heading_match:
            heading = heading_match.group(1).strip()
            if heading:
                current = []
                sections.append((heading, current))
            continue
        if current is not None:
            current.append(line)

    return [
        MarkdownAnchorSection(
            heading=heading,
            paragraphs=tuple(block.strip() for block in re.split(r"\n{2,}", "\n".join(lines).strip()) if block.strip()),
        )
        for heading, lines in sections
    ]


def mask_fenced_code_for_heading_scan(content: str) -> str:
    """Keep offsets while hiding fenced code from regular-expression heading scans."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    fence: str | None = None
    masked: list[str] = []
    for line in normalized.split("\n"):
        fence_match = _FENCE.match(line)
        in_fence = fence is not None
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            in_fence = True
        masked.append(" " * len(line) if in_fence else line)
    return "\n".join(masked)


def image_anchor_catalog(content: str) -> str:
    """Produce the small, authoritative catalogue sent to the image planner."""

    sections = markdown_anchor_sections(content)
    if not sections:
        return "（正文中没有可用的二级标题；除封面外请使用 AFTER_INTRO，anchorHeading 为空，anchorParagraph 为 0。）"
    rows = []
    for section in sections:
        positions = ", ".join(str(index) for index in range(1, len(section.paragraphs) + 1)) or "无正文段落"
        rows.append(f'- "{section.heading}"：可用段落 [{positions}]；章节末尾始终可用')
    return "\n".join(rows)


def image_visual_brief_catalog(content: str, *, excerpt_limit: int = 180) -> str:
    """Return the canonical image anchors together with the claim each anchor carries.

    The planner already receives the full article, but a compact per-paragraph
    brief makes the selected anchor an explicit visual grounding constraint.
    It deliberately reuses :func:`markdown_anchor_sections`, so fenced code,
    page controls, validation, and composition retain one definition of a
    paragraph.
    """

    sections = markdown_anchor_sections(content)
    if not sections:
        return "（正文中没有可用二级标题；除封面外请使用 AFTER_INTRO，anchorHeading 为空，anchorParagraph 为 0。）"

    rows: list[str] = []
    for section in sections:
        rows.append(f'- "{section.heading}"：章节末尾始终可用')
        if not section.paragraphs:
            rows.append("  - 没有可定位的正文段落；只能使用 SECTION_END。")
            continue
        for index, paragraph in enumerate(section.paragraphs, start=1):
            compact = _visual_excerpt(paragraph)
            if not compact:
                rows.append(f"  - 第 {index} 段：代码示例，不作为画面依据。")
                continue
            excerpt = compact[:excerpt_limit].rstrip()
            if len(compact) > excerpt_limit:
                excerpt += "…"
            rows.append(f"  - 第 {index} 段：{excerpt}")
    return "\n".join(rows)


def image_anchor_excerpt(
    content: str,
    placement: object,
    anchor_heading: str = "",
    anchor_paragraph: int = 0,
    *,
    excerpt_limit: int = 420,
) -> str:
    """Return the exact claim an image is meant to explain.

    The editor sends only a placement choice.  This helper resolves that choice
    against the canonical Markdown parser and never trusts a model- or
    browser-supplied text snippet.  It is deliberately compact because it is
    included in the downstream image-generation request.
    """

    placement_value = getattr(placement, "value", str(placement)).upper()
    sections = markdown_anchor_sections(content)
    chosen = next((section for section in sections if section.key == heading_key(anchor_heading)), None)

    excerpt = ""
    if placement_value == "AFTER_PARAGRAPH" and chosen and 1 <= anchor_paragraph <= len(chosen.paragraphs):
        excerpt = _visual_excerpt(chosen.paragraphs[anchor_paragraph - 1])
    elif placement_value == "SECTION_END" and chosen:
        excerpt = next(
            (candidate for paragraph in reversed(chosen.paragraphs) if (candidate := _visual_excerpt(paragraph))),
            "",
        )
    else:
        # Covers and introductory images should still be grounded in the
        # article rather than in a generic prompt. Prefer any visible pre-H2
        # introduction, then fall back to the first real paragraph.
        excerpt = _intro_visual_excerpt(content)
        if not excerpt:
            for section in sections:
                excerpt = next((_visual_excerpt(paragraph) for paragraph in section.paragraphs if _visual_excerpt(paragraph)), "")
                if excerpt:
                    break

    compact = re.sub(r"\s+", " ", excerpt).strip()
    if len(compact) <= excerpt_limit:
        return compact
    return compact[:excerpt_limit].rstrip() + "…"


def _intro_visual_excerpt(content: str) -> str:
    """Extract visible article text before the first true H2 section."""

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    fence: str | None = None
    lines: list[str] = []
    for line in normalized.split("\n"):
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        if _SECTION_HEADING.match(line):
            break
        if re.match(r"^#\s+", line):
            continue
        lines.append(line)
    return _visual_excerpt("\n".join(lines))


def _visual_excerpt(paragraph: str) -> str:
    """Remove fenced code so programming examples never become visual evidence."""

    fence: str | None = None
    visible: list[str] = []
    for line in paragraph.splitlines():
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            continue
        if fence is None:
            visible.append(line)
    return re.sub(r"\s+", " ", "\n".join(visible)).strip()

"""Small Markdown lexical helpers shared by validation and rendering."""

from __future__ import annotations


def split_inline_code_spans(line: str) -> list[tuple[bool, str]]:
    """Split a line into ordinary text and balanced backtick code spans.

    Markdown permits one or more backticks as the delimiter.  This deliberately
    small scanner keeps those spans opaque to HTML validation/sanitization while
    leaving unmatched backticks as ordinary prose for the Markdown parser.
    """

    parts: list[tuple[bool, str]] = []
    cursor = 0
    text_start = 0
    while cursor < len(line):
        if line[cursor] != "`":
            cursor += 1
            continue
        marker_end = cursor
        while marker_end < len(line) and line[marker_end] == "`":
            marker_end += 1
        marker = line[cursor:marker_end]
        closing = line.find(marker, marker_end)
        if closing < 0:
            cursor = marker_end
            continue
        if text_start < cursor:
            parts.append((False, line[text_start:cursor]))
        code_end = closing + len(marker)
        parts.append((True, line[cursor:code_end]))
        cursor = code_end
        text_start = cursor
    if text_start < len(line) or not parts:
        parts.append((False, line[text_start:]))
    return parts


def mask_inline_code_spans(line: str) -> str:
    """Replace inline code with equal-length spaces so match offsets stay valid."""

    return "".join(" " * len(part) if is_code else part for is_code, part in split_inline_code_spans(line))

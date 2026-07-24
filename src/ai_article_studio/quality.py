"""Deterministic delivery checks for generated Markdown articles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .markdown_syntax import mask_inline_code_spans

_H1 = re.compile(r"^#(?!#)\s+\S")
_H2 = re.compile(r"^##(?!#)\s+\S")
_FENCE_MARKER = re.compile(r"^\s*(`{3,}|~{3,})")
_FENCE_OPENING = re.compile(r"^\s*(`{3,}|~{3,})(?:[ \t]*([^\s`~]+))?[ \t]*$")
_FENCE_CLOSING = re.compile(r"^\s*(`{3,}|~{3,})[ \t]*$")
_RAW_HTML = re.compile(r"<!--|</?[a-z][a-z0-9-]*(?:\s+[^<>]*?)?\s*/?>", re.IGNORECASE)
_CODE_FENCE = re.compile(r"(?m)^\s*(`{3,}|~{3,})(?:[ \t]*([^\s`~]+))?[ \t]*$")
_UNSUPPORTED_RESOURCE = re.compile(
    r"(?:扫码|扫描(?:二维)?码|二维码|动作图解库|配套(?:动作)?图解|配套(?:资料|资源)|"
    r"(?:领取|下载|获取|查看).{0,12}(?:资料|资源|附件|课程|图解|模板|表格))"
)
_UNSOURCED_PROOF = re.compile(
    r"(?:亲测(?:有效)?|\d+(?:位|名|人)(?:首发)?(?:体验者|用户|学员|参与者)|"
    r"(?:专家|研究|科学|临床|权威).{0,16}(?:证明|证实|表明|显示))"
)
_UNSOURCED_QUANTITATIVE_CLAIM = re.compile(r"(?:\d+(?:\.\d+)?\s*[%％]|百分之[零〇一二三四五六七八九十百千万两]+)")
_HEALTH_GUARANTEE = re.compile(r"(?:(?:保证|承诺).{0,12}(?:治愈|康复|瘦身|见效)|(?:立刻|快速).{0,8}(?:见效|减脂|瘦身))")
_TECHNICAL_CONTEXT = re.compile(
    r"(?:代码|编程|程序|python|java|javascript|typescript|fastapi|django|flask|api|接口|sql|"
    r"算法|数据结构|前端|后端|工程|json|yaml|数据库)",
    re.IGNORECASE,
)
_QUALITY_GATE_MESSAGES = frozenset(
    {
        "代码围栏开头格式不正确。",
        "正文包含裸 HTML；请改用标准 Markdown。",
        "代码围栏未闭合。",
        "文章缺少唯一的一级标题（H1）。",
        "文章只能包含一个一级标题（H1）。",
        "文章缺少至少一个二级标题（H2）。",
        "JSON 代码块不是合法 JSON。",
        "代码样式内容存在明显的括号不平衡。",
        "正文包含虚构或未提供的外部资源；请删除扫码、附件、下载或配套资料宣称。",
        "正文包含无来源的用户数、亲测或权威背书；请删除或改为有明确来源的事实。",
        "正文包含无来源的比例或量化结论；请删除百分比等外部事实，或提供可核验来源。",
        "健康或健身内容不得作出保证性效果承诺；请改为审慎、个体化建议。",
        "当前选题不需要代码围栏；请改用自然语言、列表或 Markdown 表格。",
        "普通文章不得包含 JSON 代码块；请改用自然语言或 Markdown 表格。",
    }
)


@dataclass
class _CodeFence:
    marker: str
    language: str
    lines: list[str] = field(default_factory=list)


def markdown_blocking_issues(markdown_text: str) -> list[str]:
    """Return non-negotiable delivery defects without relying on an LLM.

    The checks are intentionally conservative: they protect the documented
    article contract but do not attempt to lint every programming language.
    """

    issues: list[str] = []
    h1_count = 0
    h2_count = 0
    fence: _CodeFence | None = None

    for line in markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if fence is not None:
            closing = _FENCE_CLOSING.match(line)
            if closing and _closes_fence(closing.group(1), fence.marker):
                _inspect_code_fence(fence, issues)
                fence = None
            else:
                fence.lines.append(line)
            continue

        marker = _FENCE_MARKER.match(line)
        opening = _FENCE_OPENING.match(line)
        if marker and opening is None:
            issues.append("代码围栏开头格式不正确。")
            continue
        if opening:
            fence = _CodeFence(marker=opening.group(1), language=(opening.group(2) or "").lower())
            continue
        if _H1.match(line):
            h1_count += 1
        if _H2.match(line):
            h2_count += 1
        if _RAW_HTML.search(mask_inline_code_spans(line)):
            issues.append("正文包含裸 HTML；请改用标准 Markdown。")

    if fence is not None:
        issues.append("代码围栏未闭合。")
    if h1_count == 0:
        issues.append("文章缺少唯一的一级标题（H1）。")
    elif h1_count > 1:
        issues.append("文章只能包含一个一级标题（H1）。")
    if h2_count == 0:
        issues.append("文章缺少至少一个二级标题（H2）。")
    return _unique(issues)


def factuality_contract_issues(text: str) -> list[str]:
    """Return deterministic blockers for invented resources and unsupported proof.

    The app currently accepts a topic, not a source bundle with verified URLs or
    citations.  Generated copy therefore must not manufacture external assets,
    audience counts, testimonials, or authority claims.
    """

    issues: list[str] = []
    prose = _prose_outside_code_fences(text)
    if _UNSUPPORTED_RESOURCE.search(prose):
        issues.append("正文包含虚构或未提供的外部资源；请删除扫码、附件、下载或配套资料宣称。")
    if _UNSOURCED_PROOF.search(prose):
        issues.append("正文包含无来源的用户数、亲测或权威背书；请删除或改为有明确来源的事实。")
    if _UNSOURCED_QUANTITATIVE_CLAIM.search(prose):
        issues.append("正文包含无来源的比例或量化结论；请删除百分比等外部事实，或提供可核验来源。")
    if _HEALTH_GUARANTEE.search(prose):
        issues.append("健康或健身内容不得作出保证性效果承诺；请改为审慎、个体化建议。")
    return issues


def content_contract_issues(markdown_text: str, *, topic: str, title: str, style: str) -> list[str]:
    """Validate the delivery contract for an article before it is persisted.

    Markdown syntax checks remain reusable for technical writing.  This second
    layer applies audience-facing rules: no fabricated resources or social proof,
    and no fenced code in an article that is not clearly about technical work.
    """

    issues = [*markdown_blocking_issues(markdown_text), *factuality_contract_issues(markdown_text)]
    languages = [match.group(2).lower() if match.group(2) else "" for match in _CODE_FENCE.finditer(markdown_text)]
    if languages and not _allows_fenced_code(topic, title, style):
        if any(language == "json" for language in languages):
            issues.append("普通文章不得包含 JSON 代码块；请改用自然语言或 Markdown 表格。")
        if any(language != "json" for language in languages):
            issues.append("当前选题不需要代码围栏；请改用自然语言、列表或 Markdown 表格。")
    return _unique(issues)


def is_quality_gate_issue(issue: str) -> bool:
    """Whether an exact blocker is maintained by the deterministic Markdown gate."""

    return issue in _QUALITY_GATE_MESSAGES


def _closes_fence(marker: str, opening_marker: str) -> bool:
    return marker[0] == opening_marker[0] and len(marker) >= len(opening_marker)


def _inspect_code_fence(fence: _CodeFence, issues: list[str]) -> None:
    code = "\n".join(fence.lines)
    if fence.language == "json":
        try:
            json.loads(code)
        except json.JSONDecodeError:
            issues.append("JSON 代码块不是合法 JSON。")
    if _has_unbalanced_delimiters(code):
        issues.append("代码样式内容存在明显的括号不平衡。")


def _has_unbalanced_delimiters(code: str) -> bool:
    """Catch obvious delimiter mistakes while ignoring quoted literal strings."""

    opening_for = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for character in code:
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}":
            if not stack or stack[-1] != opening_for[character]:
                return True
            stack.pop()
    return bool(stack)


def _allows_fenced_code(topic: str, title: str, style: str) -> bool:
    return style.strip().upper() == "TECH" and bool(_TECHNICAL_CONTEXT.search(f"{topic}\n{title}"))


def _prose_outside_code_fences(text: str) -> str:
    """Keep semantic claim checks out of code examples and JSON payloads."""

    prose: list[str] = []
    fence_marker: str | None = None
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        marker = _FENCE_MARKER.match(line)
        if fence_marker is not None:
            if marker and _closes_fence(marker.group(1), fence_marker):
                fence_marker = None
            continue
        opening = _FENCE_OPENING.match(line)
        if opening:
            fence_marker = opening.group(1)
            continue
        prose.append(line)
    return "\n".join(prose)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))

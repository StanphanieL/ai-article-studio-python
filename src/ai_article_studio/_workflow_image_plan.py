"""Private, side-effect-free image-plan normalization and editor contracts."""

from __future__ import annotations

import json

from .anchors import MarkdownAnchorSection, heading_key, image_anchor_excerpt, markdown_anchor_sections
from .domain import WorkflowValidationError
from .schemas import ImagePlacement, ImagePromptOption, ImageResultOption, ImageVisualRole


def normalize_generated_image_plan(
    content: str,
    plans: list[ImagePromptOption],
) -> tuple[list[ImagePromptOption], int]:
    """Repair recoverable model output before applying hard invariants.

    A planner receives an authoritative catalogue, but a model can still
    mistake a wrapped line for a second paragraph or over-request images
    for one insertion point. Preserve the image meaning while repairing or
    reducing generated plans; user edits deliberately bypass this method.
    """

    sections = markdown_anchor_sections(content)
    by_key: dict[str, MarkdownAnchorSection] = {}
    for section in sections:
        by_key.setdefault(section.key, section)
    normalized: list[ImagePromptOption] = []
    adjusted_indices: set[int] = set()
    fallback_cursor = 0
    known_sections: list[bool] = []

    def choose_fallback(excluded: set[str]) -> tuple[str, int, ImagePlacement]:
        nonlocal fallback_cursor
        if not sections:
            return "", 0, ImagePlacement.AFTER_INTRO
        for offset in range(len(sections)):
            section = sections[(fallback_cursor + offset) % len(sections)]
            candidate = f"section-end:{section.key}"
            if candidate not in excluded:
                fallback_cursor = (fallback_cursor + offset + 1) % len(sections)
                return section.heading, 0, ImagePlacement.SECTION_END
        section = sections[fallback_cursor % len(sections)]
        fallback_cursor = (fallback_cursor + 1) % len(sections)
        return section.heading, 0, ImagePlacement.SECTION_END

    previous_anchor = ""
    for index, plan in enumerate(plans):
        original = plan
        section = by_key.get(heading_key(plan.anchor_heading)) if plan.placement not in {ImagePlacement.COVER, ImagePlacement.AFTER_INTRO} else None
        known_sections.append(section is not None)
        if plan.placement is ImagePlacement.COVER:
            plan = plan.model_copy(update={"placement": ImagePlacement.COVER, "anchor_heading": "", "anchor_paragraph": 0})
        elif plan.placement is ImagePlacement.AFTER_INTRO:
            plan = plan.model_copy(update={"anchor_heading": "", "anchor_paragraph": 0})
        else:
            if section is None:
                heading, paragraph, placement = choose_fallback({previous_anchor})
                plan = plan.model_copy(update={"anchor_heading": heading, "anchor_paragraph": paragraph, "placement": placement})
            elif plan.placement is ImagePlacement.AFTER_PARAGRAPH:
                if section.paragraphs:
                    paragraph = min(max(plan.anchor_paragraph, 1), len(section.paragraphs))
                    plan = plan.model_copy(update={"anchor_heading": section.heading, "anchor_paragraph": paragraph})
                else:
                    plan = plan.model_copy(
                        update={
                            "anchor_heading": section.heading,
                            "anchor_paragraph": 0,
                            "placement": ImagePlacement.SECTION_END,
                        }
                    )
            else:
                plan = plan.model_copy(update={"anchor_heading": section.heading, "anchor_paragraph": 0, "placement": ImagePlacement.SECTION_END})

        if plan != original:
            adjusted_indices.add(index)
        normalized.append(plan)
        previous_anchor = image_anchor_key(plan)

    occupied: set[str] = set()
    grounded: list[ImagePromptOption] = []
    for index, plan in enumerate(normalized):
        # Keep every cover-shaped item for the strict cover validator below.
        # Never silently discard a malformed or second cover.
        if plan.placement is not ImagePlacement.COVER:
            anchor = _visual_insertion_key(plan, by_key)
            if anchor in occupied:
                if plan.placement is ImagePlacement.AFTER_INTRO:
                    candidates = []
                elif known_sections[index]:
                    candidates = [by_key[heading_key(plan.anchor_heading)]]
                else:
                    candidates = sections
                replacement = next(
                    (
                        candidate
                        for section in candidates
                        for candidate in _alternate_section_anchors(plan, section)
                        if _visual_insertion_key(candidate, by_key) not in occupied
                    ),
                    None,
                )
                if replacement is None:
                    adjusted_indices.add(index)
                    continue
                plan = replacement
                adjusted_indices.add(index)
                anchor = _visual_insertion_key(plan, by_key)
            occupied.add(anchor)
        evidence_bound = with_anchor_excerpt(content, plan)
        if evidence_bound != plan:
            adjusted_indices.add(index)
        grounded.append(evidence_bound)
    return grounded, len(adjusted_indices)


def _visual_insertion_key(plan: ImagePromptOption, sections_by_key: dict[str, MarkdownAnchorSection]) -> str:
    """Use the rendered position, where section end equals its final paragraph."""

    key = image_anchor_key(plan)
    if plan.placement is ImagePlacement.SECTION_END:
        section = sections_by_key.get(heading_key(plan.anchor_heading))
        if section is not None and section.paragraphs:
            return f"paragraph:{section.key}:{len(section.paragraphs)}"
    return key


def _alternate_section_anchors(plan: ImagePromptOption, section: MarkdownAnchorSection) -> list[ImagePromptOption]:
    """Offer the closest real paragraph positions without changing section or role."""

    count = len(section.paragraphs)
    if not count:
        return [plan.model_copy(update={"placement": ImagePlacement.SECTION_END, "anchor_heading": section.heading, "anchor_paragraph": 0})]
    preferred = plan.anchor_paragraph if plan.placement is ImagePlacement.AFTER_PARAGRAPH else count
    return [
        plan.model_copy(update={"placement": ImagePlacement.AFTER_PARAGRAPH, "anchor_heading": section.heading, "anchor_paragraph": paragraph})
        for paragraph in sorted(range(1, count + 1), key=lambda number: (abs(number - preferred), number))
    ]


def with_anchor_excerpt(content: str, plan: ImagePromptOption) -> ImagePromptOption:
    """Bind a plan to the authoritative prose it should make visible."""

    return plan.model_copy(
        update={
            "anchor_excerpt": image_anchor_excerpt(
                content,
                plan.placement,
                plan.anchor_heading,
                plan.anchor_paragraph,
            )
        }
    )


def replacement_visual_role(plan: ImagePromptOption, used_roles: set[ImageVisualRole]) -> ImageVisualRole | None:
    """Compatibility hook: repeated roles are valid and never need replacing."""

    return None


def validate_image_plan_anchor(content: str, plan: ImagePromptOption) -> None:
    """Reject stale or impossible editor anchors before persisting them."""

    if not content.strip():
        raise WorkflowValidationError("请先生成正文，再调整图片位置")
    anchor_heading = plan.anchor_heading.strip()
    placement = plan.placement
    if placement in {ImagePlacement.COVER, ImagePlacement.AFTER_INTRO}:
        if anchor_heading or plan.anchor_paragraph:
            raise WorkflowValidationError("封面或导语后的配图不能指定章节或段落锚点")
        return

    sections = markdown_sections(content)
    matching = next(
        ((heading, blocks) for heading, blocks in sections if canonical_heading_key(heading) == canonical_heading_key(anchor_heading)),
        None,
    )
    if matching is None:
        raise WorkflowValidationError(f"未找到锚点标题：{anchor_heading or '（未填写）'}")
    _heading, blocks = matching
    if placement is ImagePlacement.AFTER_PARAGRAPH:
        if not 1 <= plan.anchor_paragraph <= len(blocks):
            raise WorkflowValidationError(f"锚点段落无效：标题“{anchor_heading}”下共有 {len(blocks)} 段，不能定位第 {plan.anchor_paragraph} 段")
        return
    if plan.anchor_paragraph != 0:
        raise WorkflowValidationError("章节末尾配图的段落序号必须为 0")


def validate_image_plan_contract(content: str, plans: list[ImagePromptOption]) -> None:
    """Keep generated and manually arranged plans structurally composable."""

    if not plans:
        raise WorkflowValidationError("配图方案不能为空")
    for plan in plans:
        validate_image_plan_anchor(content, plan)

    cover_indices = [index for index, plan in enumerate(plans) if plan.visual_role is ImageVisualRole.COVER]
    if len(cover_indices) != 1:
        raise WorkflowValidationError("配图方案必须且只能包含一张封面图")
    if cover_indices[0] != 0:
        raise WorkflowValidationError("封面图必须位于配图方案第 1 项")
    if plans[0].placement is not ImagePlacement.COVER or any(plan.placement is ImagePlacement.COVER for plan in plans[1:]):
        raise WorkflowValidationError("封面图必须使用 COVER 位置，且其他配图不能使用 COVER 位置")

    by_key = {section.key: section for section in markdown_anchor_sections(content)}
    for previous, current in zip(plans, plans[1:]):
        if _visual_insertion_key(previous, by_key) == _visual_insertion_key(current, by_key):
            raise WorkflowValidationError("相邻配图不能使用同一锚点")


def image_anchor_key(plan: ImagePromptOption) -> str:
    if plan.placement is ImagePlacement.COVER:
        return "cover"
    if plan.placement is ImagePlacement.AFTER_INTRO:
        return "intro"
    heading = canonical_heading_key(plan.anchor_heading)
    if plan.placement is ImagePlacement.AFTER_PARAGRAPH:
        return f"paragraph:{heading}:{plan.anchor_paragraph}"
    return f"section-end:{heading}"


def markdown_sections(content: str) -> list[tuple[str, tuple[str, ...]]]:
    return [(section.heading, section.paragraphs) for section in markdown_anchor_sections(content)]


def canonical_heading_key(value: str) -> str:
    return heading_key(value)


def dump_image_records(records: list[ImagePromptOption] | list[ImageResultOption]) -> str:
    return json.dumps([record.model_dump(by_alias=True) for record in records], ensure_ascii=False)

"""Private, side-effect-free image-plan normalization and editor contracts."""

from __future__ import annotations

import json

from .anchors import heading_key, image_anchor_excerpt, markdown_anchor_sections
from .domain import WorkflowValidationError
from .schemas import ImagePlacement, ImagePromptOption, ImageResultOption, ImageVisualRole

_GENERATED_ROLE_FALLBACK_ORDER = (
    ImageVisualRole.CONCEPT,
    ImageVisualRole.DIAGRAM,
    ImageVisualRole.SCENE,
    ImageVisualRole.COMPARISON,
    ImageVisualRole.DETAIL,
    ImageVisualRole.DATA,
    ImageVisualRole.CLOSING,
)


def normalize_generated_image_plan(
    content: str,
    plans: list[ImagePromptOption],
) -> tuple[list[ImagePromptOption], int]:
    """Repair recoverable model output before applying hard invariants.

    A planner receives an authoritative catalogue, but a model can still
    mistake a wrapped line for a second paragraph or reuse an otherwise
    valid visual role. Neither should turn a harmless planning variation
    into a failed one-click article. User edits deliberately bypass this
    method and remain strict.
    """

    sections = markdown_anchor_sections(content)
    by_key: dict[str, object] = {}
    for section in sections:
        by_key.setdefault(section.key, section)
    normalized: list[ImagePromptOption] = []
    adjusted_indices: set[int] = set()
    fallback_cursor = 0

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
        if plan.placement is ImagePlacement.COVER:
            plan = plan.model_copy(update={"placement": ImagePlacement.COVER, "anchor_heading": "", "anchor_paragraph": 0})
        elif plan.placement is ImagePlacement.AFTER_INTRO:
            plan = plan.model_copy(update={"anchor_heading": "", "anchor_paragraph": 0})
        else:
            section = by_key.get(heading_key(plan.anchor_heading))
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

    # ``visual_role`` is presentation metadata, not a structural anchor.
    # The model is told to keep it unique, but repairing a duplicate here
    # is safer than failing before any image-provider call. COVER remains
    # deliberately strict: a missing, duplicated, or misplaced cover is a
    # structural error that must be shown to the user rather than guessed.
    used_roles: set[ImageVisualRole] = set()
    role_normalized: list[ImagePromptOption] = []
    for index, plan in enumerate(normalized):
        if plan.visual_role in used_roles and plan.visual_role is not ImageVisualRole.COVER:
            replacement = replacement_visual_role(plan, used_roles)
            if replacement is not None:
                plan = plan.model_copy(update={"visual_role": replacement})
                adjusted_indices.add(index)
        role_normalized.append(plan)
        used_roles.add(plan.visual_role)
    grounded: list[ImagePromptOption] = []
    for index, plan in enumerate(role_normalized):
        evidence_bound = with_anchor_excerpt(content, plan)
        if evidence_bound != plan:
            adjusted_indices.add(index)
        grounded.append(evidence_bound)
    return grounded, len(adjusted_indices)


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


def replacement_visual_role(
    plan: ImagePromptOption,
    used_roles: set[ImageVisualRole],
) -> ImageVisualRole | None:
    """Choose a deterministic unused role for a duplicated model plan.

    The role remains meaningful for AUTO provider routing, so first prefer
    candidates indicated by the image's title, scene, or prompts. The
    stable fallback order makes retries reproducible when the text offers
    no useful hint.
    """

    semantic_text = " ".join((plan.image_title, plan.usage_scene, plan.prompt_zh, plan.prompt_en)).casefold()
    preferred: list[ImageVisualRole] = []

    def add_if_matches(role: ImageVisualRole, keywords: tuple[str, ...]) -> None:
        if any(keyword in semantic_text for keyword in keywords):
            preferred.append(role)

    add_if_matches(
        ImageVisualRole.DIAGRAM,
        ("diagram", "flow", "workflow", "architecture", "interface", "ui", "图解", "流程", "架构", "界面", "信息图"),
    )
    add_if_matches(
        ImageVisualRole.DATA,
        ("data", "chart", "metric", "dashboard", "数据", "图表", "指标", "看板"),
    )
    add_if_matches(
        ImageVisualRole.COMPARISON,
        ("compare", "comparison", "versus", "对比", "比较", "前后"),
    )
    add_if_matches(
        ImageVisualRole.SCENE,
        ("scene", "photo", "portrait", "people", "人物", "场景", "照片"),
    )
    add_if_matches(
        ImageVisualRole.DETAIL,
        ("detail", "closeup", "局部", "细节", "特写"),
    )
    add_if_matches(
        ImageVisualRole.CLOSING,
        ("closing", "conclusion", "ending", "结尾", "结语", "收束"),
    )

    for candidate in (*preferred, *_GENERATED_ROLE_FALLBACK_ORDER):
        if candidate not in used_roles:
            return candidate
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

    roles = [plan.visual_role for plan in plans]
    if len(set(roles)) != len(roles):
        raise WorkflowValidationError("同一配图方案中的视觉角色不能重复")

    for previous, current in zip(plans, plans[1:]):
        if image_anchor_key(previous) == image_anchor_key(current):
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

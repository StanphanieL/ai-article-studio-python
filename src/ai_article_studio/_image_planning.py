"""Bounded visual briefs and deterministic placement for the LLM adapter.

Only visual descriptions cross the model boundary. Placement IDs resolve to
real Markdown paragraphs here, never to paragraph counts invented by a model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .anchors import image_anchor_excerpt, markdown_anchor_sections
from .schemas import ImageLayout, ImagePlacement, ImagePromptOption, ImageVisualRole


@dataclass(frozen=True, slots=True)
class ImagePlanSlot:
    slot_id: str
    placement: ImagePlacement
    heading: str = ""
    paragraph: int = 0
    excerpt: str = ""


@dataclass(frozen=True, slots=True)
class ImagePlanCall:
    stage: str
    duration_ms: int
    succeeded: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str = "unknown"

    def summary(self) -> str:
        input_count = self.input_tokens if self.input_tokens is not None else "未知"
        output_count = self.output_tokens if self.output_tokens is not None else "未知"
        usage = f"输入 {input_count} / 输出 {output_count} tokens"
        return f"{self.stage} {self.duration_ms}ms（{'成功' if self.succeeded else '失败'}，{usage}，{self.finish_reason}）"


def build_image_plan_slots(content: str, count: int) -> list[ImagePlanSlot]:
    """Spread a bounded number of positions across actual prose sections.

    Prefer section coverage, then at most one extra position per section. A
    short article with too few distinct positions gets fewer images instead of
    requiring a model to invent anchors or stack adjacent images.
    """

    count = max(1, min(count, 8))
    sections = markdown_anchor_sections(content)
    cover_excerpt = image_anchor_excerpt(content, ImagePlacement.COVER, excerpt_limit=260)
    headings = " / ".join(section.heading for section in sections)[:240]
    cover = ImagePlanSlot("cover", ImagePlacement.COVER, excerpt=f"{cover_excerpt}\n章节：{headings}" if headings else cover_excerpt)
    primary: list[tuple[int, ImagePlanSlot]] = []
    secondary: list[tuple[int, ImagePlanSlot]] = []
    seen_headings: set[str] = set()
    for section_index, section in enumerate(sections, 1):
        # Duplicate headings cannot be addressed unambiguously by the existing
        # public placement schema. Do not create conflicting IDs for them.
        if section.key in seen_headings:
            continue
        seen_headings.add(section.key)
        candidates: list[ImagePlanSlot] = []
        for paragraph in range(1, len(section.paragraphs) + 1):
            excerpt = image_anchor_excerpt(content, ImagePlacement.AFTER_PARAGRAPH, section.heading, paragraph, excerpt_limit=300)
            if excerpt:
                candidates.append(ImagePlanSlot(f"s{section_index}p{paragraph}", ImagePlacement.AFTER_PARAGRAPH, section.heading, paragraph, excerpt))
        if candidates:
            primary.append((section_index, candidates[0]))
            if len(candidates) > 1:
                secondary.append((section_index, candidates[-1]))

    capacity = count - 1
    if len(primary) > capacity:
        indexes = [round(index * (len(primary) - 1) / max(1, capacity - 1)) for index in range(capacity)]
        selected = [primary[index] for index in indexes]
    else:
        selected = primary + secondary[: max(0, capacity - len(primary))]
    selected.sort(key=lambda pair: (pair[0], pair[1].paragraph))
    if not sections and capacity and cover_excerpt:
        return [cover, ImagePlanSlot("intro", ImagePlacement.AFTER_INTRO, excerpt=cover_excerpt)]
    return [cover, *(slot for _section_index, slot in selected)]


def image_plan_prompt(topic: str, title: str, style: str, slots: list[ImagePlanSlot], *, stock_search: bool, problems: dict[str, str] | None = None) -> str:
    fields = "slotId、imageTitle、usageScene、promptZh、promptEn、visualRole"
    stock_rule = ""
    if stock_search:
        fields += "、stockQuery"
        stock_rule = "stockQuery 仅用于照片检索，写 2-5 个描述可见主体的小写英文词；DIAGRAM、DATA 可留空。\n"
    briefs = [
        {
            "slotId": slot.slot_id,
            "section": slot.heading or "文章概览",
            "evidence": slot.excerpt,
            **({"problem": problems[slot.slot_id]} if problems and slot.slot_id in problems else {}),
        }
        for slot in slots
    ]
    return (
        ("仅补全或修复下面列出的配图；已成功的项已保存，不要重复返回。\n" if problems is not None else "为下面的正文摘录设计配图。\n")
        + f"选题：{topic[:240]}；标题：{title[:240]}；风格：{style}。\n"
        + f"只输出含 {len(slots)} 项的 JSON 数组，每项字段：{fields}；所有字段都是字符串。逐项原样返回 slotId。\n"
        + "imageTitle 用简短画面名称；usageScene 用一句话说明图片解释的具体关系；"
        + "promptZh 用 60-100 字、promptEn 用 35-55 词描述同一画面，不要重复解释设计过程。\n"
        + "画面必须包含对应 evidence 中的具体对象、动作或关系，写清主体、关系、构图和风格。"
        + "机制或方法画成流程/对比，勿用无关办公或科技背景。不得凭空加入实验室、XRD/衍射、医疗或其他行业设备。"
        + "不得编造数据、二维码、下载资源；无文字、Logo、水印（no text, no logo, no watermark）。\n"
        + "visualRole 可选 COVER、SCENE、CONCEPT、DIAGRAM、COMPARISON、DETAIL、DATA、CLOSING；"
        + "cover 项必须是 COVER，其他项不能是 COVER，非封面角色可以重复。"
        + "SCENE 仅用于证据中明确的真实场景；DATA 仅用于确有数据关系的证据。位置和版式由系统处理。\n"
        + stock_rule
        + "正文摘录：\n"
        + json.dumps(briefs, ensure_ascii=False, separators=(",", ":"))
    )


def completed_image_plan_items(raw: str) -> list[dict[str, object]]:
    """Recover complete top-level objects even when the last item is truncated.

    Never repair punctuation or harvest nested objects: only a fully decoded
    item can be retained. Missing/broken items use the one bounded repair call.
    """

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start = cleaned.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    cursor = start + 1
    items: list[dict[str, object]] = []
    while cursor < len(cleaned):
        while cursor < len(cleaned) and cleaned[cursor].isspace():
            cursor += 1
        if cursor == len(cleaned) or cleaned[cursor] == "]":
            break
        try:
            item, cursor = decoder.raw_decode(cleaned, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
        while cursor < len(cleaned) and cleaned[cursor].isspace():
            cursor += 1
        if cursor == len(cleaned) or cleaned[cursor] != ",":
            break
        cursor += 1
    return items


def materialize_image_plan(item: dict[str, object], slot: ImagePlanSlot) -> ImagePromptOption:
    """Fill compatibility fields locally; never accept model-supplied anchors."""

    required = {"imageTitle": "image_title", "usageScene": "usage_scene", "promptZh": "prompt_zh", "promptEn": "prompt_en"}
    values: dict[str, str] = {}
    for public_name, python_name in required.items():
        value = item.get(public_name, item.get(python_name))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{public_name} 必须为非空字符串")
        values[python_name] = value.strip()
    is_cover = slot.placement is ImagePlacement.COVER
    role_value = item.get("visualRole", item.get("visual_role"))
    if role_value is None and is_cover and "slotId" not in item and "slot_id" not in item:
        # A historical first cover record may omit presentation metadata.
        # Compact model output must always declare its intended visual role.
        role_value = "COVER"
    try:
        role = ImageVisualRole(role_value)
    except (ValueError, TypeError):
        raise ValueError("visualRole 必须使用允许的角色") from None
    if is_cover and role is not ImageVisualRole.COVER:
        raise ValueError("cover 项必须使用 COVER 角色")
    if role is ImageVisualRole.COVER and not is_cover:
        raise ValueError("正文配图不能使用 COVER 角色")
    layout = ImageLayout.STANDARD
    if is_cover:
        layout = ImageLayout.HERO
    elif role in {ImageVisualRole.CONCEPT, ImageVisualRole.DIAGRAM, ImageVisualRole.COMPARISON, ImageVisualRole.DATA}:
        layout = ImageLayout.WIDE
    elif role is ImageVisualRole.CLOSING:
        layout = ImageLayout.COMPACT
    query = item.get("stockQuery", item.get("stock_query", ""))
    return ImagePromptOption(
        **values,
        visual_role=role,
        placement=slot.placement,
        anchor_heading=slot.heading,
        anchor_paragraph=slot.paragraph,
        layout=layout,
        stock_query=query if isinstance(query, str) else "",
    )

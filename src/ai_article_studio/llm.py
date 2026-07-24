"""Article-generation agents with real OpenAI-compatible and offline mock modes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Protocol, TypeVar

import httpx
from pydantic import ValidationError

from .anchors import image_anchor_catalog, image_visual_brief_catalog
from .config import Settings
from .schemas import (
    ArticleModelConfig,
    ImageLayout,
    ImagePlacement,
    ImagePromptOption,
    ImageVisualRole,
    OutlineItem,
    QualityReviewResult,
    TitleOption,
)

ValidatedJson = TypeVar("ValidatedJson")


class ArticleAgent(Protocol):
    def titles(self, topic: str, style: str) -> list[TitleOption]: ...

    def outline(self, topic: str, title: str, style: str) -> list[OutlineItem]: ...

    def content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str: ...

    def repair_content(
        self,
        topic: str,
        title: str,
        outline: list[OutlineItem],
        style: str,
        length: str,
        draft: str,
        violations: list[str],
    ) -> str: ...

    def stream_content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> Iterator[str]: ...

    def image_prompts(self, topic: str, title: str, content: str, style: str, count: int) -> list[ImagePromptOption]: ...

    def review(self, article: str) -> QualityReviewResult: ...


class MockArticleAgent:
    """Deterministic local content used for tests, demos, and no-key onboarding."""

    def titles(self, topic: str, style: str) -> list[TitleOption]:
        return [
            TitleOption(title=f"{topic}：从原理到落地", reason="突出方法和可执行性"),
            TitleOption(title=f"如何系统化理解{topic}", reason="面向入门读者"),
            TitleOption(title=f"{topic}的实践清单", reason="强调操作步骤"),
        ]

    def outline(self, topic: str, title: str, style: str) -> list[OutlineItem]:
        return [
            OutlineItem(heading="问题与目标", description=f"说明{topic}解决的真实问题"),
            OutlineItem(heading="核心方法", description="拆解关键流程和技术取舍"),
            OutlineItem(heading="实践建议", description="给出可复用的落地步骤"),
        ]

    def content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str:
        sections = []
        for item in outline:
            sections.append(f"## {item.heading}\n\n{item.description}。围绕“{topic}”，先明确边界，再通过可追踪的步骤验证结果。")
        return f"# {title}\n\n本文以可控、可解释的方式梳理 {topic}。\n\n" + "\n\n".join(sections)

    def repair_content(
        self,
        topic: str,
        title: str,
        outline: list[OutlineItem],
        style: str,
        length: str,
        draft: str,
        violations: list[str],
    ) -> str:
        # Keep the offline mode deterministic even when a test deliberately
        # injects an invalid subclass response.
        return MockArticleAgent.content(self, topic, title, outline, style, length)

    def stream_content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> Iterator[str]:
        content = self.content(topic, title, outline, style, length)
        for index in range(0, len(content), 48):
            yield content[index : index + 48]

    def image_prompts(self, topic: str, title: str, content: str, style: str, count: int) -> list[ImagePromptOption]:
        headings = re.findall(r"(?m)^##\s+(.+)$", content)
        plans = [
            ImagePromptOption(
                image_title=f"{title}封面",
                usage_scene="文章封面",
                prompt_zh=f"{style} 风格的 {topic} 文章封面，简洁专业，无大段文字",
                prompt_en=f"A clean professional {style} article cover about {topic}, no large text",
                visual_role=ImageVisualRole.COVER,
                placement=ImagePlacement.COVER,
                layout=ImageLayout.HERO,
                stock_query="professional editorial cover",
            )
        ]
        roles = (
            ImageVisualRole.CONCEPT,
            ImageVisualRole.DIAGRAM,
            ImageVisualRole.SCENE,
            ImageVisualRole.COMPARISON,
            ImageVisualRole.DETAIL,
            ImageVisualRole.DATA,
        )
        stock_queries = {
            ImageVisualRole.CONCEPT: "abstract editorial concept",
            ImageVisualRole.DIAGRAM: "workflow concept illustration",
            ImageVisualRole.SCENE: "professional work scene",
            ImageVisualRole.COMPARISON: "modern visual comparison",
            ImageVisualRole.DETAIL: "editorial detail closeup",
            ImageVisualRole.DATA: "clear data visualization",
            ImageVisualRole.CLOSING: "calm editorial ending",
        }
        layouts = {
            ImageVisualRole.CONCEPT: ImageLayout.WIDE,
            ImageVisualRole.DIAGRAM: ImageLayout.WIDE,
            ImageVisualRole.SCENE: ImageLayout.STANDARD,
            ImageVisualRole.COMPARISON: ImageLayout.WIDE,
            ImageVisualRole.DETAIL: ImageLayout.STANDARD,
            ImageVisualRole.DATA: ImageLayout.WIDE,
            ImageVisualRole.CLOSING: ImageLayout.COMPACT,
        }
        for index in range(1, count):
            is_closing = index == count - 1
            role = ImageVisualRole.CLOSING if is_closing else roles[(index - 1) % len(roles)]
            heading_index = -1 if is_closing and headings else min(index - 1, len(headings) - 1)
            heading = headings[heading_index] if headings else f"第 {index} 个要点"
            placement = ImagePlacement.SECTION_END if is_closing else ImagePlacement.AFTER_PARAGRAPH
            plans.append(
                ImagePromptOption(
                    image_title=f"{heading}配图",
                    usage_scene=f"第 {index} 节正文配图：{heading}",
                    prompt_zh=f"围绕{heading}的{style}风格文章配图，主体明确，无文字水印",
                    prompt_en=f"A {style} editorial illustration about {heading}, clear subject, no text",
                    visual_role=role,
                    placement=placement,
                    anchor_heading=heading if headings else "",
                    anchor_paragraph=1 if placement is ImagePlacement.AFTER_PARAGRAPH else 0,
                    layout=layouts[role],
                    stock_query=stock_queries[role],
                )
            )
        return plans

    def review(self, article: str) -> QualityReviewResult:
        return QualityReviewResult(score=92, passed=True, summary="结构完整，图文稿可交付")


class OpenAICompatibleArticleAgent(MockArticleAgent):
    """Real adapter; mock fallbacks are deliberately unavailable in REAL mode."""

    def __init__(self, settings: Settings, model_config: ArticleModelConfig | None = None) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("LLM API key is not configured")
        self.settings = settings
        self.model_config = model_config or ArticleModelConfig(text_model=settings.openai_model)
        self.client = httpx.Client(timeout=settings.request_timeout_seconds)

    def with_model_config(self, model_config: ArticleModelConfig) -> OpenAICompatibleArticleAgent:
        """Return a per-article adapter without mutable cross-task request state."""

        configured = OpenAICompatibleArticleAgent(self.settings, model_config)
        configured.client = self.client
        return configured

    def _request_payload(self, prompt: str, *, stream: bool = False) -> dict[str, object]:
        return {
            "model": self.model_config.text_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.model_config.temperature,
            "max_completion_tokens": self.model_config.max_completion_tokens,
            **({"stream": True} if stream else {}),
        }

    def _chat(self, prompt: str) -> str:
        response = self.client.post(
            _chat_completions_url(self.settings.openai_base_url),
            headers={"Authorization": f"Bearer {self.settings.openai_api_key.get_secret_value()}"},
            json=self._request_payload(prompt),
        )
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response has no content")
        return content

    def _json(
        self,
        prompt: str,
        *,
        validator: Callable[[object], ValidatedJson] | None = None,
        schema: str = "合法 JSON",
    ) -> ValidatedJson | object:
        raw = self._chat(prompt)
        try:
            payload = _parse_json_payload(raw)
            return validator(payload) if validator else payload
        except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as error:
            repaired = self._chat(_json_repair_prompt(raw, schema, violation=str(error)))
            payload = _parse_json_payload(repaired)
            return validator(payload) if validator else payload

    def stream_content(
        self,
        topic: str,
        title: str,
        outline: list[OutlineItem],
        style: str,
        length: str,
    ) -> Iterator[str]:
        prompt = _content_prompt(topic, title, outline, style, length)
        with self.client.stream(
            "POST",
            _chat_completions_url(self.settings.openai_base_url),
            headers={"Authorization": f"Bearer {self.settings.openai_api_key.get_secret_value()}"},
            json=self._request_payload(prompt, stream=True),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    return
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (IndexError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if isinstance(delta, str) and delta:
                    yield delta

    def titles(self, topic: str, style: str) -> list[TitleOption]:
        return self._json(
            f"为选题“{topic}”生成 3 个 {style} 风格标题。\n{_factuality_contract()}\n"
            "只返回 JSON 数组，每项必须是 {\"title\": string, \"reason\": string}。",
            validator=_title_options,
            schema="JSON 数组；每一项严格为 title 和 reason 两个字符串字段。",
        )  # type: ignore[return-value]

    def outline(self, topic: str, title: str, style: str) -> list[OutlineItem]:
        return self._json(
            f"为选题“{topic}”、标题“{title}”生成 3-5 项 {style} 风格文章大纲。\n{_factuality_contract()}\n"
            "只返回 JSON 数组，每项必须是 {\"heading\": string, \"description\": string}。",
            validator=_outline_items,
            schema="JSON 数组；每一项严格为 heading 和 description 两个字符串字段。",
        )  # type: ignore[return-value]

    def content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str:
        return self._chat(_content_prompt(topic, title, outline, style, length))

    def repair_content(
        self,
        topic: str,
        title: str,
        outline: list[OutlineItem],
        style: str,
        length: str,
        draft: str,
        violations: list[str],
    ) -> str:
        return self._chat(_content_repair_prompt(topic, title, outline, style, length, draft, violations))

    def image_prompts(self, topic: str, title: str, content: str, style: str, count: int) -> list[ImagePromptOption]:
        def validate_plan_count(payload: object) -> list[ImagePromptOption]:
            plans = _generated_image_prompt_options(payload)
            if len(plans) != count:
                raise ValueError(f"expected exactly {count} image prompts, received {len(plans)}")
            _validate_visual_grounding(plans, f"{topic}\n{title}\n{content}")
            return plans

        prompts = self._json(
            f"根据文章“{title}”生成严格 {count} 个语义配图方案，第一项必须为封面。\n"
            "图片不得包含二维码、下载/课程宣称、虚构数据背书、大段文字、Logo 或水印。\n"
            "每张图必须解释一个明确论点，而不是只为章节装饰：先从下方“段落视觉简报”选择锚点，再把该段的因果、对比、过程或具体对象转成可见画面。"
            "usageScene 必须用一句话说明“这张图让读者看懂哪一个关系”，不得写“第几节配图”“科研感场景”之类的空话。"
            "所选段落中的至少一个具体对象、动作或关系必须进入 promptZh 和 promptEn；两种提示词必须描述同一画面，"
            "不能把段落的论点改写成无关的通用办公、实验室、设备或科技背景。"
            "promptZh 与 promptEn 都必须写出可见主体、主体之间的动作/关系、环境或构图、画面风格，并明确 no text/no logo/no watermark；"
            "不得使用“抽象概念图、专业场景、科技感背景、实验室氛围”等无法直接生成的泛化描述。\n"
            "角色选择规则：SCENE 只能用于锚点中明确出现的人物、地点或实物场景；机制、方法、判断条件、流程和限制优先使用 DIAGRAM、COMPARISON 或 CONCEPT；"
            "DATA 只能在锚点确有数据关系时使用。除非文章或所选段落明确提到，否则不得凭空加入实验室、XRD/衍射、显微镜、医疗设备、仪器屏幕或任何具体行业设备。\n"
            "例如，抽象的“可重复性失效”应画成同一流程反复运行后的分叉、偏离或对照关系，不可擅自画实验室设备。\n"
            "避免每个二级标题后都机械地放图；短章节可以没有图片，长章节最多两张，不得让图片相邻。"
            "同一数组内每个 visualRole 必须唯一：封面后每项都必须选择尚未使用的视觉角色。"
            "visualRole 只能是 COVER、SCENE、CONCEPT、DIAGRAM、COMPARISON、DETAIL、DATA、CLOSING；"
            "placement 只能是 COVER、AFTER_INTRO、AFTER_PARAGRAPH、SECTION_END。封面必须使用 COVER/COVER/HERO，"
            "下面的“段落视觉简报”由系统按最终 Markdown 解析，是唯一权威来源。AFTER_PARAGRAPH 只能从该目录对应标题的段落号中选择；"
            "若不能精确定位段落，必须改用 SECTION_END 且 anchorParagraph=0。不得自行为换行、列表或代码块计数，也不得编造标题。"
            "layout 只能是 HERO、WIDE、STANDARD、COMPACT。stockQuery 只用于 Pexels 照片检索，必须是 2-5 个小写 ASCII 英文词，"
            "只允许字母和单个空格，必须是 promptEn 中真实可见主体/场景的英文关键词；绝不能写抽象词、中文、标点、连字符或完整句子。"
            "DIAGRAM、DATA 等不适合图库照片的方案必须返回空字符串。\n"
            "只返回 JSON 数组，每项必须含 imageTitle、usageScene、promptZh、promptEn、visualRole、placement、"
            "anchorHeading、anchorParagraph、layout、stockQuery。\n"
            f"可用锚点目录：\n{image_anchor_catalog(content)}\n\n段落视觉简报：\n{image_visual_brief_catalog(content)}\n\n文章：{content}",
            validator=validate_plan_count,
            schema=(
                "JSON 数组；每项严格含 imageTitle、usageScene、promptZh、promptEn、visualRole、placement、"
                "anchorHeading、anchorParagraph、layout、stockQuery。visualRole、placement、layout 必须使用允许的枚举值；"
                "同一数组的 visualRole 不得重复；anchorParagraph 是非负整数；"
                f"stockQuery 仅可为空（DIAGRAM）或 2-5 个 ASCII 英文词；数组长度必须严格为 {count}。"
            ),
        )
        return prompts

    def review(self, article: str) -> QualityReviewResult:
        return self._json(
            "评审下列 Markdown 图文稿。重点检查：虚构二维码/扫码/附件/下载/课程资源、无来源用户数、百分比或亲测/专家/研究背书、"
            "健康健身类保证性效果承诺，以及不必要的 JSON/YAML/代码围栏。上述任一项必须写入 blockingIssues。\n"
            "仅返回 JSON 对象，严格 Schema：{\"score\": 0-100 的整数, \"passed\": boolean, \"summary\": string, "
            "\"revisionAction\": string|null, \"revisionReason\": string|null, \"blockingIssues\": string[], "
            "\"issues\": string[], \"suggestions\": string[]}。三个 Issues 数组只能包含简短字符串，不能包含对象。"
            "只有 blockingIssues 为空且达到交付标准时，passed 才能为 true。\n文章：\n"
            + article,
            validator=QualityReviewResult.model_validate,
            schema=(
                "JSON 对象，score 为 0-100 整数，passed 为布尔值，summary 为字符串；revisionAction/revisionReason 为字符串或 null；"
                "blockingIssues、issues、suggestions 必须都是字符串数组，绝不能是对象数组。"
            ),
        )  # type: ignore[return-value]


def create_article_agent(settings: Settings) -> ArticleAgent:
    return MockArticleAgent() if settings.app_mode.value == "mock" else OpenAICompatibleArticleAgent(settings)


def _parse_json_payload(raw: str) -> object:
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()
    # Some OpenAI-compatible models append a polite sentence after an
    # otherwise complete object/array.  Decode the first full JSON value rather
    # than needlessly spending a repair request, while still skipping prose
    # containing bracket-like characters before the actual payload.
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            payload, _end = decoder.raw_decode(cleaned[match.start() :])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response contains no JSON value")


def _title_options(payload: object) -> list[TitleOption]:
    if not isinstance(payload, list):
        raise ValueError("标题候选必须是 JSON 数组")
    return [TitleOption.model_validate(item) for item in payload]


def _outline_items(payload: object) -> list[OutlineItem]:
    if not isinstance(payload, list):
        raise ValueError("文章大纲必须是 JSON 数组")
    return [OutlineItem.model_validate(item) for item in payload]


def _image_prompt_options(payload: object) -> list[ImagePromptOption]:
    if not isinstance(payload, list):
        raise ValueError("配图方案必须是 JSON 数组")
    return [ImagePromptOption.model_validate(item) for item in payload]


_SEMANTIC_IMAGE_PLAN_FIELDS = {
    "visualRole": "visual_role",
    "placement": "placement",
    "anchorHeading": "anchor_heading",
    "anchorParagraph": "anchor_paragraph",
    "layout": "layout",
    "stockQuery": "stock_query",
}
_STOCK_QUERY_PATTERN = re.compile(r"[A-Za-z]+(?: [A-Za-z]+){1,4}")
_STOCK_QUERY_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_STOCK_QUERY_FALLBACKS = {
    ImageVisualRole.COVER: "editorial article cover",
    ImageVisualRole.SCENE: "professional lifestyle scene",
    ImageVisualRole.CONCEPT: "abstract editorial concept",
    ImageVisualRole.COMPARISON: "professional comparison scene",
    ImageVisualRole.DETAIL: "professional detail closeup",
    ImageVisualRole.DATA: "data visualization screen",
    ImageVisualRole.CLOSING: "calm editorial scene",
}

_UNSUPPORTED_SCENE_DETAILS = {
    "XRD/衍射设备": ("xrd", "x-ray diffraction", "x ray diffraction", "x射线衍射", "x 射线衍射"),
    "实验室": ("laboratory", "实验室"),
    "显微镜": ("microscope", "显微镜"),
    "离心机": ("centrifuge", "离心机"),
    "医疗影像设备": ("mri", "ct scanner", "medical scanner", "核磁", "断层扫描"),
}


def _generated_image_prompt_options(payload: object) -> list[ImagePromptOption]:
    """Accept full semantic LLM plans while leaving historical stored JSON compatible."""

    if not isinstance(payload, list):
        raise ValueError("配图方案必须是 JSON 数组")
    plans: list[ImagePromptOption] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("配图方案的每一项必须是 JSON 对象")
        missing = [
            public_name
            for public_name, python_name in _SEMANTIC_IMAGE_PLAN_FIELDS.items()
            if public_name not in item and python_name not in item
        ]
        if missing:
            raise ValueError(f"配图方案缺少语义字段：{', '.join(missing)}")
        normalized_item = dict(item)
        for field in ("stockQuery", "stock_query"):
            if field in normalized_item and not isinstance(normalized_item[field], str):
                normalized_item[field] = ""
        option = _normalize_stock_query(ImagePromptOption.model_validate(normalized_item))
        _validate_generated_image_plan(option)
        plans.append(option)
    return plans


def _normalize_stock_query(option: ImagePromptOption) -> ImagePromptOption:
    """Make an LLM's stock-search hint safe without discarding its visual plan.

    ``stockQuery`` is auxiliary metadata, not a reason to reject a complete
    placement/visual-role plan.  Real providers often return Chinese text,
    punctuation, or an overlong phrase despite the JSON prompt.  Prefer the
    model's English visual prompt, then use a role-specific neutral fallback.
    """

    current = option.stock_query.strip()
    if _STOCK_QUERY_PATTERN.fullmatch(current):
        return option.model_copy(update={"stock_query": current})

    words: list[str] = []
    for source in (current, option.prompt_en):
        for word in _STOCK_QUERY_WORD_PATTERN.findall(source.casefold()):
            if word not in words:
                words.append(word)
    if len(words) >= 2:
        return option.model_copy(update={"stock_query": " ".join(words[:5])})
    if option.visual_role in {ImageVisualRole.DIAGRAM, ImageVisualRole.DATA}:
        return option.model_copy(update={"stock_query": ""})
    return option.model_copy(update={"stock_query": _STOCK_QUERY_FALLBACKS[option.visual_role]})


def _validate_generated_image_plan(option: ImagePromptOption) -> None:
    if not option.stock_query:
        if option.visual_role not in {ImageVisualRole.DIAGRAM, ImageVisualRole.DATA}:
            raise ValueError("除 DIAGRAM 和 DATA 外的配图必须提供 2-5 个英文词的 stockQuery")
    elif not _STOCK_QUERY_PATTERN.fullmatch(option.stock_query.strip()):
        raise ValueError("stockQuery 必须是 2-5 个 ASCII 英文词")
    if option.visual_role is ImageVisualRole.COVER:
        if (
            option.placement is not ImagePlacement.COVER
            or option.anchor_heading
            or option.anchor_paragraph != 0
            or option.layout is not ImageLayout.HERO
        ):
            raise ValueError("封面必须使用 COVER 位置、空锚点、段落 0 和 HERO 布局")
        return
    if option.placement is ImagePlacement.COVER:
        raise ValueError("非封面图片不能使用 COVER 位置")
    if option.placement is ImagePlacement.AFTER_PARAGRAPH:
        if not option.anchor_heading or option.anchor_paragraph < 1:
            raise ValueError("AFTER_PARAGRAPH 必须提供章节锚点和从 1 开始的段落序号")
        return
    if option.placement is ImagePlacement.SECTION_END:
        if not option.anchor_heading or option.anchor_paragraph != 0:
            raise ValueError("SECTION_END 必须提供章节锚点，且段落序号为 0")
        return
    if option.anchor_heading or option.anchor_paragraph != 0:
        raise ValueError("AFTER_INTRO 不应包含章节或段落锚点")


def _validate_visual_grounding(plans: list[ImagePromptOption], article_context: str) -> None:
    """Reject only clear domain inventions before they become paid images.

    This is intentionally narrow: it does not try to judge whether a visual
    metaphor is aesthetically good.  It prevents the recurrent failure mode in
    which an abstract claim such as reproducibility is turned into unrelated
    XRD or laboratory equipment that the article never mentioned.
    """

    source = article_context.casefold()
    for plan in plans:
        visual_text = " ".join((plan.image_title, plan.usage_scene, plan.prompt_zh, plan.prompt_en)).casefold()
        for label, aliases in _UNSUPPORTED_SCENE_DETAILS.items():
            if any(alias in visual_text for alias in aliases) and not any(alias in source for alias in aliases):
                raise ValueError(f"配图不可凭空加入正文未出现的具体场景或设备：{label}")


def _json_repair_prompt(raw: str, schema: str, *, violation: str = "") -> str:
    header = (
        "将以下模型输出修复为可通过严格 Schema 校验的 JSON；只输出 JSON，不要解释。\n"
        f"目标 Schema：{schema}\n"
    )
    return "".join(
        (
            header,
            f"必须修复的问题：{violation}\n" if violation else "",
            "若问题列表中原本是对象，请提取其含义并改成简短字符串。\n"
            f"原始输出：\n{raw}",
        )
    )


def _factuality_contract() -> str:
    return (
        "事实与资源规则：不得虚构或暗示用户未提供的二维码、扫码入口、附件、下载链接、资料包、课程、动作图解库、"
        "真实案例、用户数、体验者、百分比、比例、亲测、专家背书或研究结论。没有可核验来源时，使用中性、审慎的表述，不编造数字。"
    )


def _content_prompt(topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str:
    technical_code_allowed = _is_clearly_technical(topic, title, style)
    code_rule = (
        "仅在解释该技术主题所必需时，才使用带明确语言名的围栏代码块；JSON/YAML 只在选题明确要求其格式时使用。"
        if technical_code_allowed
        else "不得输出围栏代码块、JSON 代码块或 YAML；表格必须使用普通 Markdown 表格。"
    )
    return (
        f"按 {style} 风格写 Markdown 文章。标题：{title}；选题：{topic}；"
        f"大纲：{json.dumps([item.model_dump() for item in outline], ensure_ascii=False)}；篇幅：{length}。\n"
        f"{_factuality_contract()}\n"
        "输出要求：全文只能有唯一的一级标题（#），并使用二级标题（##）组织各部分；"
        f"{code_rule}；不得输出裸 HTML；只输出可直接发布的 Markdown 正文。"
    )


def _content_repair_prompt(
    topic: str,
    title: str,
    outline: list[OutlineItem],
    style: str,
    length: str,
    draft: str,
    violations: list[str],
) -> str:
    return (
        "重写下列 Markdown 正文，使其可以直接发布。保留主题、标题和可验证的通用建议，"
        "但必须删除或改写所有违规内容；不要解释修改过程。\n"
        f"违规清单：{json.dumps(violations, ensure_ascii=False)}\n"
        f"{_content_prompt(topic, title, outline, style, length)}\n"
        f"待修订正文：\n{draft}"
    )


def _is_clearly_technical(topic: str, title: str, style: str) -> bool:
    if style.strip().upper() != "TECH":
        return False
    return bool(
        re.search(
            r"代码|编程|程序|python|java|javascript|typescript|fastapi|django|flask|api|接口|sql|算法|"
            r"数据结构|前端|后端|工程|json|yaml|数据库",
            f"{topic}\n{title}",
            re.IGNORECASE,
        )
    )


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions" if normalized.endswith("/v1") else "/v1/chat/completions"
    return normalized + suffix

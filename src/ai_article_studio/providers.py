"""Image providers, local asset persistence, and deterministic fallback routing."""

from __future__ import annotations

import base64
import inspect
import math
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from .config import Settings
from .contracts import ImageProviderClient
from .domain import (
    AdapterAuthenticationError,
    AdapterConfigurationError,
    AdapterResponseError,
    AdapterTransportError,
    WorkflowValidationError,
    safe_adapter_error_message,
)
from .safe_fetch import fetch_trusted_image
from .schemas import ImagePromptOption, ImageProvider, ImageResultOption, ImageVisualRole

MIN_IMAGE_COUNT = 3
MAX_IMAGE_COUNT = 8
CHARS_PER_IMAGE = 900

# These prompts describe an illustration, diagram, or UI render rather than a stock photo.
# AUTO deliberately excludes Google: it remains an explicit, opt-in provider only.
TECHNICAL_ILLUSTRATION_CJK_KEYWORDS = (
    "图解",
    "示意图",
    "信息图",
    "架构",
    "流程",
    "界面",
    "代码",
    "技术图",
)
TECHNICAL_ILLUSTRATION_ENGLISH_KEYWORDS = (
    "diagram",
    "infographic",
    "architecture",
    "flowchart",
    "flow diagram",
    "interface",
    "wireframe",
    "dashboard",
    "ui",
    "ux",
    "code",
)
STOCK_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "article",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


class MissingProvider:
    """Produces a safe error when a real integration lacks credentials."""

    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption:
        raise AdapterConfigurationError(f"{self.name} API key is not configured")


def calculate_image_count(content: str) -> int:
    """Use article structure or length, then clamp to the supported UI range."""

    heading_count = len(re.findall(r"(?m)^##\s+.+$", content or ""))
    compact_length = len(re.sub(r"\s+", "", content or ""))
    by_length = math.ceil(compact_length / CHARS_PER_IMAGE) if compact_length else 0
    return max(MIN_IMAGE_COUNT, min(MAX_IMAGE_COUNT, max(heading_count + 1, by_length + 1)))


def fallback_image(prompt: ImagePromptOption, index: int, *, reason: str = "") -> ImageResultOption:
    return ImageResultOption(
        image_title=prompt.image_title,
        usage_scene=prompt.usage_scene,
        prompt_zh=prompt.prompt_zh,
        prompt_en=prompt.prompt_en,
        visual_role=prompt.visual_role,
        placement=prompt.placement,
        anchor_heading=prompt.anchor_heading,
        anchor_paragraph=prompt.anchor_paragraph,
        anchor_excerpt=prompt.anchor_excerpt,
        layout=prompt.layout,
        stock_query=prompt.stock_query,
        image_url=f"https://picsum.photos/seed/ai-article-{index}/900/500",
        source="FALLBACK",
        author="AI Article Studio",
        fallback_reason=reason,
    )


class ImageGenerationService:
    """Routes explicit provider requests and owns the AUTO fallback contract."""

    def __init__(
        self,
        providers: Mapping[ImageProvider, ImageProviderClient],
        fallback_provider: ImageProviderClient | None = None,
    ) -> None:
        self.providers = providers
        self.fallback_provider = fallback_provider

    def generate(
        self,
        prompts: Sequence[ImagePromptOption],
        provider: ImageProvider,
        *,
        task_id: str | None = None,
    ) -> list[ImageResultOption]:
        return [
            self.generate_one(prompt, provider, index, task_id=task_id)
            for index, prompt in enumerate(prompts)
        ]

    def generate_one(
        self,
        prompt: ImagePromptOption,
        provider: ImageProvider,
        index: int,
        *,
        task_id: str | None = None,
    ) -> ImageResultOption:
        if provider is ImageProvider.AUTO:
            technical_illustration = _is_technical_illustration(prompt)
            for auto_provider in _auto_provider_order(prompt):
                try:
                    return self._generate(self.providers[auto_provider], prompt, index, task_id)
                except Exception:
                    continue
            fallback_reason = (
                "生成式图解暂不可用，已使用中性占位图，避免以无关图库照片替代。"
                if technical_illustration
                else "自动图片来源暂不可用，已使用中性占位图。"
            )
            if self.fallback_provider is not None:
                local = self._generate(self.fallback_provider, prompt, index, task_id)
                return local.model_copy(
                    update={
                        "source": "FALLBACK",
                        "author": "AI Article Studio",
                        "fallback_reason": fallback_reason,
                    }
                )
            return fallback_image(prompt, index, reason=fallback_reason)
        return self._generate(self.providers[provider], prompt, index, task_id)

    @staticmethod
    def _generate(provider: ImageProviderClient, prompt: ImagePromptOption, index: int, task_id: str | None) -> ImageResultOption:
        try:
            if task_id is None or not _supports_task_id(provider.generate):
                return provider.generate(prompt, index)
            return provider.generate(prompt, index, task_id=task_id)
        except httpx.HTTPError as error:
            raise AdapterTransportError(safe_adapter_error_message(error, "图片服务暂不可用，请稍后重试。")) from error


class MockImageProvider:
    """Offline provider which creates a deterministic, local SVG image asset."""

    def __init__(self, upload_dir: Path) -> None:
        self.upload_dir = upload_dir

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption:
        target_dir, url_prefix = _task_upload_location(self.upload_dir, task_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"mock-{uuid.uuid4().hex}.svg"
        file_path = target_dir / file_name
        label = _xml_escape(prompt.image_title[:80])
        file_path.write_text(
            (
                '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500">'
                '<rect width="100%" height="100%" fill="#172554"/>'
                '<circle cx="750" cy="120" r="110" fill="#38bdf8" opacity=".75"/>'
                '<text x="60" y="250" fill="white" font-size="42" '
                'font-family="Arial, sans-serif">'
                f"{label}</text></svg>"
            ),
            encoding="utf-8",
        )
        return ImageResultOption(
            image_title=prompt.image_title,
            usage_scene=prompt.usage_scene,
            prompt_zh=prompt.prompt_zh,
            prompt_en=prompt.prompt_en,
            visual_role=prompt.visual_role,
            placement=prompt.placement,
            anchor_heading=prompt.anchor_heading,
            anchor_paragraph=prompt.anchor_paragraph,
            anchor_excerpt=prompt.anchor_excerpt,
            layout=prompt.layout,
            stock_query=prompt.stock_query,
            image_url=f"{url_prefix}/{file_name}",
            source="MOCK",
            author="AI Article Studio Mock",
        )


class PexelsImageProvider:
    def __init__(self, api_key: str, client: httpx.Client | None = None, *, timeout_seconds: float = 30) -> None:
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption:
        query = _resolve_query(prompt)
        try:
            response = self.client.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": self.api_key},
                params={"query": query, "per_page": 10, "orientation": "landscape"},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AdapterTransportError("Pexels 图片服务暂不可用，请稍后重试。") from error
        try:
            photos = response.json().get("photos", [])
        except (AttributeError, TypeError, ValueError) as error:
            raise AdapterResponseError("Pexels 返回的图片数据不合法") from error
        photo = next((candidate for candidate in photos if _matches_stock_query(candidate, query)), None)
        if not photo:
            raise AdapterResponseError("Pexels returned no photo matching stock query")
        src = photo.get("src", {})
        image_url = src.get("landscape") or src.get("large") or src.get("original")
        if not image_url:
            raise AdapterResponseError("Pexels photo has no usable image URL")
        return ImageResultOption(
            image_title=prompt.image_title,
            usage_scene=prompt.usage_scene,
            prompt_zh=prompt.prompt_zh,
            prompt_en=prompt.prompt_en,
            visual_role=prompt.visual_role,
            placement=prompt.placement,
            anchor_heading=prompt.anchor_heading,
            anchor_paragraph=prompt.anchor_paragraph,
            anchor_excerpt=prompt.anchor_excerpt,
            layout=prompt.layout,
            stock_query=prompt.stock_query,
            image_url=image_url,
            source="PEXELS",
            source_url=photo.get("url", ""),
            author=photo.get("photographer", ""),
            author_url=photo.get("photographer_url", ""),
        )


class SiliconFlowImageProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        upload_dir: Path,
        client: httpx.Client | None = None,
        *,
        image_size: str = "1024x1024",
        batch_size: int = 1,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        cfg: float = 4.0,
        negative_prompt: str = "",
        request_timeout_seconds: float = 90,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.upload_dir = upload_dir
        self.client = client or httpx.Client(timeout=request_timeout_seconds, follow_redirects=True)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.cfg = cfg
        self.negative_prompt = negative_prompt

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption:
        model_name = self.model.casefold()
        if model_name.startswith("qwen/qwen-image-edit"):
            raise WorkflowValidationError(
                "SiliconFlow 图像编辑模型需要提供源图片；当前文章配图仅支持文生图，请改用 Kwai-Kolors/Kolors 或 Qwen/Qwen-Image。"
            )
        payload = {
            "model": self.model,
            "prompt": _prompt_text(prompt),
        }
        if model_name == "kwai-kolors/kolors":
            payload.update(
                {
                    "image_size": self.image_size,
                    "batch_size": self.batch_size,
                    "num_inference_steps": self.num_inference_steps,
                    "guidance_scale": self.guidance_scale,
                }
            )
        elif model_name.startswith("qwen/qwen-image"):
            payload.update(
                {
                    "image_size": self.image_size,
                    "num_inference_steps": self.num_inference_steps,
                    "cfg": self.cfg,
                }
            )
        else:
            payload["image_size"] = self.image_size
        if self.negative_prompt.strip():
            payload["negative_prompt"] = self.negative_prompt.strip()
        response = self._post_generation(payload)
        if response.status_code == 401:
            raise AdapterAuthenticationError(
                "SiliconFlow 鉴权失败（401）：请检查 AI_ARTICLE_STUDIO_SILICONFLOW_API_KEY。"
                "百炼 / DashScope 的 Key 不能用于 SiliconFlow 图片接口。"
            )
        if response.status_code == 403:
            raise AdapterAuthenticationError("SiliconFlow 无权调用当前图片模型（403）：请检查账号权限、余额和模型可用性。")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise AdapterTransportError("SiliconFlow 图片服务暂不可用，请稍后重试。") from error
        try:
            image_url = _siliconflow_image_url(response.json())
        except (TypeError, ValueError) as error:
            raise AdapterResponseError("SiliconFlow 响应中没有可下载的图片地址") from error
        if not image_url:
            raise AdapterResponseError("SiliconFlow 响应中没有可下载的图片地址")
        local_url = persist_remote_image(image_url, self.upload_dir, self.client, task_id=task_id)
        return _generated_result(prompt, local_url, "SILICONFLOW")

    def _post_generation(self, payload: dict[str, object]) -> httpx.Response:
        """Retry only server-confirmed, pre-generation transient rejections.

        We intentionally do not retry transport/read timeouts: a remote model
        may have accepted the image request already, and replaying it could
        create duplicate images and charges.  429/503/504 are explicit retry
        responses, so one bounded retry is both safe and helpful.
        """

        for attempt in range(2):
            try:
                response = self.client.post(
                    f"{self.base_url}/images/generations",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
            except httpx.HTTPError as error:
                raise AdapterTransportError("SiliconFlow 图片服务暂不可用，请稍后重试。") from error
            if response.status_code not in {429, 503, 504} or attempt == 1:
                return response
            time.sleep(_retry_after_seconds(response))
        raise AdapterTransportError("SiliconFlow 图片服务暂不可用，请稍后重试。")


class GoogleAIImageProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        upload_dir: Path,
        client: httpx.Client | None = None,
        *,
        request_timeout_seconds: float = 90,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.upload_dir = upload_dir
        self.client = client or httpx.Client(timeout=request_timeout_seconds)

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption:
        try:
            response = self.client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json={"contents": [{"parts": [{"text": _prompt_text(prompt)}]}]},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise AdapterTransportError("Google AI 图片服务暂不可用，请稍后重试。") from error
        try:
            parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise AdapterResponseError("Google AI response has no inline image") from error
        inline = next((part.get("inlineData") for part in parts if part.get("inlineData")), None)
        if not inline or not inline.get("data"):
            raise AdapterResponseError("Google AI response has no inline image")
        suffix = _extension_for_content_type(inline.get("mimeType", "image/png"))
        local_url = persist_bytes(base64.b64decode(inline["data"]), self.upload_dir, suffix, task_id=task_id)
        return _generated_result(prompt, local_url, "GOOGLE_AI")


def create_image_service(settings: Settings) -> ImageGenerationService:
    """Create real adapters in REAL mode and a credential-free MOCK adapter otherwise."""

    if settings.app_mode.value == "mock":
        mock = MockImageProvider(settings.upload_dir)
        return ImageGenerationService({provider: mock for provider in ImageProvider}, fallback_provider=mock)
    pexels_key = _secret_value(settings.pexels_api_key)
    siliconflow_key = _secret_value(settings.siliconflow_api_key)
    google_key = _secret_value(settings.google_ai_api_key)
    fallback = MockImageProvider(settings.upload_dir)
    return ImageGenerationService(
        {
            ImageProvider.PEXELS: (
                PexelsImageProvider(pexels_key, timeout_seconds=settings.request_timeout_seconds)
                if pexels_key
                else MissingProvider("Pexels")
            ),
            ImageProvider.SILICONFLOW: (
                SiliconFlowImageProvider(
                    siliconflow_key,
                    settings.siliconflow_base_url,
                    settings.siliconflow_image_model,
                    settings.upload_dir,
                    image_size=settings.siliconflow_image_size,
                    batch_size=settings.siliconflow_batch_size,
                    num_inference_steps=settings.siliconflow_num_inference_steps,
                    guidance_scale=settings.siliconflow_guidance_scale,
                    cfg=settings.siliconflow_cfg,
                    negative_prompt=settings.siliconflow_negative_prompt,
                    request_timeout_seconds=settings.request_timeout_seconds,
                )
                if siliconflow_key
                else MissingProvider("SiliconFlow")
            ),
            ImageProvider.GOOGLE_AI: (
                GoogleAIImageProvider(
                    google_key,
                    settings.google_image_model,
                    settings.upload_dir,
                    request_timeout_seconds=settings.request_timeout_seconds,
                )
                if google_key
                else MissingProvider("Google AI")
            ),
            ImageProvider.MOCK: fallback,
        },
        fallback_provider=fallback,
    )


def persist_remote_image(url: str, upload_dir: Path, client: httpx.Client, *, task_id: str | None = None) -> str:
    """Persist only images served by an approved image/CDN domain.

    Generation and CDN delivery are separate operations.  Retrying a failed
    download once is safe (it never re-submits the billed generation request)
    and absorbs short-lived 503/504 or connection failures.
    """

    for attempt in range(2):
        try:
            image = fetch_trusted_image(client, url)
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) and error.response else None
            if attempt or status not in {None, 429, 500, 502, 503, 504}:
                raise AdapterTransportError("图片资源下载服务暂不可用，请稍后重试。") from error
            time.sleep(0.2)
    return persist_bytes(image.content, upload_dir, _extension_for_content_type(image.content_type), task_id=task_id)


def persist_bytes(content: bytes, upload_dir: Path, suffix: str, *, task_id: str | None = None) -> str:
    target_dir, url_prefix = _task_upload_location(upload_dir, task_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    (target_dir / name).write_bytes(content)
    return f"{url_prefix}/{name}"


def _task_upload_location(upload_dir: Path, task_id: str | None) -> tuple[Path, str]:
    root = upload_dir.resolve()
    if task_id is None:
        return root, "/uploads/images"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task_id):
        raise WorkflowValidationError("任务图片目录不合法")
    target = (root / task_id).resolve()
    if not target.is_relative_to(root):
        raise WorkflowValidationError("任务图片目录不合法")
    return target, f"/uploads/images/{task_id}"


def _supports_task_id(operation: object) -> bool:
    """Keep third-party/custom providers using the pre-task-directory signature working."""

    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "task_id" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _resolve_query(prompt: ImagePromptOption) -> str:
    # Pexels ranks English natural-language search terms substantially more
    # reliably than a short Chinese card title.  This only affects the photo
    # route: diagrams never silently fall back to a stock photo in AUTO mode.
    return prompt.stock_query.strip() or prompt.prompt_en[:120] or "editorial workspace"


def _auto_provider_order(prompt: ImagePromptOption) -> tuple[ImageProvider, ...]:
    """Select the stable AUTO order without ever adding Google to the fallback chain."""

    if _is_technical_illustration(prompt):
        # A generic photo is misleading when the requested asset is an
        # architecture diagram, flowchart, UI render, or code illustration.
        # Preserve semantic correctness over visual variety and fall back to a
        # clearly-labelled neutral placeholder if generation is unavailable.
        return (ImageProvider.SILICONFLOW,)
    return (ImageProvider.PEXELS, ImageProvider.SILICONFLOW)


def _is_technical_illustration(prompt: ImagePromptOption) -> bool:
    if prompt.visual_role == ImageVisualRole.SCENE:
        return False
    if prompt.visual_role in {ImageVisualRole.DIAGRAM, ImageVisualRole.DATA, ImageVisualRole.COMPARISON}:
        return True
    # A server-grounded concept illustration has a precise claim to depict;
    # routing it to a stock-photo search would usually lose that relationship.
    # Keep legacy/manual plans without evidence on their previous route.
    if prompt.visual_role is ImageVisualRole.CONCEPT and prompt.anchor_excerpt.strip():
        return True
    prompt_text = " ".join((prompt.image_title, prompt.usage_scene, prompt.prompt_zh, prompt.prompt_en)).casefold()
    has_cjk_keyword = any(keyword in prompt_text for keyword in TECHNICAL_ILLUSTRATION_CJK_KEYWORDS)
    has_english_keyword = any(
        _matches_english_keyword(prompt_text, keyword) for keyword in TECHNICAL_ILLUSTRATION_ENGLISH_KEYWORDS
    )
    return has_cjk_keyword or has_english_keyword


def _matches_english_keyword(text: str, keyword: str) -> bool:
    """Allow English terms beside CJK text but not inside unrelated English words."""

    return bool(re.search(f"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text))


def _matches_stock_query(photo: object, query: str) -> bool:
    if not isinstance(photo, dict):
        return False
    query_terms = _meaningful_stock_terms(query)
    if not query_terms:
        return False
    candidate_text = " ".join(
        value
        for key in ("alt", "description")
        if isinstance((value := photo.get(key)), str)
    )
    required_matches = 2 if len(query_terms) >= 2 else 1
    return len(query_terms & _meaningful_stock_terms(candidate_text)) >= required_matches


def _meaningful_stock_terms(value: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z]+(?:-[a-z]+)?", value.casefold())
        if term not in STOCK_QUERY_STOPWORDS
    }


def _prompt_text(prompt: ImagePromptOption) -> str:
    visual_spec = (prompt.prompt_en or prompt.prompt_zh or prompt.image_title).strip()
    evidence = _compact_generation_fragment(prompt.anchor_excerpt, 480)
    if not evidence:
        # Direct provider tests and legacy records have no derived evidence.
        # Preserve the original payload for those intentional low-level calls.
        return visual_spec

    chinese_spec = _compact_generation_fragment(prompt.prompt_zh, 420)
    chinese_line = ""
    if chinese_spec and chinese_spec != visual_spec:
        chinese_line = f"\nChinese visual constraints: {chinese_spec}"
    return (
        "Create one editorial image that directly explains the article claim below.\n"
        f"Article claim: {evidence}\n"
        f"Reader takeaway: {_compact_generation_fragment(prompt.usage_scene, 240)}\n"
        f"Required visual composition: {_compact_generation_fragment(visual_spec, 680)}"
        f"{chinese_line}\n"
        "Use only people, objects, actions, and settings supported by the article claim or visual specification. "
        "Do not invent an unrelated domain, laboratory, device, or data. No text, logo, watermark, QR code, or UI labels."
    )


def _compact_generation_fragment(value: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "…"


def _generated_result(prompt: ImagePromptOption, image_url: str, source: str) -> ImageResultOption:
    return ImageResultOption(
        image_title=prompt.image_title,
        usage_scene=prompt.usage_scene,
        prompt_zh=prompt.prompt_zh,
        prompt_en=prompt.prompt_en,
        visual_role=prompt.visual_role,
        placement=prompt.placement,
        anchor_heading=prompt.anchor_heading,
        anchor_paragraph=prompt.anchor_paragraph,
        anchor_excerpt=prompt.anchor_excerpt,
        layout=prompt.layout,
        stock_query=prompt.stock_query,
        image_url=image_url,
        source=source,
        author=f"{source} generated",
    )


def _siliconflow_image_url(payload: object) -> str:
    """Read SiliconFlow's documented ``images`` payload, accepting legacy ``data`` too."""

    if not isinstance(payload, dict):
        return ""
    for field in ("images", "data"):
        images = payload.get(field)
        if isinstance(images, list) and images and isinstance(images[0], dict):
            image_url = images[0].get("url")
            if isinstance(image_url, str) and image_url:
                return image_url
    return ""


def _extension_for_content_type(content_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/avif": ".avif",
        "image/gif": ".gif",
    }.get(content_type, ".png")


def _retry_after_seconds(response: httpx.Response) -> float:
    """Respect a provider's short retry hint without parking the local queue."""

    try:
        return max(0.0, min(float(response.headers.get("retry-after", "0.5")), 5.0))
    except ValueError:
        return 0.5


def _secret_value(secret: object) -> str:
    return secret.get_secret_value() if secret else ""  # type: ignore[union-attr]


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

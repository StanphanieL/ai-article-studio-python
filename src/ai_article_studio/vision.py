"""Optional OpenAI-compatible visual checks for generated article images.

The reviewer deliberately accepts only images the existing exporter can load:
task-local uploads or remote URLs approved by the shared allowlist.  It stores a
small, normalized result and never retains provider payloads or credentials.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
from pydantic import ValidationError

from .config import Settings
from .exports import ArticleExporter
from .schemas import (
    ImagePromptOption,
    ImageResultOption,
    ImageVisualReview,
    ImageVisualReviewStatus,
)

_UNAVAILABLE_MESSAGE = "视觉质检结果不可用，可按需重试。"
_VALID_PROVIDER_STATUSES = {
    ImageVisualReviewStatus.PASSED,
    ImageVisualReviewStatus.NEEDS_RETRY,
}


class VisionReviewer:
    """Use a configured vision model to review one generated image.

    A blank ``vision_model`` is an explicit no-op.  This keeps normal image
    generation free of extra provider calls and image egress.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.request_timeout_seconds)
        self.exporter = ArticleExporter(settings.upload_dir, client=self.client)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.vision_model.strip())

    def review(self, image: ImageResultOption, prompt: ImagePromptOption) -> ImageVisualReview:
        """Return a safe, non-blocking quality signal for ``image``.

        Any setup, download, transport, schema, or provider failure becomes
        ``UNAVAILABLE``.  Callers can persist the generated image unchanged.
        """

        if not self.enabled:
            return ImageVisualReview(status=ImageVisualReviewStatus.SKIPPED)

        api_key = self.settings.vision_api_key or self.settings.openai_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            return self._unavailable()

        try:
            binary = self.exporter.load_image(image.image_url)
            if not binary.content_type.startswith("image/"):
                return self._unavailable()
            data_uri = f"data:{binary.content_type};base64,{base64.b64encode(binary.content).decode('ascii')}"
            response = self.client.post(
                _chat_completions_url(self.settings.vision_base_url or self.settings.openai_base_url),
                headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
                json={
                    "model": self.settings.vision_model.strip(),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _review_prompt(image, prompt)},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 400,
                },
            )
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content")
            return _parse_review(content)
        except (
            httpx.HTTPError,
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValidationError,
        ):
            return self._unavailable()

    @staticmethod
    def _unavailable() -> ImageVisualReview:
        return ImageVisualReview(
            status=ImageVisualReviewStatus.UNAVAILABLE,
            issues=[_UNAVAILABLE_MESSAGE],
        )


def _chat_completions_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _review_prompt(image: ImageResultOption, prompt: ImagePromptOption) -> str:
    return (
        "评估这张文章配图是否与规划一致、主体是否清晰、是否存在明显文字水印/无关内容或严重变形。"
        "不要评价文章事实，不要输出图片内容原文。只返回 JSON 对象："
        '{"status":"PASSED"|"NEEDS_RETRY","score":0-100整数,"issues":["简短问题"]}。'
        "issues 只能是简短字符串数组；画面符合计划时返回 PASSED 和空数组。\n"
        f"图片标题：{image.image_title}\n使用位置：{prompt.usage_scene}\n视觉角色：{prompt.visual_role.value}\n"
        f"英文提示词：{prompt.prompt_en}"
    )


def _parse_review(raw: Any) -> ImageVisualReview:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("vision response has no content")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    payload = json.loads(cleaned.strip())
    parsed = ImageVisualReview.model_validate(payload)
    if parsed.status not in _VALID_PROVIDER_STATUSES:
        raise ValueError("vision response has an unsupported status")
    safe_issues = [issue.strip()[:300] for issue in parsed.issues if isinstance(issue, str) and issue.strip()][:8]
    return parsed.model_copy(update={"issues": safe_issues})

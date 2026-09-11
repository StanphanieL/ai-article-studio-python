"""Typed ports used by the workflow to call article and image adapters."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from .schemas import ImagePromptOption, ImageResultOption, OutlineItem, QualityReviewResult, TitleOption


@runtime_checkable
class ArticleAgent(Protocol):
    def titles(self, topic: str, style: str) -> list[TitleOption]: ...

    def outline(self, topic: str, title: str, style: str) -> list[OutlineItem]: ...

    def content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> str: ...

    def repair_content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str, draft: str, violations: list[str]) -> str: ...

    def stream_content(self, topic: str, title: str, outline: list[OutlineItem], style: str, length: str) -> Iterator[str]: ...

    def image_prompts(self, topic: str, title: str, content: str, style: str, count: int) -> list[ImagePromptOption]: ...

    def review(self, article: str) -> QualityReviewResult: ...


@runtime_checkable
class ImageProviderClient(Protocol):
    """A single, explicit image provider implementation."""

    def generate(self, prompt: ImagePromptOption, index: int, *, task_id: str | None = None) -> ImageResultOption: ...

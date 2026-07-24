"""Public request, response, and structured AI-output contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _camel_case(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(piece.title() for piece in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=_camel_case, populate_by_name=True)


class BaseResponse[T](ApiModel):
    code: int = 0
    message: str = "success"
    data: T | None = None


class WorkflowJobStatus(ApiModel):
    task_id: str
    status: str
    action: str | None = None
    execution_mode: str | None = None
    message: str = ""
    error_message: str | None = None
    started_at: datetime | None = None
    elapsed_seconds: int | None = None


class ImageProvider(StrEnum):
    AUTO = "AUTO"
    PEXELS = "PEXELS"
    SILICONFLOW = "SILICONFLOW"
    GOOGLE_AI = "GOOGLE_AI"
    MOCK = "MOCK"


class WorkflowRunMode(StrEnum):
    """How the currently persisted task is expected to continue after a retry."""

    STEP = "STEP"
    ONE_CLICK = "ONE_CLICK"


class ImageCountMode(StrEnum):
    AUTO = "AUTO"
    CUSTOM = "CUSTOM"


class ContentLengthLevel(StrEnum):
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"


class ImageVisualRole(StrEnum):
    COVER = "COVER"
    SCENE = "SCENE"
    CONCEPT = "CONCEPT"
    DIAGRAM = "DIAGRAM"
    COMPARISON = "COMPARISON"
    DETAIL = "DETAIL"
    DATA = "DATA"
    CLOSING = "CLOSING"


class ImagePlacement(StrEnum):
    COVER = "COVER"
    AFTER_INTRO = "AFTER_INTRO"
    AFTER_PARAGRAPH = "AFTER_PARAGRAPH"
    SECTION_END = "SECTION_END"


class ImageLayout(StrEnum):
    HERO = "HERO"
    WIDE = "WIDE"
    STANDARD = "STANDARD"
    COMPACT = "COMPACT"


class ImageVisualReviewStatus(StrEnum):
    SKIPPED = "SKIPPED"
    PASSED = "PASSED"
    NEEDS_RETRY = "NEEDS_RETRY"
    UNAVAILABLE = "UNAVAILABLE"


class ArticleModelConfig(ApiModel):
    text_model: str = "MiniMax-M2.5"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_completion_tokens: int = Field(default=8192, ge=1, le=65536)
    content_length_level: ContentLengthLevel = ContentLengthLevel.MEDIUM
    image_provider: ImageProvider = ImageProvider.AUTO
    image_count_mode: ImageCountMode = ImageCountMode.AUTO
    image_count: int | None = Field(default=None, ge=3, le=8)

    @model_validator(mode="after")
    def validate_image_count(self) -> ArticleModelConfig:
        if self.image_count_mode is ImageCountMode.CUSTOM and self.image_count is None:
            raise ValueError("image_count is required when image_count_mode is CUSTOM")
        return self


class TitleOption(ApiModel):
    title: str
    reason: str


class OutlineItem(ApiModel):
    heading: str
    description: str


class ImagePromptOption(ApiModel):
    image_title: str
    usage_scene: str
    prompt_zh: str
    prompt_en: str
    visual_role: ImageVisualRole = ImageVisualRole.CONCEPT
    placement: ImagePlacement = ImagePlacement.SECTION_END
    anchor_heading: str = ""
    anchor_paragraph: int = Field(default=0, ge=0)
    # Server-derived evidence for this image's placement anchor.  It is kept
    # out of the page editor: users arrange images by heading/paragraph while
    # the workflow always refreshes the actual source text from the article.
    anchor_excerpt: str = ""
    layout: ImageLayout = ImageLayout.STANDARD
    stock_query: str = ""


class ImageVisualReview(ApiModel):
    """Sanitized, non-blocking image quality signal from an optional vision model."""

    status: ImageVisualReviewStatus = ImageVisualReviewStatus.SKIPPED
    score: int | None = Field(default=None, ge=0, le=100)
    issues: list[str] = Field(default_factory=list)


class ImageResultOption(ImagePromptOption):
    image_url: str
    source: str
    source_url: str = ""
    author: str = ""
    author_url: str = ""
    fallback_reason: str = ""
    visual_review: ImageVisualReview = Field(default_factory=ImageVisualReview)


class ImagePlanUpdateRequest(ImagePromptOption):
    """Validated page-only payload for editing one persisted image plan."""

    model_config = ConfigDict(alias_generator=_camel_case, populate_by_name=True, extra="forbid")


class ImagePlanReorderRequest(ApiModel):
    """Validated page-only payload for arranging persisted image-plan cards."""

    model_config = ConfigDict(alias_generator=_camel_case, populate_by_name=True, extra="forbid")

    indices: list[int] = Field(min_length=1)


class QualityReviewResult(ApiModel):
    score: int = Field(ge=0, le=100)
    passed: bool
    summary: str
    revision_action: str | None = None
    revision_reason: str | None = None
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)


class ArticleCreateRequest(ApiModel):
    topic: str = Field(min_length=1, max_length=500)
    style: str = "TECH"
    # Optional so older API clients can keep relying on the environment defaults.
    config: ArticleModelConfig | None = None


class OneClickGenerateRequest(ArticleCreateRequest):
    pass


class ConfirmTitleRequest(ApiModel):
    task_id: str = Field(min_length=1)
    selected_title: str = Field(min_length=1)


class SaveTitleRequest(ConfirmTitleRequest):
    pass


class SaveOutlineRequest(ApiModel):
    task_id: str = Field(min_length=1)
    outline: list[OutlineItem] = Field(min_length=1)


class SaveContentRequest(ApiModel):
    task_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SaveModelConfigRequest(ApiModel):
    task_id: str = Field(min_length=1)
    style: str = Field(min_length=1)
    config: ArticleModelConfig

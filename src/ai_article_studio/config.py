"""Environment-backed application configuration."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(StrEnum):
    """Whether external AI services are contacted."""

    MOCK = "mock"
    REAL = "real"


class Settings(BaseSettings):
    """Settings intentionally keep all credentials on the server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_ARTICLE_STUDIO_",
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = "AI Article Studio"
    app_mode: RuntimeMode = RuntimeMode.MOCK
    database_url: str = Field(
        default="sqlite:///data/ai_article_studio.db",
        validation_alias=AliasChoices("AI_ARTICLE_STUDIO_DATABASE_URL", "DATABASE_URL"),
    )
    upload_dir: Path = Path("uploads/images")
    openai_base_url: str = "https://api.minimax.chat"
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_ARTICLE_STUDIO_OPENAI_API_KEY", "LLM_API_KEY"),
    )
    openai_model: str = "MiniMax-M2.5"
    # Visual review is opt-in because it incurs a second model call per image.
    # An empty model guarantees no image is loaded or sent anywhere.
    vision_model: str = ""
    vision_base_url: str = ""
    vision_api_key: SecretStr | None = None
    pexels_api_key: SecretStr | None = None
    siliconflow_api_key: SecretStr | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_image_model: str = "Kwai-Kolors/Kolors"
    siliconflow_image_size: str = "1024x1024"
    siliconflow_batch_size: int = Field(default=1, ge=1, le=4)
    siliconflow_num_inference_steps: int = Field(default=20, ge=1, le=100)
    siliconflow_guidance_scale: float = Field(default=7.5, ge=0, le=20)
    siliconflow_cfg: float = Field(default=4.0, ge=0.1, le=20)
    siliconflow_negative_prompt: str = "low quality, blurry, distorted, deformed, watermark, text artifacts"
    google_ai_api_key: SecretStr | None = None
    google_image_model: str = "gemini-2.5-flash-image"
    request_timeout_seconds: float = 90.0
    max_auto_review_rounds: int = 1


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings instance for the running application."""

    return Settings()

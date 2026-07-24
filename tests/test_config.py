from __future__ import annotations

from ai_article_studio.config import RuntimeMode, Settings


def test_settings_defaults_to_local_sqlite_and_mock_mode() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite:///data/ai_article_studio.db"
    assert settings.app_mode is RuntimeMode.MOCK


def test_settings_reads_prefixed_environment_values(monkeypatch) -> None:
    monkeypatch.setenv(
        "AI_ARTICLE_STUDIO_DATABASE_URL",
        "postgresql+psycopg://article:secret@localhost/articles",
    )
    monkeypatch.setenv("AI_ARTICLE_STUDIO_APP_MODE", "real")
    monkeypatch.setenv("AI_ARTICLE_STUDIO_OPENAI_API_KEY", "test-key")

    settings = Settings(_env_file=None)

    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.app_mode is RuntimeMode.REAL
    assert settings.openai_api_key.get_secret_value() == "test-key"


def test_settings_supports_standard_database_url_alias(monkeypatch) -> None:
    monkeypatch.delenv("AI_ARTICLE_STUDIO_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://article:secret@localhost/articles")

    settings = Settings(_env_file=None)

    assert settings.database_url.startswith("postgresql+psycopg://")


def test_settings_supports_legacy_llm_api_key_alias(monkeypatch) -> None:
    monkeypatch.delenv("AI_ARTICLE_STUDIO_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "legacy-key")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key.get_secret_value() == "legacy-key"


def test_siliconflow_defaults_follow_the_documented_image_api() -> None:
    settings = Settings(_env_file=None)

    assert settings.siliconflow_base_url == "https://api.siliconflow.cn/v1"
    assert settings.siliconflow_image_model == "Kwai-Kolors/Kolors"
    assert settings.siliconflow_image_size == "1024x1024"
    assert settings.siliconflow_cfg == 4.0


def test_vision_review_is_opt_in_and_can_use_dedicated_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AI_ARTICLE_STUDIO_VISION_MODEL", "qwen3-vl-plus")
    monkeypatch.setenv("AI_ARTICLE_STUDIO_VISION_BASE_URL", "https://vision.example.test/v1")
    monkeypatch.setenv("AI_ARTICLE_STUDIO_VISION_API_KEY", "vision-key")

    settings = Settings(_env_file=None)

    assert settings.vision_model == "qwen3-vl-plus"
    assert settings.vision_base_url == "https://vision.example.test/v1"
    assert settings.vision_api_key is not None
    assert settings.vision_api_key.get_secret_value() == "vision-key"

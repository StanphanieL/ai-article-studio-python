from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_article_studio.schemas import (
    ArticleModelConfig,
    ImageCountMode,
    ImageProvider,
)


def test_custom_image_count_requires_an_allowed_count() -> None:
    config = ArticleModelConfig(
        image_count_mode=ImageCountMode.CUSTOM,
        image_count=4,
    )

    assert config.image_provider is ImageProvider.AUTO
    assert config.image_count == 4


@pytest.mark.parametrize("image_count", [None, 2, 9])
def test_custom_image_count_rejects_missing_or_out_of_range_count(
    image_count: int | None,
) -> None:
    with pytest.raises(ValidationError, match="image_count"):
        ArticleModelConfig(
            image_count_mode=ImageCountMode.CUSTOM,
            image_count=image_count,
        )


def test_image_count_always_stays_within_the_supported_range() -> None:
    with pytest.raises(ValidationError, match="image_count"):
        ArticleModelConfig(
            image_count_mode=ImageCountMode.AUTO,
            image_count=9,
        )


def test_temperature_must_remain_within_the_openai_compatible_range() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        ArticleModelConfig(temperature=2.1)

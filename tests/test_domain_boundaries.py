from __future__ import annotations

from dataclasses import FrozenInstanceError

import httpx
import pytest

from ai_article_studio.config import Settings
from ai_article_studio.schemas import ImagePromptOption


def _prompt() -> ImagePromptOption:
    return ImagePromptOption(
        image_title="封面",
        usage_scene="文章封面",
        prompt_zh="科技感封面",
        prompt_en="technology cover",
    )


def test_workflow_operation_is_an_immutable_domain_value() -> None:
    from ai_article_studio.domain import WorkflowOperation

    operation = WorkflowOperation("GENERATE_IMAGE_RESULTS", provider="SILICONFLOW")

    assert operation.action == "GENERATE_IMAGE_RESULTS"
    assert operation.provider == "SILICONFLOW"
    with pytest.raises(FrozenInstanceError):
        operation.action = "REVIEW_ARTICLE"  # type: ignore[misc]


def test_llm_and_image_contracts_remain_available_from_legacy_modules() -> None:
    from ai_article_studio.contracts import ArticleAgent, ImageProviderClient
    from ai_article_studio.llm import ArticleAgent as LegacyArticleAgent
    from ai_article_studio.llm import MockArticleAgent
    from ai_article_studio.providers import ImageProviderClient as LegacyImageProviderClient
    from ai_article_studio.providers import MockImageProvider

    assert LegacyArticleAgent is ArticleAgent
    assert LegacyImageProviderClient is ImageProviderClient
    assert isinstance(MockArticleAgent(), ArticleAgent)
    assert isinstance(MockImageProvider.__new__(MockImageProvider), ImageProviderClient)


def test_llm_vendor_http_error_is_normalized_without_response_body_or_secret() -> None:
    from ai_article_studio.domain import AdapterTransportError
    from ai_article_studio.llm import OpenAICompatibleArticleAgent

    secret = "llm-secret-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=f'{{"error":"{secret}"}}', request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key=secret, _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(AdapterTransportError, match="LLM 服务暂不可用，请稍后重试。") as error:
        agent._chat("hello")

    assert secret not in str(error.value)


def test_llm_stream_vendor_http_error_is_normalized_without_response_body_or_secret() -> None:
    from ai_article_studio.domain import AdapterTransportError
    from ai_article_studio.llm import OpenAICompatibleArticleAgent

    secret = "llm-stream-secret-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text=f'{{"error":"{secret}"}}', request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key=secret, _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(AdapterTransportError, match="LLM 服务暂不可用，请稍后重试。") as error:
        list(agent.stream_content("topic", "title", [], "TECH", "SHORT"))

    assert secret not in str(error.value)


def test_image_vendor_http_error_is_normalized_without_response_body_or_secret() -> None:
    from ai_article_studio.domain import AdapterTransportError
    from ai_article_studio.providers import PexelsImageProvider

    secret = "image-secret-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text=f'{{"error":"{secret}"}}', request=request)

    provider = PexelsImageProvider(secret, httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(AdapterTransportError, match="Pexels 图片服务暂不可用，请稍后重试。") as error:
        provider.generate(_prompt(), 0)

    assert secret not in str(error.value)


def test_remote_image_download_error_is_normalized_without_response_body_or_secret(tmp_path) -> None:
    from ai_article_studio.domain import AdapterTransportError
    from ai_article_studio.providers import persist_remote_image

    secret = "download-secret-must-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=f'{{"error":"{secret}"}}', request=request)

    with pytest.raises(AdapterTransportError, match="图片资源下载服务暂不可用，请稍后重试。") as error:
        persist_remote_image(
            "https://images.pexels.com/generated.png",
            tmp_path,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )

    assert secret not in str(error.value)


def test_image_service_normalizes_http_errors_from_a_contract_implementation() -> None:
    from ai_article_studio.domain import AdapterTransportError
    from ai_article_studio.providers import ImageGenerationService
    from ai_article_studio.schemas import ImageProvider

    secret = "contract-secret-must-not-leak"

    class FailingProvider:
        def generate(self, prompt: ImagePromptOption, index: int):
            request = httpx.Request("POST", "https://provider.example.test/generate")
            response = httpx.Response(502, text=secret, request=request)
            raise httpx.HTTPStatusError("vendor failure", request=request, response=response)

    service = ImageGenerationService({ImageProvider.PEXELS: FailingProvider()})

    with pytest.raises(AdapterTransportError, match="vendor failure") as error:
        service.generate_one(_prompt(), ImageProvider.PEXELS, 0)

    assert secret not in str(error.value)


@pytest.mark.parametrize(
    "message",
    [
        "Bearer inline-secret",
        "API-Key inline-secret",
        "Authorization: API-Key inline-secret",
        "api_key=inline-secret",
        "https://provider.example.test/?token=inline-secret",
        '{"apiKey":"inline-secret"}',
    ],
)
def test_adapter_error_messages_redact_secrets_embedded_in_exception_text(message: str) -> None:
    from ai_article_studio.domain import safe_adapter_error_message

    safe_message = safe_adapter_error_message(RuntimeError(message), "fallback")

    assert "inline-secret" not in safe_message
    assert "[REDACTED]" in safe_message


def test_pexels_malformed_photo_shape_becomes_an_adapter_response_error() -> None:
    from ai_article_studio.domain import AdapterResponseError
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"photos": [{"alt": "technology cover", "src": "not-an-object"}]},
            request=request,
        )

    provider = PexelsImageProvider("test-key", httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(AdapterResponseError, match="Pexels 返回的图片数据不合法"):
        provider.generate(_prompt(), 0)


@pytest.mark.parametrize(
    "photos",
    [
        None,
        {"alt": "technology cover"},
        [{"alt": "technology cover", "src": {"landscape": 123}}],
    ],
)
def test_pexels_malformed_collection_or_url_becomes_an_adapter_response_error(photos: object) -> None:
    from ai_article_studio.domain import AdapterResponseError
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"photos": photos}, request=request)

    provider = PexelsImageProvider("test-key", httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(AdapterResponseError, match="Pexels 返回的图片数据不合法"):
        provider.generate(_prompt(), 0)


def test_google_malformed_inline_image_becomes_an_adapter_response_error(tmp_path) -> None:
    from ai_article_studio.domain import AdapterResponseError
    from ai_article_studio.providers import GoogleAIImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "%%%"}}]}}]},
            request=request,
        )

    provider = GoogleAIImageProvider(
        "test-key",
        "gemini-test",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(AdapterResponseError, match="Google AI response has no inline image"):
        provider.generate(_prompt(), 0)


def test_llm_unrepairable_json_becomes_a_response_error_with_legacy_validation_semantics() -> None:
    from ai_article_studio.domain import AdapterResponseError
    from ai_article_studio.llm import OpenAICompatibleArticleAgent

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "{broken"}}]}, request=request)

    agent = OpenAICompatibleArticleAgent(Settings(openai_api_key="test-key", _env_file=None))
    agent.client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(AdapterResponseError, match="LLM response contains no JSON value") as error:
        agent.titles("topic", "TECH")

    assert isinstance(error.value, ValueError)
    assert calls == 2

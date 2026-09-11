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

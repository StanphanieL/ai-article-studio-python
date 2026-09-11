from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest

from ai_article_studio.schemas import ImagePromptOption, ImageProvider, ImageResultOption


@dataclass
class StubProvider:
    name: str
    fail: bool = False
    calls: list[int] = field(default_factory=list)

    def generate(self, prompt: ImagePromptOption, index: int) -> ImageResultOption:
        self.calls.append(index)
        if self.fail:
            raise RuntimeError(f"{self.name} unavailable")
        return ImageResultOption(
            image_title=prompt.image_title,
            usage_scene=prompt.usage_scene,
            prompt_zh=prompt.prompt_zh,
            prompt_en=prompt.prompt_en,
            image_url=f"https://example.test/{self.name}/{index}.png",
            source=self.name,
        )


def prompt() -> ImagePromptOption:
    return ImagePromptOption(
        image_title="封面",
        usage_scene="文章封面",
        prompt_zh="科技感封面",
        prompt_en="technology cover",
    )


def test_auto_falls_back_from_pexels_to_siliconflow() -> None:
    from ai_article_studio.providers import ImageGenerationService

    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: StubProvider("PEXELS", fail=True),
            ImageProvider.SILICONFLOW: StubProvider("SILICONFLOW"),
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )

    result = service.generate_one(prompt(), ImageProvider.AUTO, 0)

    assert result.source == "SILICONFLOW"


def test_auto_prefers_siliconflow_for_a_technical_diagram() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    technical_prompt = prompt().model_copy(update={"usage_scene": "前端架构流程图", "prompt_en": "UI architecture flow diagram with code"})

    result = service.generate_one(technical_prompt, ImageProvider.AUTO, 0)

    assert result.source == "SILICONFLOW"
    assert siliconflow.calls == [0]
    assert pexels.calls == []


def test_auto_prefers_siliconflow_for_a_claim_comparison_not_suited_to_stock_photos() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    comparison_prompt = prompt().model_copy(
        update={
            "visual_role": "CONCEPT",
            "usage_scene": "对比重复流程的两种结果",
            "prompt_en": "two process paths",
            "anchor_excerpt": "同一流程重复执行后出现不同结果。",
        }
    )

    result = service.generate_one(comparison_prompt, ImageProvider.AUTO, 0)

    assert result.source == "SILICONFLOW"
    assert siliconflow.calls == [0]
    assert pexels.calls == []


def test_auto_technical_route_uses_safe_placeholder_without_stock_photo_when_generation_fails() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS", fail=True)
    siliconflow = StubProvider("SILICONFLOW", fail=True)
    google = StubProvider("GOOGLE_AI")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: google,
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    technical_prompt = prompt().model_copy(update={"usage_scene": "信息图", "prompt_en": "data infographic"})

    result = service.generate_one(technical_prompt, ImageProvider.AUTO, 2)

    assert result.source == "FALLBACK"
    assert siliconflow.calls == [2]
    assert pexels.calls == []
    assert google.calls == []


def test_auto_prefers_pexels_for_a_people_photo_scene() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    photo_prompt = prompt().model_copy(update={"usage_scene": "设计师人物工作场景照片", "prompt_en": "designer portrait photo"})

    result = service.generate_one(photo_prompt, ImageProvider.AUTO, 1)

    assert result.source == "PEXELS"
    assert pexels.calls == [1]
    assert siliconflow.calls == []


def test_auto_does_not_treat_ui_as_an_arbitrary_english_substring() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    photo_prompt = prompt().model_copy(update={"usage_scene": "人物照片", "prompt_en": "A guide to portrait photography"})

    result = service.generate_one(photo_prompt, ImageProvider.AUTO, 3)

    assert result.source == "PEXELS"
    assert pexels.calls == [3]
    assert siliconflow.calls == []


def test_auto_treats_ui_next_to_chinese_text_as_a_technical_keyword() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    ui_prompt = prompt().model_copy(update={"usage_scene": "UI设计展示", "prompt_en": "product screen"})

    result = service.generate_one(ui_prompt, ImageProvider.AUTO, 4)

    assert result.source == "SILICONFLOW"
    assert siliconflow.calls == [4]
    assert pexels.calls == []


def test_auto_uses_fallback_after_two_provider_failures() -> None:
    from ai_article_studio.providers import ImageGenerationService

    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: StubProvider("PEXELS", fail=True),
            ImageProvider.SILICONFLOW: StubProvider("SILICONFLOW", fail=True),
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )

    result = service.generate_one(prompt(), ImageProvider.AUTO, 2)

    assert result.source == "FALLBACK"
    assert result.image_url == "https://picsum.photos/seed/ai-article-2/900/500"


def test_auto_persists_a_local_fallback_when_all_automatic_suppliers_fail(tmp_path) -> None:
    from ai_article_studio.providers import ImageGenerationService, MockImageProvider

    pexels = StubProvider("PEXELS", fail=True)
    siliconflow = StubProvider("SILICONFLOW", fail=True)
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: MockImageProvider(tmp_path),
        },
        fallback_provider=MockImageProvider(tmp_path),
    )

    result = service.generate_one(prompt(), ImageProvider.AUTO, 0, task_id="task")

    assert pexels.calls == [0]
    assert siliconflow.calls == [0]
    assert result.source == "FALLBACK"
    assert result.image_url.startswith("/uploads/images/task/")
    assert result.fallback_reason


def test_trusted_image_download_allows_picsum_cdn_redirects() -> None:
    from ai_article_studio.safe_fetch import fetch_trusted_image

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "picsum.photos":
            return httpx.Response(302, headers={"location": "https://fastly.picsum.photos/seed/image.jpg"}, request=request)
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=b"image", request=request)

    image = fetch_trusted_image(
        httpx.Client(transport=httpx.MockTransport(handler)),
        "https://picsum.photos/seed/image/900/500",
    )

    assert image.content == b"image"


def test_trusted_image_download_accepts_a_verified_png_with_generic_cdn_content_type() -> None:
    from ai_article_studio.safe_fetch import fetch_trusted_image

    image = fetch_trusted_image(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "application/octet-stream"},
                    content=b"\x89PNG\r\n\x1a\nvalid-png-payload",
                    request=request,
                )
            )
        ),
        "https://delivery.eu2.bfl.ai/images/generated.png",
    )

    assert image.content_type == "image/png"
    assert image.content.startswith(b"\x89PNG")


def test_trusted_image_download_rejects_unverified_generic_binary_response() -> None:
    from ai_article_studio.safe_fetch import UnsafeImageUrlError, fetch_trusted_image

    with pytest.raises(UnsafeImageUrlError, match="Content-Type: application/octet-stream"):
        fetch_trusted_image(
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(
                        200,
                        headers={"content-type": "application/octet-stream"},
                        content=b"not an image",
                        request=request,
                    )
                )
            ),
            "https://delivery.eu2.bfl.ai/images/generated.png",
        )


@pytest.mark.parametrize(
    ("content_type", "suffix"),
    [("image/avif", ".avif"), ("image/gif", ".gif"), ("image/webp", ".webp")],
)
def test_provider_preserves_the_received_image_file_extension(content_type: str, suffix: str) -> None:
    from ai_article_studio.providers import _extension_for_content_type

    assert _extension_for_content_type(content_type) == suffix


def test_explicit_provider_failure_does_not_silently_fall_back() -> None:
    from ai_article_studio.providers import ImageGenerationService

    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: StubProvider("PEXELS", fail=True),
            ImageProvider.SILICONFLOW: StubProvider("SILICONFLOW"),
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )

    with pytest.raises(RuntimeError, match="PEXELS unavailable"):
        service.generate_one(prompt(), ImageProvider.PEXELS, 0)


def test_mock_provider_creates_a_local_image_file(tmp_path) -> None:
    from ai_article_studio.providers import MockImageProvider

    result = MockImageProvider(tmp_path).generate(prompt(), 1)

    assert result.source == "MOCK"
    assert result.image_url.startswith("/uploads/images/")
    assert len(list(tmp_path.glob("*.svg"))) == 1


def test_mock_provider_scopes_new_assets_to_the_task_directory(tmp_path) -> None:
    from ai_article_studio.providers import MockImageProvider

    task_id = "a" * 32
    result = MockImageProvider(tmp_path).generate(prompt(), 1, task_id=task_id)

    assert result.image_url.startswith(f"/uploads/images/{task_id}/")
    assert len(list((tmp_path / task_id).glob("*.svg"))) == 1


def test_real_image_clients_inherit_the_configured_request_timeout(tmp_path) -> None:
    from ai_article_studio.config import RuntimeMode, Settings
    from ai_article_studio.providers import create_image_service

    service = create_image_service(
        Settings(
            app_mode=RuntimeMode.REAL,
            pexels_api_key="pexels-key",
            siliconflow_api_key="siliconflow-key",
            google_ai_api_key="google-key",
            request_timeout_seconds=123,
            upload_dir=tmp_path,
            _env_file=None,
        )
    )

    assert service.providers[ImageProvider.PEXELS].client.timeout.connect == 123  # type: ignore[union-attr]
    assert service.providers[ImageProvider.SILICONFLOW].client.timeout.connect == 123  # type: ignore[union-attr]
    assert service.providers[ImageProvider.GOOGLE_AI].client.timeout.connect == 123  # type: ignore[union-attr]


def test_image_count_uses_headings_or_content_length_and_clamps() -> None:
    from ai_article_studio.providers import calculate_image_count

    assert calculate_image_count("short") == 3
    assert calculate_image_count("## 一\n内容\n## 二\n内容") == 3
    assert calculate_image_count("字" * 10_000) == 8


def test_pexels_provider_maps_provider_response_to_attributed_image() -> None:
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "key"
        assert request.url.params["orientation"] == "landscape"
        return httpx.Response(
            200,
            json={
                "photos": [
                    {
                        "url": "https://www.pexels.com/photo/1/",
                        "alt": "Technology cover design",
                        "photographer": "Ada",
                        "photographer_url": "https://www.pexels.com/@ada/",
                        "src": {"landscape": "https://images.pexels.com/photo.jpg"},
                    }
                ]
            },
            request=request,
        )

    provider = PexelsImageProvider("key", httpx.Client(transport=httpx.MockTransport(handler)))

    result = provider.generate(prompt(), 0)

    assert result.source == "PEXELS"
    assert result.author == "Ada"
    assert result.source_url == "https://www.pexels.com/photo/1/"


def test_pexels_provider_uses_the_planned_english_stock_query() -> None:
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "designer workspace"
        return httpx.Response(
            200,
            json={
                "photos": [
                    {
                        "url": "https://www.pexels.com/photo/1/",
                        "alt": "Designer workspace with a desk",
                        "src": {"landscape": "https://images.pexels.com/photo.jpg"},
                    }
                ]
            },
            request=request,
        )

    planned_prompt = prompt().model_copy(update={"image_title": "中文标题不会用于检索", "stock_query": "designer workspace"})
    result = PexelsImageProvider("key", httpx.Client(transport=httpx.MockTransport(handler))).generate(planned_prompt, 0)

    assert result.source == "PEXELS"


def test_pexels_provider_skips_a_mismatched_candidate_and_selects_the_first_matching_stock_query() -> None:
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "designer workspace"
        assert request.url.params["per_page"] == "10"
        return httpx.Response(
            200,
            json={
                "photos": [
                    {
                        "url": "https://www.pexels.com/photo/mountain/",
                        "alt": "Snow mountain landscape",
                        "src": {"landscape": "https://images.pexels.com/mountain.jpg"},
                    },
                    {
                        "url": "https://www.pexels.com/photo/designer/",
                        "alt": "Designer working at a shared workspace",
                        "src": {"landscape": "https://images.pexels.com/designer.jpg"},
                    },
                ]
            },
            request=request,
        )

    planned_prompt = prompt().model_copy(update={"stock_query": "designer workspace"})
    result = PexelsImageProvider("key", httpx.Client(transport=httpx.MockTransport(handler))).generate(planned_prompt, 0)

    assert result.source_url == "https://www.pexels.com/photo/designer/"


def test_pexels_provider_rejects_when_no_candidate_matches_the_stock_query() -> None:
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "photos": [
                    {
                        "url": "https://www.pexels.com/photo/mountain/",
                        "alt": "Snow mountain landscape",
                        "src": {"landscape": "https://images.pexels.com/mountain.jpg"},
                    }
                ]
            },
            request=request,
        )

    planned_prompt = prompt().model_copy(update={"stock_query": "designer workspace"})
    provider = PexelsImageProvider("key", httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(RuntimeError, match="matching stock query"):
        provider.generate(planned_prompt, 0)


def test_pexels_provider_requires_two_matching_terms_for_a_two_word_stock_query() -> None:
    from ai_article_studio.providers import PexelsImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "photos": [
                    {
                        "url": "https://www.pexels.com/photo/designer-outdoors/",
                        "alt": "Designer working outdoors in sunlight",
                        "src": {"landscape": "https://images.pexels.com/designer-outdoors.jpg"},
                    }
                ]
            },
            request=request,
        )

    planned_prompt = prompt().model_copy(update={"stock_query": "designer workspace"})
    provider = PexelsImageProvider("key", httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(RuntimeError, match="matching stock query"):
        provider.generate(planned_prompt, 0)


def test_auto_diagram_role_never_uses_pexels_when_generation_fails() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW", fail=True)
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    diagram_prompt = prompt().model_copy(update={"visual_role": "DIAGRAM", "stock_query": ""})

    result = service.generate_one(diagram_prompt, ImageProvider.AUTO, 0)

    assert result.source == "FALLBACK"
    assert siliconflow.calls == [0]
    assert pexels.calls == []


def test_auto_scene_role_prefers_pexels_even_when_its_prompt_mentions_an_interface() -> None:
    from ai_article_studio.providers import ImageGenerationService

    pexels = StubProvider("PEXELS")
    siliconflow = StubProvider("SILICONFLOW")
    service = ImageGenerationService(
        {
            ImageProvider.PEXELS: pexels,
            ImageProvider.SILICONFLOW: siliconflow,
            ImageProvider.GOOGLE_AI: StubProvider("GOOGLE_AI"),
            ImageProvider.MOCK: StubProvider("MOCK"),
        }
    )
    scene_prompt = prompt().model_copy(update={"visual_role": "SCENE", "usage_scene": "人物使用 UI 界面", "prompt_en": "designer at code interface"})

    result = service.generate_one(scene_prompt, ImageProvider.AUTO, 0)

    assert result.source == "PEXELS"
    assert pexels.calls == [0]
    assert siliconflow.calls == []


def test_google_provider_uses_header_for_key_and_persists_inline_image(tmp_path) -> None:
    from ai_article_studio.providers import GoogleAIImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-google-key"
        assert "key" not in request.url.params
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]}}]},
            request=request,
        )

    provider = GoogleAIImageProvider(
        "test-google-key",
        "gemini-test",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.generate(prompt(), 0)

    assert result.source == "GOOGLE_AI"
    assert (tmp_path / result.image_url.rsplit("/", 1)[-1]).read_bytes() == b"image"


def test_google_provider_rejects_empty_inline_image_without_persisting(tmp_path, monkeypatch) -> None:
    import ai_article_studio.providers as providers_module
    from ai_article_studio.domain import AdapterResponseError
    from ai_article_studio.providers import GoogleAIImageProvider

    persist_calls: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": ""}}]}}]},
            request=request,
        )

    def record_persist(content: bytes, *_args, **_kwargs) -> str:
        persist_calls.append(content)
        return "/uploads/images/unexpected.png"

    monkeypatch.setattr(providers_module, "persist_bytes", record_persist)
    provider = GoogleAIImageProvider(
        "test-google-key",
        "gemini-test",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(AdapterResponseError, match="Google AI response has no inline image"):
        provider.generate(prompt(), 0)

    assert persist_calls == []


def test_siliconflow_provider_uses_the_current_images_endpoint_payload_and_persists_the_asset(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/generations":
            assert request.headers["Authorization"] == "Bearer test-siliconflow-key"
            payload = __import__("json").loads(request.content)
            assert payload == {
                "model": "black-forest-labs/FLUX.2-pro",
                "prompt": "technology cover",
                "image_size": "512x512",
            }
            return httpx.Response(200, json={"images": [{"url": "https://images.siliconflow.cn/generated.png"}]}, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"\x89PNG\r\n\x1a\nimage",
            request=request,
        )

    provider = SiliconFlowImageProvider(
        "test-siliconflow-key",
        "https://api.siliconflow.com/v1",
        "black-forest-labs/FLUX.2-pro",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
        image_size="512x512",
    )

    result = provider.generate(prompt(), 0)

    assert result.source == "SILICONFLOW"
    assert (tmp_path / result.image_url.rsplit("/", 1)[-1]).read_bytes() == b"\x89PNG\r\n\x1a\nimage"


def test_siliconflow_generation_prompt_carries_the_selected_article_claim(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/generations":
            captured.update(__import__("json").loads(request.content))
            return httpx.Response(200, json={"images": [{"url": "https://images.siliconflow.cn/generated.png"}]}, request=request)
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image", request=request)

    provider = SiliconFlowImageProvider(
        "test-siliconflow-key",
        "https://api.siliconflow.cn/v1",
        "Qwen/Qwen-Image",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    grounded_prompt = prompt().model_copy(
        update={
            "usage_scene": "让读者看懂重复流程如何出现偏离",
            "prompt_en": "split view of one repeated workflow, one stable path and one diverging path, clean editorial illustration",
            "prompt_zh": "同一流程重复执行后分成稳定路径与偏离路径的对照图，无文字",
            "anchor_excerpt": "重复执行同一流程后，记录偏离发生的位置与原因。",
        }
    )

    provider.generate(grounded_prompt, 0)

    request_prompt = str(captured["prompt"])
    assert "重复执行同一流程后，记录偏离发生的位置与原因。" in request_prompt
    assert "让读者看懂重复流程如何出现偏离" in request_prompt
    assert "one stable path and one diverging path" in request_prompt
    assert "Do not invent an unrelated domain" in request_prompt


def test_siliconflow_qwen_image_uses_cfg_without_kolors_only_parameters(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/generations":
            requests.append(__import__("json").loads(request.content))
            return httpx.Response(
                200,
                json={"images": [{"url": "https://images.siliconflow.cn/generated.png"}]},
                request=request,
            )
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image", request=request)

    provider = SiliconFlowImageProvider(
        "test-siliconflow-key",
        "https://api.siliconflow.cn/v1",
        "Qwen/Qwen-Image",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
        image_size="1664x928",
    )
    provider.cfg = 4.0

    provider.generate(prompt(), 0)

    assert requests == [
        {
            "model": "Qwen/Qwen-Image",
            "prompt": "technology cover",
            "image_size": "1664x928",
            "num_inference_steps": 20,
            "cfg": 4.0,
        }
    ]


def test_siliconflow_rejects_an_image_edit_model_without_a_source_image(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(500, request=request)

    provider = SiliconFlowImageProvider(
        "test-siliconflow-key",
        "https://api.siliconflow.cn/v1",
        "Qwen/Qwen-Image-Edit-2509",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValueError, match="需要提供源图片"):
        provider.generate(prompt(), 0)

    assert calls == []


def test_siliconflow_401_is_rewritten_as_a_safe_actionable_message(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Invalid token", request=request)

    provider = SiliconFlowImageProvider(
        "secret-must-not-leak",
        "https://api.siliconflow.cn/v1",
        "Kwai-Kolors/Kolors",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError, match="SiliconFlow 鉴权失败（401）") as error:
        provider.generate(prompt(), 0)

    assert "secret-must-not-leak" not in str(error.value)


def test_siliconflow_retries_a_confirmed_transient_generation_rejection_once(tmp_path) -> None:
    from ai_article_studio.providers import SiliconFlowImageProvider

    generation_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation_calls
        if request.url.path == "/v1/images/generations":
            generation_calls += 1
            if generation_calls == 1:
                return httpx.Response(503, headers={"retry-after": "0"}, request=request)
            return httpx.Response(200, json={"images": [{"url": "https://images.siliconflow.cn/generated.png"}]}, request=request)
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image", request=request)

    provider = SiliconFlowImageProvider(
        "test-siliconflow-key",
        "https://api.siliconflow.cn/v1",
        "Kwai-Kolors/Kolors",
        tmp_path,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.generate(prompt(), 0)

    assert result.source == "SILICONFLOW"
    assert generation_calls == 2


def test_trusted_image_download_rejects_large_declared_content_length() -> None:
    from ai_article_studio.safe_fetch import UnsafeImageUrlError, fetch_trusted_image

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png", "content-length": str(10 * 1024 * 1024 + 1)},
            request=request,
        )

    with pytest.raises(UnsafeImageUrlError, match="大小限制"):
        fetch_trusted_image(
            httpx.Client(transport=httpx.MockTransport(handler)),
            "https://images.pexels.com/too-large.png",
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://sf-maas.s3.us-east-1.amazonaws.com/images/generated.png",
        "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/images/generated.png",
        "https://delivery.us2.bfl.ai/images/generated.png",
        "https://delivery.us4.bfl.ai/images/generated.png",
        "https://delivery.eu2.bfl.ai/images/generated.png",
    ],
)
def test_trusted_image_download_allows_documented_siliconflow_asset_hosts(url: str) -> None:
    from ai_article_studio.safe_fetch import fetch_trusted_image

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"image",
            request=request,
        )

    image = fetch_trusted_image(
        httpx.Client(transport=httpx.MockTransport(handler)),
        url,
    )

    assert image.content == b"image"


@pytest.mark.parametrize(
    "host",
    [
        "delivery.eu2.bfl.ai.evil.example",
        "evil.delivery.eu2.bfl.ai",
        "delivery.-eu.bfl.ai",
        "delivery.eu-.bfl.ai",
        "delivery..bfl.ai",
    ],
)
def test_bfl_delivery_allowlist_rejects_non_delivery_edge_hosts(host: str) -> None:
    from ai_article_studio.safe_fetch import allowed_image_host

    assert allowed_image_host(host) is False

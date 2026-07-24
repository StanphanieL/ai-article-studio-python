from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.schemas import ImagePromptOption, ImageResultOption, ImageVisualReviewStatus


def _image() -> ImageResultOption:
    return ImageResultOption(
        image_title="交互流程图",
        usage_scene="解释工作流",
        prompt_zh="简洁流程图",
        prompt_en="clean workflow diagram",
        image_url="/uploads/images/task/flow.png",
        source="MOCK",
    )


def test_disabled_vision_review_skips_http_and_marks_image_skipped(tmp_path: Path) -> None:
    from ai_article_studio.vision import VisionReviewer

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, request=request)

    reviewer = VisionReviewer(
        Settings(app_mode=RuntimeMode.MOCK, upload_dir=tmp_path, _env_file=None),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    review = reviewer.review(_image(), ImagePromptOption(**_image().model_dump(exclude={"image_url", "source"})))

    assert review.status is ImageVisualReviewStatus.SKIPPED
    assert review.score is None
    assert calls == []


def test_configured_vision_review_sends_safe_image_data_uri_and_parses_result(tmp_path: Path) -> None:
    from ai_article_studio.vision import VisionReviewer

    image_path = tmp_path / "task" / "flow.png"
    image_path.parent.mkdir()
    image_bytes = b"\x89PNG\r\n\x1a\nvisual-test"
    image_path.write_bytes(image_bytes)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"status": "PASSED", "score": 91, "issues": []})}}
                ]
            },
            request=request,
        )

    settings = Settings(
        app_mode=RuntimeMode.REAL,
        upload_dir=tmp_path,
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key="text-key",
        vision_model="qwen3-vl-plus",
        _env_file=None,
    )
    reviewer = VisionReviewer(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))

    review = reviewer.review(_image(), ImagePromptOption(**_image().model_dump(exclude={"image_url", "source"})))

    assert review.status is ImageVisualReviewStatus.PASSED
    assert review.score == 91
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["authorization"] == "Bearer text-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    content = payload["messages"][0]["content"]  # type: ignore[index]
    image_url = content[1]["image_url"]["url"]  # type: ignore[index]
    assert image_url == "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")


def test_unavailable_or_malformed_vision_response_is_nonfatal(tmp_path: Path) -> None:
    from ai_article_studio.vision import VisionReviewer

    image_path = tmp_path / "task" / "flow.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nvisual-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}, request=request)

    reviewer = VisionReviewer(
        Settings(
            upload_dir=tmp_path,
            openai_base_url="https://vision.example.test/v1",
            openai_api_key="key",
            vision_model="vision-test",
            _env_file=None,
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    review = reviewer.review(_image(), ImagePromptOption(**_image().model_dump(exclude={"image_url", "source"})))

    assert review.status is ImageVisualReviewStatus.UNAVAILABLE
    assert review.score is None
    assert review.issues == ["视觉质检结果不可用，可按需重试。"]


def test_vision_transport_failure_becomes_unavailable_without_provider_details(tmp_path: Path) -> None:
    from ai_article_studio.vision import VisionReviewer

    image_path = tmp_path / "task" / "flow.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nvisual-test")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private provider details", request=request)

    reviewer = VisionReviewer(
        Settings(upload_dir=tmp_path, openai_api_key="key", vision_model="vision-test", _env_file=None),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    review = reviewer.review(_image(), ImagePromptOption(**_image().model_dump(exclude={"image_url", "source"})))

    assert review.status is ImageVisualReviewStatus.UNAVAILABLE
    assert "private provider details" not in " ".join(review.issues)


def test_malformed_vision_provider_envelope_is_unavailable(tmp_path: Path) -> None:
    from ai_article_studio.vision import VisionReviewer

    image_path = tmp_path / "task" / "flow.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nvisual-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[], request=request)

    reviewer = VisionReviewer(
        Settings(upload_dir=tmp_path, openai_api_key="key", vision_model="vision-test", _env_file=None),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    review = reviewer.review(_image(), ImagePromptOption(**_image().model_dump(exclude={"image_url", "source"})))

    assert review.status is ImageVisualReviewStatus.UNAVAILABLE

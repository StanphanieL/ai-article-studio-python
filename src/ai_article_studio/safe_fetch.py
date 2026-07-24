"""SSRF-safe remote image fetching shared by providers and exporters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

MAX_IMAGE_DOWNLOAD_BYTES = 10 * 1024 * 1024
_GENERIC_BINARY_CONTENT_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


class UnsafeImageUrlError(ValueError):
    """Raised before a request can reach an untrusted image host."""


@dataclass(frozen=True)
class DownloadedImage:
    content: bytes
    content_type: str


def allowed_image_host(host: str) -> bool:
    normalized = host.lower().rstrip(".")
    if (
        _is_picsum_host(normalized)
        or _is_bfl_delivery_host(normalized)
        or _is_siliconflow_asset_host(normalized)
    ):
        return True
    return normalized.endswith(
        (
            ".pexels.com",
            ".siliconflow.cn",
            ".siliconflow.com",
            ".googleusercontent.com",
            ".googleapis.com",
        )
    )


def _is_picsum_host(host: str) -> bool:
    """Allow Picsum itself and its single-label CDN redirect host only."""

    return host == "picsum.photos" or re.fullmatch(r"[a-z0-9-]+\.picsum\.photos", host) is not None


def _is_siliconflow_asset_host(host: str) -> bool:
    """Restrict vendor object-storage URLs to the documented SiliconFlow bucket prefix."""

    return host.startswith("sf-maas") and host.endswith(
        (".s3.us-east-1.amazonaws.com", ".oss-cn-shanghai.aliyuncs.com")
    )


def _is_bfl_delivery_host(host: str) -> bool:
    """Allow only single-label Black Forest Labs image-delivery edge hosts."""

    # BFL routes generated images through region-specific hosts such as us2 and
    # eu2.  Keep the trusted ``delivery`` service boundary while accepting a
    # future vendor-controlled region label; do not permit nested subdomains.
    return re.fullmatch(r"delivery\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.bfl\.ai", host) is not None


def fetch_trusted_image(
    client: httpx.Client,
    image_url: str,
    max_download_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES,
) -> DownloadedImage:
    """Fetch an HTTPS image only after validating every redirect target."""

    current_url = image_url
    for _ in range(4):
        _validate_remote_url(current_url)
        with client.stream("GET", current_url, follow_redirects=False) as response:
            if 300 <= response.status_code < 400:
                location = response.headers.get("location")
                if not location:
                    raise UnsafeImageUrlError("图片重定向缺少目标地址")
                current_url = urljoin(current_url, location)
                _validate_remote_url(current_url)
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            declared_size = int(response.headers.get("content-length", "0") or 0)
            if declared_size > max_download_bytes:
                raise UnsafeImageUrlError("远程图片超过下载大小限制")
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > max_download_bytes:
                    raise UnsafeImageUrlError("远程图片超过下载大小限制")
            image_content_type = _confirmed_image_content_type(content_type, bytes(content))
            if image_content_type is None:
                label = content_type or "未提供"
                raise UnsafeImageUrlError(f"远程响应不是图片（Content-Type: {label}）")
            return DownloadedImage(bytes(content), image_content_type)
    raise UnsafeImageUrlError("图片重定向次数超过上限")


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or not allowed_image_host(parsed.hostname):
        host = parsed.hostname or "未知主机"
        raise UnsafeImageUrlError(f"图片来源不在允许列表中（{host}）")


def _confirmed_image_content_type(declared_content_type: str, content: bytes) -> str | None:
    """Keep strict MIME checks while accepting honest CDN header mistakes.

    Some image-generation CDNs return a valid raster image with the generic
    ``application/octet-stream`` header.  Trusting that header would permit an
    arbitrary download, so accept it only when the payload itself has a known
    image signature and persist it under the canonical image MIME type.
    """

    if declared_content_type.startswith("image/"):
        return declared_content_type
    if declared_content_type not in _GENERIC_BINARY_CONTENT_TYPES:
        return None
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    if len(content) >= 12 and content[4:8] == b"ftyp" and content[8:12] in {b"avif", b"avis"}:
        return "image/avif"
    return None

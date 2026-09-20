# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Policy matrix for utils/media_downloader.py (issue #4974).

The downloader pulls client-supplied media (e.g. presigned S3 URLs) under a
process-wide policy: http(s) only, a REQUIRED hostname allowlist (exact or
label-anchored wildcard) checked on every redirect hop, a total deadline, and
a streamed byte cap. Transports are faked so no test touches the network.
"""

import asyncio

import time

import httpx
import pytest
from config.settings import settings
from utils.media_downloader import (
    MediaDownloadError,
    MediaDownloadFetchError,
    MediaDownloadPolicyError,
    MediaDownloadTooLargeError,
    check_media_url_policy,
    download_media_url,
    is_media_url,
)


class _HandlerTransport(httpx.AsyncBaseTransport):
    """Route every request through an async handler; allows sleeps."""

    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request):
        return await self._handler(request)


class _ChunkStream(httpx.AsyncByteStream):
    """Body stream that never advertises Content-Length."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_HandlerTransport(handler))


@pytest.fixture
def media_url_defaults(monkeypatch):
    """Pin the media-URL settings so tests don't depend on env state.

    The allowlist is required (deny-when-empty), so the default covers every
    hostname the download tests use.
    """
    monkeypatch.setattr(settings, "media_url_download_enabled", True, raising=False)
    monkeypatch.setattr(
        settings,
        "media_url_allowed_domains",
        "host.example,bucket.s3.amazonaws.com,allowed.example",
        raising=False,
    )
    monkeypatch.setattr(settings, "media_url_max_bytes", 1024 * 1024, raising=False)
    monkeypatch.setattr(settings, "media_url_timeout_seconds", 5.0, raising=False)
    monkeypatch.setattr(settings, "media_url_max_redirects", 5, raising=False)
    return settings


class TestIsMediaUrl:
    def test_http_and_https_are_urls(self):
        assert is_media_url("http://host/x")
        assert is_media_url("https://host/x")

    def test_scheme_detection_is_case_insensitive(self):
        assert is_media_url("HTTPS://Host/X")
        assert is_media_url("Http://host/x")

    def test_base64_and_other_schemes_are_not(self):
        assert not is_media_url("iVBORw0KGgo=")
        assert not is_media_url("data:image/png;base64,AAAA")
        assert not is_media_url("ftp://host/x")
        assert not is_media_url("file:///etc/passwd")


class TestPolicy:
    def test_rejects_non_http_scheme(self, media_url_defaults):
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("ftp://host/x")

    def test_rejects_missing_hostname(self, media_url_defaults):
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https:///no-host")

    def test_rejects_disallowed_domain(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(settings, "media_url_allowed_domains", "allowed.example")
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https://evil.example/x")

    def test_allowlist_is_case_insensitive(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(
            settings, "media_url_allowed_domains", "BUCKET.S3.AmazonAWS.com"
        )
        check_media_url_policy("https://bucket.s3.amazonaws.com/key")

    def test_empty_allowlist_denies_all_urls(self, media_url_defaults, monkeypatch):
        # The allowlist is required: deny-when-empty, not allow-all.
        monkeypatch.setattr(settings, "media_url_allowed_domains", "")
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https://anything.example/x")

    def test_wildcard_matches_any_subdomain_depth(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_allowed_domains", "*.s3.amazonaws.com")
        check_media_url_policy("https://bucket.s3.amazonaws.com/key")
        check_media_url_policy("https://a.b.s3.amazonaws.com/key")

    def test_wildcard_is_label_anchored(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(settings, "media_url_allowed_domains", "*.s3.amazonaws.com")
        # No '.' label boundary: a lookalike host must not match.
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https://evil-s3.amazonaws.com/key")
        # The bare suffix itself must not match a '*.suffix' entry.
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https://s3.amazonaws.com/key")

    def test_bare_star_entry_is_a_server_error(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(settings, "media_url_allowed_domains", "*")
        with pytest.raises(MediaDownloadError) as exc_info:
            check_media_url_policy("https://host.example/x")
        assert type(exc_info.value) is MediaDownloadError

    def test_disabled_rejects_urls(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(settings, "media_url_download_enabled", False)
        with pytest.raises(MediaDownloadPolicyError):
            check_media_url_policy("https://allowed.example/x")


class TestDownload:
    async def test_success_returns_bytes_and_keeps_query_auth(self, media_url_defaults):
        seen = {}

        async def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, content=b"png-bytes")

        async with _client(handler) as client:
            data = await download_media_url(
                "https://bucket.s3.amazonaws.com/k.png?X-Amz-Signature=abc123",
                client=client,
            )
        assert data == b"png-bytes"
        # Presigned query-string auth must pass through untouched.
        assert "X-Amz-Signature=abc123" in seen["url"]

    async def test_follows_redirect_to_allowed_host(self, media_url_defaults):
        async def handler(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final"})
            return httpx.Response(200, content=b"after-redirect")

        async with _client(handler) as client:
            data = await download_media_url("https://host.example/start", client=client)
        assert data == b"after-redirect"

    async def test_redirect_to_disallowed_host_is_rejected(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_allowed_domains", "allowed.example")

        async def handler(request):
            if request.url.host == "allowed.example":
                return httpx.Response(
                    302, headers={"location": "https://evil.example/steal"}
                )
            return httpx.Response(200, content=b"nope")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadPolicyError):
                await download_media_url("https://allowed.example/x", client=client)

    async def test_redirect_limit(self, media_url_defaults, monkeypatch):
        monkeypatch.setattr(settings, "media_url_max_redirects", 2)

        async def handler(request):
            return httpx.Response(302, headers={"location": "/again"})

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadPolicyError):
                await download_media_url("https://host.example/x", client=client)

    async def test_content_length_over_cap_rejected_before_read(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_max_bytes", 10)

        async def handler(request):
            # Content-Length is set by httpx from the body size (100 > 10).
            return httpx.Response(200, content=b"x" * 100)

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadTooLargeError):
                await download_media_url("https://host.example/x", client=client)

    async def test_streamed_body_over_cap_rejected_mid_stream(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_max_bytes", 10)

        async def handler(request):
            # No Content-Length header: the cap must trip while streaming.
            return httpx.Response(200, stream=_ChunkStream([b"x" * 8, b"x" * 8]))

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadTooLargeError):
                await download_media_url("https://host.example/x", client=client)

    async def test_error_text_never_carries_the_query_string(self, media_url_defaults):
        async def handler(request):
            return httpx.Response(403, content=b"expired")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadFetchError) as exc_info:
                await download_media_url(
                    "https://bucket.s3.amazonaws.com/k.png?X-Amz-Signature=secret",
                    client=client,
                )
        # Presigned query params are credentials; they must not leak into
        # error details or logs.
        assert "X-Amz-Signature" not in str(exc_info.value)
        assert "secret" not in str(exc_info.value)

    async def test_origin_error_maps_to_fetch_error(self, media_url_defaults):
        async def handler(request):
            # An expired presigned URL surfaces as 403 from S3.
            return httpx.Response(403, content=b"expired")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadFetchError):
                await download_media_url("https://host.example/x", client=client)

    async def test_total_deadline_covers_body_read(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_timeout_seconds", 0.05)

        async def handler(request):
            await asyncio.sleep(0.2)
            return httpx.Response(200, content=b"late")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadFetchError):
                await download_media_url("https://host.example/x", client=client)

    async def test_redirect_without_location_is_a_fetch_error(self, media_url_defaults):
        async def handler(request):
            return httpx.Response(302, content=b"lost")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadFetchError):
                await download_media_url("https://host.example/x", client=client)

    async def test_caller_supplied_deadline_is_shared(self, media_url_defaults):
        async def handler(request):  # pragma: no cover - must never run
            raise AssertionError("expired deadline must not send a request")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadFetchError):
                await download_media_url(
                    "https://host.example/x",
                    client=client,
                    deadline=time.monotonic() - 1,
                )

    async def test_misconfigured_allowlist_is_a_server_error(
        self, media_url_defaults, monkeypatch
    ):
        # An operator typo must not surface as a client-fault policy error.
        # A 64-char DNS label is invalid and fails IDNA encoding.
        monkeypatch.setattr(
            settings, "media_url_allowed_domains", "a" * 64 + ".example"
        )

        async def handler(request):  # pragma: no cover - must never run
            raise AssertionError("misconfiguration must fail before any request")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadError) as exc_info:
                await download_media_url("https://host.example/x", client=client)
        assert type(exc_info.value) is MediaDownloadError

    async def test_nonpositive_timeout_is_a_server_error(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_timeout_seconds", 0)

        async def handler(request):  # pragma: no cover - must never run
            raise AssertionError("misconfiguration must fail before any request")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadError) as exc_info:
                await download_media_url("https://host.example/x", client=client)
        assert type(exc_info.value) is MediaDownloadError

    async def test_disabled_rejects_before_any_request(
        self, media_url_defaults, monkeypatch
    ):
        monkeypatch.setattr(settings, "media_url_download_enabled", False)

        async def handler(request):  # pragma: no cover - must never run
            raise AssertionError("request must not be sent when disabled")

        async with _client(handler) as client:
            with pytest.raises(MediaDownloadPolicyError):
                await download_media_url("https://host.example/x", client=client)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""I2V presigned-URL wiring (issue #4974).

``image_prompts[].image`` accepts an http(s) URL next to inline base64.
The URL is downloaded at the API layer and replaced with base64 before the
job is enqueued, so runners and workers keep seeing base64 only.
"""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import PIL.Image
import pytest
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from fastapi import HTTPException
from open_ai_api.video import (
    _resolve_image_prompt_urls,
    submit_generate_video_i2v_request,
)
from utils.image_manager import ImageManager
from utils.media_downloader import (
    MediaDownloadFetchError,
    MediaDownloadPolicyError,
    MediaDownloadTooLargeError,
)

# Same 1x1 PNG as tests/test_video_api.py — hardcoded so the test does not
# depend on PIL being real (conftest mocks PIL when Pillow is not installed).
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)
_TINY_PNG_BYTES = base64.b64decode(_TINY_PNG_BASE64)

_URL = "https://bucket.s3.us-east-1.amazonaws.com/frame0.png?X-Amz-Signature=sig"

_PIL_IS_MOCKED = isinstance(PIL.Image, MagicMock)


class TestImagePromptEntryAcceptsUrls:
    def test_url_value_skips_base64_validation(self):
        entry = ImagePromptEntry(image=_URL, frame_pos=0)
        assert entry.image == _URL

    def test_inline_base64_still_validates(self):
        entry = ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=0)
        assert entry.image == _TINY_PNG_BASE64

    def test_garbage_string_is_still_rejected(self):
        with pytest.raises(Exception):
            ImagePromptEntry(image="not-a-url-and-not-base64!!", frame_pos=0)


class TestResolveImagePromptUrls:
    async def test_url_entry_is_replaced_with_base64(self):
        request = VideoI2VGenerateRequest(
            prompt="p",
            image_prompts=[
                {"image": _URL, "frame_pos": 0},
                {"image": _TINY_PNG_BASE64, "frame_pos": 1},
            ],
        )
        with patch(
            "open_ai_api.video.download_media_url",
            new=AsyncMock(return_value=_TINY_PNG_BYTES),
        ) as mock_download:
            await _resolve_image_prompt_urls(request)

        mock_download.assert_awaited_once()
        assert mock_download.await_args.args == (_URL,)
        resolved = request.image_prompts[0].image
        assert resolved != _URL
        assert base64.b64decode(resolved) == _TINY_PNG_BYTES
        # Inline entry untouched.
        assert request.image_prompts[1].image == _TINY_PNG_BASE64

    async def test_t2v_request_is_a_noop(self):
        request = VideoGenerateRequest(prompt="p")
        with patch(
            "open_ai_api.video.download_media_url", new=AsyncMock()
        ) as mock_download:
            await _resolve_image_prompt_urls(request)
        mock_download.assert_not_awaited()

    @pytest.mark.parametrize(
        "error,status",
        [
            (MediaDownloadPolicyError("blocked"), 400),
            (MediaDownloadTooLargeError("too big"), 413),
            (MediaDownloadFetchError("HTTP 403"), 422),
        ],
    )
    async def test_download_errors_map_to_http_statuses(self, error, status):
        request = VideoI2VGenerateRequest(
            prompt="p", image_prompts=[{"image": _URL, "frame_pos": 0}]
        )
        with patch(
            "open_ai_api.video.download_media_url",
            new=AsyncMock(side_effect=error),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _resolve_image_prompt_urls(request)
        assert exc_info.value.status_code == status

    async def test_oversized_download_is_rejected_with_413(self):
        # 8,000,000 bytes base64-encode past the 10,000,000-char field cap;
        # without the endpoint check this would 202 and then fail validation
        # inside an SP-runner worker mid-job.
        request = VideoI2VGenerateRequest(
            prompt="p", image_prompts=[{"image": _URL, "frame_pos": 0}]
        )
        with patch(
            "open_ai_api.video.download_media_url",
            new=AsyncMock(return_value=b"x" * 8_000_000),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _resolve_image_prompt_urls(request)
        assert exc_info.value.status_code == 413

    @pytest.mark.skipif(
        _PIL_IS_MOCKED, reason="needs real Pillow to detect a non-image body"
    )
    async def test_undecodable_download_is_rejected_with_422(self):
        request = VideoI2VGenerateRequest(
            prompt="p", image_prompts=[{"image": _URL, "frame_pos": 0}]
        )
        with patch(
            "open_ai_api.video.download_media_url",
            new=AsyncMock(return_value=b"<html>not an image</html>"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _resolve_image_prompt_urls(request)
        assert exc_info.value.status_code == 422


class TestDecoderGuardsAgainstUnresolvedUrls:
    """Runners must only ever see base64; a URL reaching the decoder means a
    submit path skipped `_resolve_image_prompt_urls` — it must fail loudly."""

    @pytest.mark.parametrize(
        "value", ["https://host.example/a.png", "HTTP://host.example/a.png"]
    )
    def test_url_input_raises_a_clear_error(self, value):
        with pytest.raises(ValueError, match="unresolved media URL"):
            ImageManager().base64_to_pil_image(value)


class TestSubmitPathResolvesUrls:
    async def test_i2v_submit_invokes_url_resolution(self):
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            return_value={"id": "job_1", "object": "video", "status": "pending"}
        )
        request = VideoI2VGenerateRequest(
            prompt="p", image_prompts=[{"image": _TINY_PNG_BASE64, "frame_pos": 0}]
        )
        with patch(
            "open_ai_api.video._resolve_image_prompt_urls", new=AsyncMock()
        ) as mock_resolve:
            response = await submit_generate_video_i2v_request(
                request=request,
                service=mock_service,
                api_key="test_key",
            )
        assert response.status_code == 202
        mock_resolve.assert_awaited_once_with(request)

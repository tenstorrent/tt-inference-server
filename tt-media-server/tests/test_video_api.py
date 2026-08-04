# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.constants import JobTypes
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from fastapi import HTTPException
from open_ai_api.video import (
    _is_i2v_only_deployment,
    cancel_video_job,
    download_video_content,
    get_jobs_metadata,
    get_video_metadata,
    reject_text_to_video_on_i2v_deployment,
    submit_generate_video_i2v_request,
    submit_generate_video_i2v_upload,
    submit_generate_video_request,
)


# A real 1x1 red PNG, base64-encoded. Hardcoded (not generated via PIL) because
# ``conftest.py`` mocks the ``PIL`` module to keep unit tests ttnn-free — so
# ``Image.new(...).save(buf)`` is a no-op and ``buf.getvalue()`` returns b""
# under pytest, which breaks the ``min_length=1`` validator on ImagePromptEntry.
# Only the string length matters for DTO validation; content is never decoded.
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _tiny_png_base64() -> str:
    """Return a minimal base64-encoded PNG for I2V request fixtures.

    We don't call PIL here because the test conftest mocks the PIL module;
    a generated image would end up as an empty bytes buffer.
    """
    return _TINY_PNG_BASE64


class TestRejectTextToVideoOnI2VDeployment:
    """``POST /generations`` must not reach the worker on an I2V-only deploy."""

    @patch("open_ai_api.video._is_i2v_only_deployment", return_value=True)
    def test_i2v_deployment_rejects_text_only(self, _mock_i2v):
        with pytest.raises(HTTPException) as exc_info:
            reject_text_to_video_on_i2v_deployment()

        assert exc_info.value.status_code == 422

    @patch("open_ai_api.video._is_i2v_only_deployment", return_value=True)
    def test_rejection_points_at_the_i2v_endpoint(self, _mock_i2v):
        with pytest.raises(HTTPException) as exc_info:
            reject_text_to_video_on_i2v_deployment()

        assert "/generations/i2v" in exc_info.value.detail

    @patch("open_ai_api.video._is_i2v_only_deployment", return_value=False)
    def test_t2v_deployment_allows_text_only(self, _mock_i2v):
        assert reject_text_to_video_on_i2v_deployment() is None


class TestIsI2VOnlyDeployment:
    """Deployment detection: runner first, MODEL only as a fallback.

    Patch through ``open_ai_api.video.settings``, not ``config.settings``: other
    test modules replace ``sys.modules["config.settings"]`` with a Mock at import
    time and never restore it, so patching there would only touch the Mock while
    the code under test keeps its reference to the real settings object.
    """

    @patch("open_ai_api.video.settings.model_runner", "tt-wan2.2-i2v")
    def test_i2v_runner_is_detected(self):
        assert _is_i2v_only_deployment() is True

    @patch("open_ai_api.video.settings.model_runner", "tt-wan2.2")
    def test_t2v_runner_is_not_i2v(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_i2v_only_deployment() is False

    @patch("open_ai_api.video.settings.model_runner", "tt-wan2.2")
    def test_t2v_runner_ignores_a_contradictory_model_env(self):
        """The runner is 1:1 with its model, so a stale MODEL must not win."""
        with patch.dict(os.environ, {"MODEL": "Wan2.2-I2V-A14B-Diffusers"}):
            assert _is_i2v_only_deployment() is False

    @patch("open_ai_api.video.settings.model_runner", "not-a-runner")
    def test_unknown_runner_is_not_i2v(self):
        with patch.dict(os.environ, {"MODEL": "Wan2.2-I2V-A14B-Diffusers"}):
            assert _is_i2v_only_deployment() is False

    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_proxy_runner_falls_back_to_model_env(self):
        """SP_RUNNER serves either T2V or I2V, so MODEL disambiguates it."""
        with patch.dict(os.environ, {"MODEL": "Wan2.2-I2V-A14B-Diffusers"}):
            assert _is_i2v_only_deployment() is True

    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_proxy_runner_with_t2v_model_is_not_i2v(self):
        with patch.dict(os.environ, {"MODEL": "Wan2.2-T2V-A14B-Diffusers"}):
            assert _is_i2v_only_deployment() is False

    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_unknown_model_env_is_not_i2v(self):
        with patch.dict(os.environ, {"MODEL": "not-a-model"}):
            assert _is_i2v_only_deployment() is False


class TestSubmitGenerateVideoRequest:
    """Tests for POST /generations endpoint"""

    @pytest.mark.asyncio
    async def test_submit_generate_video_request_success(self):
        """Test successful video generation job submission"""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            return_value={
                "id": "job_123",
                "object": "video",
                "status": "pending",
                "created_at": 1234567890,
            }
        )

        request = VideoGenerateRequest(prompt="A cat walking in the park")

        response = await submit_generate_video_request(
            request=request,
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 202
        assert response.body is not None
        mock_service.create_job.assert_called_once_with(JobTypes.VIDEO, request)

    @pytest.mark.asyncio
    async def test_submit_generate_video_request_failure(self):
        """Test video generation job submission failure"""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            side_effect=Exception("Service unavailable")
        )

        request = VideoGenerateRequest(prompt="Test video")

        with pytest.raises(Exception) as exc_info:
            await submit_generate_video_request(
                request=request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 500
        assert "Service unavailable" in exc_info.value.detail


class TestGetVideoMetadata:
    """Tests for GET /generations/{job_id} endpoint"""

    def test_get_video_metadata_success(self):
        """Test successful retrieval of video metadata"""
        mock_service = MagicMock()
        mock_service.get_job_metadata = MagicMock(
            return_value={
                "id": "job_123",
                "object": "video",
                "status": "completed",
                "progress": 100,
            }
        )

        response = get_video_metadata(
            job_id="job_123",
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 200
        mock_service.get_job_metadata.assert_called_once_with("job_123")

    def test_get_video_metadata_not_found(self):
        """Test video metadata retrieval when job not found"""
        mock_service = MagicMock()
        mock_service.get_job_metadata = MagicMock(return_value=None)

        with pytest.raises(Exception) as exc_info:
            get_video_metadata(
                job_id="non_existent_job",
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video job not found"


class TestGetJobsMetadata:
    """Tests for GET /jobs endpoint"""

    def test_get_jobs_metadata_success(self):
        """Test successful retrieval of all jobs metadata"""
        mock_service = MagicMock()
        mock_service.get_all_jobs_metadata = MagicMock(
            return_value=[
                {"id": "job_1", "status": "completed"},
                {"id": "job_2", "status": "pending"},
                {"id": "job_3", "status": "processing"},
            ]
        )

        response = get_jobs_metadata(
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 200
        mock_service.get_all_jobs_metadata.assert_called_once()

    def test_get_jobs_metadata_not_found(self):
        """Test jobs metadata retrieval when no jobs found"""
        mock_service = MagicMock()
        mock_service.get_all_jobs_metadata = MagicMock(return_value=None)

        with pytest.raises(Exception) as exc_info:
            get_jobs_metadata(
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Job metadata not found"


class TestDownloadVideoContent:
    """Tests for GET /generations/{job_id}/download endpoint"""

    def test_download_video_content_success(self):
        """Test successful video download"""
        # Create a temporary file to simulate video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            mock_service = MagicMock()
            mock_service.get_job_result_path = MagicMock(return_value=tmp_path)

            mock_request = MagicMock()

            with patch("open_ai_api.video.VideoManager") as mock_video_manager:
                # Make ensure_faststart raise an exception so it uses original path
                mock_video_manager.ensure_faststart.side_effect = Exception(
                    "skip faststart"
                )

                response = download_video_content(
                    job_id="job_123",
                    request=mock_request,
                    service=mock_service,
                    api_key="test_key",
                )

                assert response.path == tmp_path
                assert response.media_type == "video/mp4"
                mock_service.get_job_result_path.assert_called_once_with("job_123")
        finally:
            os.unlink(tmp_path)

    def test_download_video_content_with_faststart(self):
        """Test video download with faststart processing"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            mock_service = MagicMock()
            mock_service.get_job_result_path = MagicMock(return_value=tmp_path)

            mock_request = MagicMock()

            with patch("open_ai_api.video.VideoManager") as mock_video_manager:
                # Make ensure_faststart succeed
                mock_video_manager.ensure_faststart.return_value = None

                response = download_video_content(
                    job_id="job_123",
                    request=mock_request,
                    service=mock_service,
                    api_key="test_key",
                )

                # Should still return FileResponse
                assert response.media_type == "video/mp4"
                mock_video_manager.ensure_faststart.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_download_video_content_not_found(self):
        """Test video download when job result not found"""
        mock_service = MagicMock()
        mock_service.get_job_result_path = MagicMock(return_value=None)

        mock_request = MagicMock()

        with pytest.raises(Exception) as exc_info:
            download_video_content(
                job_id="job_123",
                request=mock_request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video content not available"

    def test_download_video_content_file_not_exists(self):
        """Test video download when file path doesn't exist"""
        mock_service = MagicMock()
        mock_service.get_job_result_path = MagicMock(
            return_value="/nonexistent/path.mp4"
        )

        mock_request = MagicMock()

        with pytest.raises(Exception) as exc_info:
            download_video_content(
                job_id="job_123",
                request=mock_request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video content not available"

    def test_download_video_content_invalid_type(self):
        """Test video download when result is not a string"""
        mock_service = MagicMock()
        mock_service.get_job_result_path = MagicMock(
            return_value={"error": "not a path"}
        )

        mock_request = MagicMock()

        with pytest.raises(Exception) as exc_info:
            download_video_content(
                job_id="job_123",
                request=mock_request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video content not available"


class TestCancelVideoJob:
    """Tests for DELETE /generations/{job_id} endpoint"""

    def test_cancel_video_job_success(self):
        """Test successful video job cancellation"""
        mock_service = MagicMock()
        mock_service.cancel_job = MagicMock(
            return_value={
                "id": "job_123",
                "object": JobTypes.VIDEO.value,
                "status": "cancelling",
                "created_at": 1000,
            }
        )

        response = cancel_video_job(
            job_id="job_123",
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 200
        mock_service.cancel_job.assert_called_once_with("job_123")

    def test_cancel_video_job_not_found(self):
        """Test video job cancellation when job not found"""
        mock_service = MagicMock()
        mock_service.cancel_job = MagicMock(return_value=None)

        with pytest.raises(Exception) as exc_info:
            cancel_video_job(
                job_id="non_existent_job",
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video job not found"


class TestVideoGenerateRequestValidation:
    """Tests for VideoGenerateRequest validation"""

    def test_video_generate_request_with_prompt(self):
        """Test VideoGenerateRequest creation with required prompt"""
        request = VideoGenerateRequest(prompt="A cat walking in the park")
        assert request.prompt == "A cat walking in the park"

    def test_video_generate_request_with_all_params(self):
        """Test VideoGenerateRequest with all optional parameters"""
        request = VideoGenerateRequest(
            prompt="A sunset over the ocean",
            negative_prompt="blurry, low quality",
            num_inference_steps=30,
            seed=42,
        )
        assert request.prompt == "A sunset over the ocean"
        assert request.negative_prompt == "blurry, low quality"
        assert request.num_inference_steps == 30
        assert request.seed == 42


class TestResponseContent:
    """Tests for response content structure"""

    def test_cancel_response_structure(self):
        """Test that cancel response has correct structure"""
        mock_service = MagicMock()
        mock_service.cancel_job = MagicMock(
            return_value={
                "id": "job_123",
                "object": JobTypes.VIDEO.value,
                "status": "cancelling",
                "created_at": 1000,
            }
        )

        response = cancel_video_job(
            job_id="job_123",
            service=mock_service,
            api_key="test_key",
        )

        # JSONResponse body contains the serialized content
        import json

        content = json.loads(response.body)

        assert content["id"] == "job_123"
        assert content["object"] == JobTypes.VIDEO.value
        assert content["status"] == "cancelling"


class TestSubmitGenerateVideoI2VRequest:
    """Tests for POST /generations/i2v endpoint and VideoI2VGenerateRequest validation."""

    @pytest.mark.asyncio
    async def test_submit_i2v_request_success(self):
        """I2V job submission reuses the same create_job path as T2V."""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            return_value={
                "id": "job_i2v_1",
                "object": "video",
                "status": "pending",
                "created_at": 1234567890,
            }
        )

        request = VideoI2VGenerateRequest(
            prompt="A cat on a hill",
            image_prompts=[
                ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0),
            ],
        )

        response = await submit_generate_video_i2v_request(
            request=request,
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 202
        mock_service.create_job.assert_called_once_with(JobTypes.VIDEO, request)

    @pytest.mark.asyncio
    async def test_submit_i2v_request_multiple_image_prompts(self):
        """Multi-image conditioning is accepted and passed through unchanged."""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            return_value={"id": "job_i2v_2", "status": "pending"}
        )

        b64 = _tiny_png_base64()
        request = VideoI2VGenerateRequest(
            prompt="A cat on a hill",
            image_prompts=[
                ImagePromptEntry(image=b64, frame_pos=0),
                ImagePromptEntry(image=b64, frame_pos=40),
                ImagePromptEntry(image=b64, frame_pos=80),
            ],
        )

        await submit_generate_video_i2v_request(
            request=request,
            service=mock_service,
            api_key="test_key",
        )

        args, _ = mock_service.create_job.call_args
        passed_request = args[1]
        assert len(passed_request.image_prompts) == 3
        assert [e.frame_pos for e in passed_request.image_prompts] == [0, 40, 80]


class TestVideoI2VGenerateRequestValidation:
    """Tests for VideoI2VGenerateRequest pydantic validation."""

    def test_valid_request(self):
        request = VideoI2VGenerateRequest(
            prompt="A cat",
            image_prompts=[ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)],
        )
        assert len(request.image_prompts) == 1
        assert request.image_prompts[0].frame_pos == 0

    def test_missing_image_prompts_rejected(self):
        """A request without image_prompts is not a valid I2V request."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoI2VGenerateRequest(prompt="A cat")

    def test_empty_image_prompts_rejected(self):
        """Upstream would crash on image_prompt=[]; reject at API boundary."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoI2VGenerateRequest(prompt="A cat", image_prompts=[])

    def test_duplicate_frame_pos_rejected(self):
        """Upstream asserts on duplicate frame positions; pre-empt at API."""
        from pydantic import ValidationError

        b64 = _tiny_png_base64()
        with pytest.raises(ValidationError, match="duplicate frame_pos"):
            VideoI2VGenerateRequest(
                prompt="A cat",
                image_prompts=[
                    ImagePromptEntry(image=b64, frame_pos=5),
                    ImagePromptEntry(image=b64, frame_pos=5),
                ],
            )

    def test_negative_frame_pos_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImagePromptEntry(image=_tiny_png_base64(), frame_pos=-1)

    def test_empty_base64_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImagePromptEntry(image="", frame_pos=0)

    def test_frame_pos_out_of_range_rejected(self):
        """frame_pos must be < WAN22_NUM_FRAMES (81); upstream
        would otherwise raise IndexError writing into a 81-slot tensor."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImagePromptEntry(image=_tiny_png_base64(), frame_pos=81)

    def test_too_many_image_prompts_rejected(self):
        """List length is capped at num_frames (81): one conditioning image
        per output frame is the upstream hard limit."""
        from pydantic import ValidationError

        b64 = _tiny_png_base64()
        # 82 entries exceeds WAN22_NUM_FRAMES, and the only valid
        # frame_pos values are [0, 80] so this list is invalid on two axes.
        with pytest.raises(ValidationError):
            VideoI2VGenerateRequest(
                prompt="A cat",
                image_prompts=[
                    ImagePromptEntry(image=b64, frame_pos=i % 81) for i in range(82)
                ],
            )

    def test_max_valid_image_prompts_accepted(self):
        """81 distinct frame positions, one per frame, is the upstream max."""
        b64 = _tiny_png_base64()
        request = VideoI2VGenerateRequest(
            prompt="A cat",
            image_prompts=[ImagePromptEntry(image=b64, frame_pos=i) for i in range(81)],
        )
        assert len(request.image_prompts) == 81

    def test_inherits_video_generate_request_fields(self):
        """I2V subclasses T2V: all T2V fields remain usable."""
        request = VideoI2VGenerateRequest(
            prompt="A cat",
            negative_prompt="blurry",
            num_inference_steps=30,
            seed=42,
            image_prompts=[ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)],
        )
        assert request.negative_prompt == "blurry"
        assert request.num_inference_steps == 30
        assert request.seed == 42


# Valid base64, 804 bytes of payload, not an image — the size and shape of the
# conditioning image in #4811's reproduction curl.
_NOT_AN_IMAGE_BASE64 = base64.b64encode(b"x" * 804).decode("ascii")


class TestRogueImagePayloadsRejectedAtTheBoundary:
    """Undecodable conditioning images must not reach the device (#4811).

    PR #4817 stopped text-only requests on an I2V deployment, but a request that
    *does* carry ``image_prompts`` was only checked for string length. Anything
    that is not a decodable image then failed inside the worker, where it counted
    against ``error_count`` and cost the worker a restart every 6 requests.
    """

    @pytest.mark.parametrize(
        "payload,why",
        [
            ("!!!not-base64!!!", "not base64 at all"),
            ("data:image/png;base64,!!!!", "data URL wrapping non-base64"),
            (_NOT_AN_IMAGE_BASE64, "valid base64, bytes are not an image"),
            (
                f"data:image/png;base64,{_NOT_AN_IMAGE_BASE64}",
                "png content type, non-image bytes (the issue's payload)",
            ),
            (base64.b64encode(b"%PDF-1.7 not an image").decode("ascii"), "a PDF"),
        ],
    )
    def test_rogue_image_payloads_are_rejected(self, payload, why):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImagePromptEntry(image=payload, frame_pos=0)

    @pytest.mark.parametrize(
        "header",
        [
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
            b"\xff\xd8\xff\xe0" + b"\x00" * 32,  # JPEG
            b"GIF89a" + b"\x00" * 32,
            b"BM" + b"\x00" * 32,  # BMP
            b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32,
        ],
    )
    def test_real_image_headers_are_accepted(self, header):
        """The boundary check is a header sniff, not a full decode: it must not
        start 422-ing formats the pipeline can actually read."""
        entry = ImagePromptEntry(
            image=base64.b64encode(header).decode("ascii"), frame_pos=0
        )
        assert entry.frame_pos == 0

    def test_a_real_png_is_still_accepted(self):
        """Regression guard: the happy path must not move."""
        entry = ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)
        assert entry.image == _tiny_png_base64()

    def test_padding_stripped_base64_is_still_accepted(self):
        """HTTP/JSON transports strip ``=`` padding. The boundary check has to
        restore it exactly like ``ImageManager.base64_to_pil_image`` does, or it
        would reject images the runner can decode fine."""
        entry = ImagePromptEntry(image=_tiny_png_base64().rstrip("="), frame_pos=0)
        assert entry.image

    def test_line_wrapped_base64_is_still_accepted(self):
        """MIME-style wrapping at 76 columns is what the ``base64`` CLI produces.
        The boundary must tolerate it, or a payload the runner decodes fine would
        be rejected as malformed."""
        b64 = _tiny_png_base64()
        wrapped = "\n".join([b64[:40], b64[40:]])

        entry = ImagePromptEntry(image=wrapped, frame_pos=0)
        assert entry.image == wrapped

    @pytest.mark.asyncio
    async def test_rogue_request_never_reaches_the_service(self):
        """The whole point: rejected during request parsing, so the scheduler and
        the device worker never see it."""
        from pydantic import ValidationError

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock()

        with pytest.raises(ValidationError):
            VideoI2VGenerateRequest(
                prompt="A serene mountain landscape with flowing water",
                num_inference_steps=12,
                seed=42,
                image_prompts=[{"image": _NOT_AN_IMAGE_BASE64, "frame_pos": 0}],
            )

        mock_service.create_job.assert_not_called()


class TestRogueSeedRejectedAtTheBoundary:
    """``seed`` is packed as a signed 64-bit int in SHM (``ipc/video_shm.py``).

    Out-of-range values used to be accepted by the API and then raise
    ``struct.error`` inside the device worker — a different rogue-parameter
    signature from the image case, with the same #4811 restart consequence.
    """

    @pytest.mark.parametrize("seed", [2**63, 2**64, -(2**63) - 1, 10**30])
    def test_seed_outside_int64_is_rejected(self, seed):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenerateRequest(prompt="A cat", seed=seed)

    @pytest.mark.parametrize("seed", [0, 42, 2**63 - 1, -(2**63), None])
    def test_seed_inside_int64_is_accepted(self, seed):
        assert VideoGenerateRequest(prompt="A cat", seed=seed).seed == seed

    def test_the_rejected_range_is_exactly_what_shm_cannot_pack(self):
        """Pins the bound to its reason: the boundary rejects precisely the
        values that would have blown up in ``VideoShm.write_request``."""
        import struct

        with pytest.raises(struct.error):
            struct.pack("<q", 2**63)

        struct.pack("<q", 2**63 - 1)  # in range, must not raise


class TestI2VUploadRogueBytes:
    """``/generations/i2v/upload`` builds the DTO in code, so a validation error
    there would surface as a 500 unless the endpoint translates it."""

    @pytest.mark.asyncio
    async def test_png_content_type_with_garbage_bytes_is_a_422(self):
        upload = MagicMock()
        upload.content_type = "image/png"
        upload.read = AsyncMock(side_effect=[b"x" * 804, b""])

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await submit_generate_video_i2v_upload(
                prompt="A cat",
                image=upload,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 422
        mock_service.create_job.assert_not_called()


class TestRogueRequestOverHttp:
    """The status the client actually receives, through a real FastAPI app.

    The DTO tests above pin validation; this pins the contract #4811 complained
    about — a rogue body must come back 4xx, not ``500 Internal Server Error``,
    and must not reach the service.
    """

    @pytest.fixture
    def mock_service(self):
        service = MagicMock()
        service.scheduler = MagicMock()
        service.scheduler.check_is_model_ready = MagicMock()
        service.create_job = AsyncMock(
            return_value={"id": "job_1", "object": "video", "status": "pending"}
        )
        return service

    @pytest.fixture
    def client(self, mock_service):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from open_ai_api.video import router as video_router
        from resolver.service_resolver import service_resolver
        from security.api_key_checker import get_api_key

        app = FastAPI()
        app.include_router(video_router, prefix="/v1/videos")
        app.dependency_overrides[service_resolver] = lambda: mock_service
        app.dependency_overrides[get_api_key] = lambda: "test-key"
        return TestClient(app)

    def test_the_issues_payload_gets_a_422(self, client, mock_service):
        """Body copied from #4811's reproduction curl, with its 804-byte
        non-image conditioning payload."""
        response = client.post(
            "/v1/videos/generations/i2v",
            json={
                "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "prompt": "A serene mountain landscape with flowing water",
                "num_inference_steps": 12,
                "seed": 42,
                "image_prompts": [
                    {
                        "image": f"data:image/png;base64,{_NOT_AN_IMAGE_BASE64}",
                        "frame_pos": 0,
                    }
                ],
            },
        )

        assert response.status_code == 422
        assert "image" in response.text
        mock_service.create_job.assert_not_called()

    def test_out_of_range_seed_gets_a_422(self, client, mock_service):
        response = client.post(
            "/v1/videos/generations/i2v",
            json={
                "prompt": "A cat",
                "seed": 2**63,
                "image_prompts": [{"image": _tiny_png_base64(), "frame_pos": 0}],
            },
        )

        assert response.status_code == 422
        mock_service.create_job.assert_not_called()

    @patch("open_ai_api.video.settings")
    def test_a_valid_i2v_request_still_reaches_the_service(
        self, mock_settings, client, mock_service
    ):
        """Regression guard: the boundary checks must not shut the door on real
        traffic."""
        mock_settings.use_async_video = True
        mock_settings.model_runner = "tt-wan2.2-i2v"

        response = client.post(
            "/v1/videos/generations/i2v",
            json={
                "prompt": "A serene mountain landscape with flowing water",
                "num_inference_steps": 12,
                "seed": 42,
                "image_prompts": [{"image": _tiny_png_base64(), "frame_pos": 0}],
            },
        )

        assert response.status_code == 202
        mock_service.create_job.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

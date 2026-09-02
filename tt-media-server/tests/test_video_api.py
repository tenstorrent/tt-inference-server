# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.constants import (
    DEFAULT_VIDEO_INFERENCE_STEPS,
    JobTypes,
    MAX_VIDEO_INFERENCE_STEPS,
    MIN_VIDEO_INFERENCE_STEPS,
)
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from fastapi import HTTPException
from open_ai_api.video import (
    _is_i2v_only_deployment,
    _sp_peer_is_known_non_ref2va,
    cancel_video_job,
    delete_video_job,
    download_video_content,
    get_jobs_metadata,
    get_video_metadata,
    reject_ref2va_on_wrong_deployment,
    reject_text_to_video_on_i2v_deployment,
    submit_generate_video_i2v_request,
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

    @pytest.mark.asyncio
    async def test_submit_generate_video_request_queue_full_returns_429(self):
        """Issue #4959: queue-full admission must surface as 429, not 500."""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            side_effect=HTTPException(
                status_code=429, detail="Task queue is full. Please try again later."
            )
        )
        request = VideoGenerateRequest(prompt="Test video")

        with pytest.raises(HTTPException) as exc_info:
            await submit_generate_video_request(
                request=request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 429
        assert "Task queue is full" in exc_info.value.detail


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
    """Tests for POST /generations/{job_id}/cancel endpoint"""

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


class TestRejectRef2vaOnWrongDeployment:
    """/generations/ref2va must refuse only a deployment PROVEN not to serve it.

    An SP frontend loads no weights, so MODEL is its only evidence about the
    peer's task -- and MODEL is advisory there. Recognising any known model name
    would 422 a working ref2va deployment whose MODEL happens to name something
    else, so only the H3 T2VA/FL2VA names may trigger the refusal.
    """

    @staticmethod
    def _refuses():
        try:
            reject_ref2va_on_wrong_deployment()
        except HTTPException as e:
            return e.status_code
        return None

    @patch("open_ai_api.video.settings.model_runner", "tt-minimax-h3-ref2va")
    def test_in_process_ref2va_runner_is_allowed(self):
        assert self._refuses() is None

    @patch("open_ai_api.video.settings.model_runner", "tt-minimax-h3-t2va")
    def test_in_process_t2va_runner_is_refused(self):
        assert self._refuses() == 422

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3-Ref2VA"})
    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_for_a_ref2va_peer_is_allowed(self):
        assert self._refuses() is None

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3"})
    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_naming_t2va_is_refused(self):
        assert self._refuses() == 422

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3-FL2VA"})
    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_naming_fl2va_is_refused(self):
        assert self._refuses() == 422

    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_without_model_stays_permissive(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL", None)
            assert self._refuses() is None

    @patch.dict(os.environ, {"MODEL": "Wan2.2-T2V-A14B-Diffusers"})
    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_naming_an_unrelated_model_stays_permissive(self):
        """The regression guard: a known name that says nothing about the H3
        task must not refuse a peer that may well be serving Ref2VA."""
        assert self._refuses() is None

    @patch.dict(os.environ, {"MODEL": "not-a-model-name"})
    @patch("open_ai_api.video.settings.model_runner", "sp_runner")
    def test_sp_frontend_with_an_unknown_model_stays_permissive(self):
        assert self._refuses() is None


class TestSpPeerIsKnownNonRef2va:
    """The helper must recognise exactly the two H3 non-Ref2VA task names."""

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3"})
    def test_t2va_name(self):
        assert _sp_peer_is_known_non_ref2va() is True

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3-FL2VA"})
    def test_fl2va_name(self):
        assert _sp_peer_is_known_non_ref2va() is True

    @patch.dict(os.environ, {"MODEL": "MiniMax-H3-Ref2VA"})
    def test_ref2va_name_is_not_a_non_ref2va_model(self):
        assert _sp_peer_is_known_non_ref2va() is False

    @patch.dict(os.environ, {"MODEL": "Wan2.2-T2V-A14B-Diffusers"})
    def test_a_known_but_unrelated_model_is_inconclusive(self):
        assert _sp_peer_is_known_non_ref2va() is False

    @patch.dict(os.environ, {"MODEL": "garbage"})
    def test_unknown_string_is_inconclusive(self):
        assert _sp_peer_is_known_non_ref2va() is False

    def test_unset_model_is_inconclusive(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL", None)
            assert _sp_peer_is_known_non_ref2va() is False


class TestDownloadTempFileCleanup:
    """The faststart copy is a per-response artefact and must not accumulate."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from open_ai_api.video import router
        from resolver.service_resolver import service_resolver
        from security.api_key_checker import get_api_key

        self.service = MagicMock()
        app = FastAPI()
        app.include_router(router, prefix="/v1/videos")
        app.dependency_overrides[service_resolver] = lambda: self.service
        app.dependency_overrides[get_api_key] = lambda: "test-key"
        return TestClient(app)

    def test_remuxed_copy_is_removed_after_the_response(self, client, tmp_path):
        src = tmp_path / "result.mp4"
        src.write_bytes(b"original bytes")
        self.service.get_job_result_path.return_value = str(src)
        made = []

        def fake_faststart(inp, out):
            made.append(out)
            with open(out, "wb") as f:
                f.write(b"remuxed bytes")

        with patch(
            "open_ai_api.video.VideoManager.ensure_faststart",
            side_effect=fake_faststart,
        ):
            response = client.get("/v1/videos/generations/job_123/download")

        assert response.status_code == 200
        assert response.content == b"remuxed bytes"  # the remux was served
        assert len(made) == 1
        assert not os.path.exists(made[0]), "faststart temp copy leaked"
        assert src.exists(), "the job's own result must survive a download"

    def test_temp_file_is_removed_when_the_remux_fails(self, client, tmp_path):
        src = tmp_path / "result.mp4"
        src.write_bytes(b"original bytes")
        self.service.get_job_result_path.return_value = str(src)
        made = []

        def failing_faststart(inp, out):
            made.append(out)
            raise RuntimeError("ffmpeg exploded")

        with patch(
            "open_ai_api.video.VideoManager.ensure_faststart",
            side_effect=failing_faststart,
        ):
            response = client.get("/v1/videos/generations/job_123/download")

        assert response.status_code == 200
        assert response.content == b"original bytes"  # fell back to the original
        assert len(made) == 1
        assert not os.path.exists(made[0]), "empty temp file leaked on the failure path"


class TestDeleteVideoJob:
    """Tests for DELETE /generations/{job_id} endpoint"""

    def test_route_is_registered_as_delete(self):
        """The same path serves GET (metadata) and DELETE (remove)."""
        from open_ai_api.video import router

        methods = {
            m
            for r in router.routes
            if getattr(r, "path", None) == "/generations/{job_id}"
            for m in r.methods
        }
        assert {"GET", "DELETE"} <= methods

    def test_delete_video_job_success(self):
        """A finished job is deleted and the OpenAI-style receipt is returned."""
        mock_service = MagicMock()
        mock_service.delete_job = MagicMock(
            return_value={
                "id": "job_123",
                "job_type": JobTypes.VIDEO.value,
                "status": "completed",
                "created_at": 1000,
            }
        )

        response = delete_video_job(
            job_id="job_123",
            service=mock_service,
            api_key="test_key",
        )

        assert response.status_code == 200
        assert json.loads(response.body) == {
            "id": "job_123",
            "object": JobTypes.VIDEO.value,
            "deleted": True,
        }
        mock_service.delete_job.assert_called_once_with("job_123")

    def test_delete_video_job_not_found(self):
        mock_service = MagicMock()
        mock_service.delete_job = MagicMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            delete_video_job(
                job_id="non_existent_job",
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Video job not found"

    def test_delete_video_job_active_conflict_propagates(self):
        """The manager's 409 for an unfinished job reaches the client unchanged."""
        mock_service = MagicMock()
        mock_service.delete_job = MagicMock(
            side_effect=HTTPException(status_code=409, detail="Job is in_progress")
        )

        with pytest.raises(HTTPException) as exc_info:
            delete_video_job(
                job_id="job_123",
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 409


class TestDeleteVideoJobHTTP:
    """DELETE through the real router, under both the /v1 and the legacy prefix."""

    @pytest.fixture
    def mock_service(self):
        return MagicMock()

    @pytest.fixture
    def client(self, mock_service):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from open_ai_api.video import router
        from resolver.service_resolver import service_resolver
        from security.api_key_checker import get_api_key

        app = FastAPI()
        # Same shape as open_ai_api.__init__: primary /v1 prefix + deprecated alias.
        app.include_router(router, prefix="/v1/videos")
        app.include_router(router, prefix="/video", deprecated=True)
        app.dependency_overrides[service_resolver] = lambda: mock_service
        app.dependency_overrides[get_api_key] = lambda: "test-key"
        return TestClient(app)

    @pytest.mark.parametrize("prefix", ["/v1/videos", "/video"])
    def test_delete_finished_job(self, client, mock_service, prefix):
        mock_service.delete_job.return_value = {"id": "job_123", "status": "completed"}

        response = client.delete(f"{prefix}/generations/job_123")

        assert response.status_code == 200
        assert response.json() == {
            "id": "job_123",
            "object": JobTypes.VIDEO.value,
            "deleted": True,
        }
        mock_service.delete_job.assert_called_once_with("job_123")

    def test_delete_unknown_job_is_404(self, client, mock_service):
        mock_service.delete_job.return_value = None

        response = client.delete("/v1/videos/generations/nope")

        assert response.status_code == 404
        assert response.json()["detail"] == "Video job not found"

    def test_delete_active_job_is_409(self, client, mock_service):
        mock_service.delete_job.side_effect = HTTPException(
            status_code=409, detail="Job is in_progress; cancel it first"
        )

        response = client.delete("/v1/videos/generations/job_123")

        assert response.status_code == 409
        assert "in_progress" in response.json()["detail"]

    def test_get_on_same_path_still_works(self, client, mock_service):
        """Adding DELETE must not shadow the existing GET on /generations/{job_id}."""
        mock_service.get_job_metadata.return_value = {
            "id": "job_123",
            "status": "queued",
        }

        response = client.get("/v1/videos/generations/job_123")

        assert response.status_code == 200
        assert response.json()["id"] == "job_123"


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

    def test_default_inference_steps(self):
        request = VideoGenerateRequest(prompt="A cat walking in the park")
        assert request.num_inference_steps == DEFAULT_VIDEO_INFERENCE_STEPS

    def test_min_inference_steps_accepted(self):
        request = VideoGenerateRequest(
            prompt="A cat walking in the park",
            num_inference_steps=MIN_VIDEO_INFERENCE_STEPS,
        )
        assert request.num_inference_steps == MIN_VIDEO_INFERENCE_STEPS

    def test_below_min_inference_steps_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenerateRequest(
                prompt="A cat walking in the park",
                num_inference_steps=MIN_VIDEO_INFERENCE_STEPS - 1,
            )

    def test_max_inference_steps_accepted(self):
        request = VideoGenerateRequest(
            prompt="A cat walking in the park",
            num_inference_steps=MAX_VIDEO_INFERENCE_STEPS,
        )
        assert request.num_inference_steps == MAX_VIDEO_INFERENCE_STEPS

    def test_above_max_inference_steps_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoGenerateRequest(
                prompt="A cat walking in the park",
                num_inference_steps=MAX_VIDEO_INFERENCE_STEPS + 1,
            )


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
    async def test_submit_i2v_request_queue_full_returns_429(self):
        """Issue #4959 repro path: POST /generations/i2v returns 429 when full."""
        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(
            side_effect=HTTPException(
                status_code=429, detail="Task queue is full. Please try again later."
            )
        )
        request = VideoI2VGenerateRequest(
            prompt="A cat on a hill",
            image_prompts=[
                ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0),
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            await submit_generate_video_i2v_request(
                request=request,
                service=mock_service,
                api_key="test_key",
            )

        assert exc_info.value.status_code == 429

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

    def test_negative_one_is_last_frame_sentinel(self):
        """frame_pos=-1 is last-frame (Python list indexing); Wan and FL2VA both use it."""
        entry = ImagePromptEntry(image=_tiny_png_base64(), frame_pos=-1)
        assert entry.frame_pos == -1

    def test_negative_two_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImagePromptEntry(image=_tiny_png_base64(), frame_pos=-2)

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

    def test_inherits_min_inference_steps(self):
        """I2V uses the same API floor as T2V; 4 must not 422."""
        request = VideoI2VGenerateRequest(
            prompt="A cat",
            num_inference_steps=MIN_VIDEO_INFERENCE_STEPS,
            image_prompts=[ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)],
        )
        assert request.num_inference_steps == MIN_VIDEO_INFERENCE_STEPS

    def test_inherits_below_min_inference_steps_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoI2VGenerateRequest(
                prompt="A cat",
                num_inference_steps=MIN_VIDEO_INFERENCE_STEPS - 1,
                image_prompts=[ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)],
            )

    def test_valid_image_with_data_uri_prefix_accepted(self):
        """base64 image with 'data:image/png;base64,' prefix should pass."""
        prefixed = "data:image/png;base64," + _tiny_png_base64()
        entry = ImagePromptEntry(image=prefixed, frame_pos=0)
        assert entry.image == prefixed

    def test_invalid_base64_rejected(self):
        """Garbage string that isn't valid base64 should fail validation."""
        from pydantic import ValidationError

        with pytest.raises(
            ValidationError, match="could not be decoded to a valid PIL image"
        ):
            ImagePromptEntry(image="not-a-real-image!!!", frame_pos=0)

    def test_valid_base64_non_image_rejected(self):
        """Valid base64 encoding of non-image bytes should fail validation."""
        import base64

        from pydantic import ValidationError

        fake = base64.b64encode(b"hello world this is not an image").decode()
        with pytest.raises(
            ValidationError, match="could not be decoded to a valid PIL image"
        ):
            ImagePromptEntry(image=fake, frame_pos=0)

    def test_raw_base64_png_accepted(self):
        """Raw base64 PNG without data URI prefix should pass."""
        entry = ImagePromptEntry(image=_tiny_png_base64(), frame_pos=0)
        assert entry.frame_pos == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

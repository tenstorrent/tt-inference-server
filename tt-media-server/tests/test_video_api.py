# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.constants import JobTypes
from domain.video_generate_request import VideoGenerateRequest
from open_ai_api.video import (
    cancel_video_job,
    download_video_content,
    get_jobs_metadata,
    get_video_metadata,
    submit_generate_video_request,
)


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
        mock_service.get_job_result_path = MagicMock(return_value="/nonexistent/path.mp4")

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
        mock_service.get_job_result_path = MagicMock(return_value={"error": "not a path"})

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
        mock_service.cancel_job = MagicMock(return_value=True)

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
        mock_service.cancel_job = MagicMock(return_value=False)

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
        mock_service.cancel_job = MagicMock(return_value=True)

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
        assert content["cancelled"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

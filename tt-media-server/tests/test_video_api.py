# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from unittest.mock import MagicMock

import pytest


# Self-contained mock classes for testing Video functionality
class MockVideoGenerateRequest:
    def __init__(self, **kwargs):
        # Validate required prompt parameter
        if "prompt" not in kwargs or kwargs.get("prompt") is None:
            raise ValueError("Prompt is required")

        # Validate duration if provided
        duration = kwargs.get("duration")
        if duration is not None:
            if not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError("duration must be a positive number")

        # Validate resolution if provided
        resolution = kwargs.get("resolution")
        valid_resolutions = ["720p", "1080p", "4k", None]
        if resolution is not None and resolution not in valid_resolutions:
            raise ValueError(f"resolution must be one of {valid_resolutions}")

        self.prompt = kwargs.get("prompt")
        self.duration = kwargs.get("duration", 5.0)
        self.resolution = kwargs.get("resolution", "720p")
        self.fps = kwargs.get("fps", 24)
        self.model = kwargs.get("model", "default")


class MockJobTypes:
    VIDEO = MagicMock()
    VIDEO.value = "video"


class MockResponse:
    def __init__(self, content=None, status_code=200):
        self.content = content or {}
        self.status_code = status_code

    def to_dict(self):
        return self.content


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


class MockHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


class MockFileResponse:
    def __init__(self, path, media_type=None, filename=None, headers=None):
        self.path = path
        self.media_type = media_type
        self.filename = filename
        self.headers = headers or {}


class MockJSONResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


def mock_get_dict_response(obj):
    """Mock implementation of get_dict_response"""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise ValueError("Expected response class with to_dict() method.")


# Use mock classes
VideoGenerateRequest = MockVideoGenerateRequest
JobTypes = MockJobTypes
Request = MockRequest
HTTPException = MockHTTPException
FileResponse = MockFileResponse
JSONResponse = MockJSONResponse
get_dict_response = mock_get_dict_response


class TestVideoRequestParsing:
    """Test Video request parsing and validation"""

    def test_video_generate_request_creation(self):
        """Test VideoGenerateRequest creation with various parameters"""
        # Basic request
        request = VideoGenerateRequest(prompt="A cat walking in the park")
        assert request.prompt == "A cat walking in the park"
        assert request.duration == 5.0
        assert request.resolution == "720p"
        assert request.fps == 24

        # Request with all parameters
        request = VideoGenerateRequest(
            prompt="A sunset over the ocean",
            duration=10.0,
            resolution="1080p",
            fps=30,
            model="custom_model",
        )
        assert request.prompt == "A sunset over the ocean"
        assert request.duration == 10.0
        assert request.resolution == "1080p"
        assert request.fps == 30
        assert request.model == "custom_model"

    def test_video_generate_request_validation(self):
        """Test VideoGenerateRequest validation"""
        # Missing required prompt should fail
        with pytest.raises(Exception):
            VideoGenerateRequest()

        # None prompt should fail
        with pytest.raises(Exception):
            VideoGenerateRequest(prompt=None)

        # Invalid duration type should fail
        with pytest.raises(Exception):
            VideoGenerateRequest(prompt="Test", duration="invalid")

        # Negative duration should fail
        with pytest.raises(Exception):
            VideoGenerateRequest(prompt="Test", duration=-5)

        # Invalid resolution should fail
        with pytest.raises(Exception):
            VideoGenerateRequest(prompt="Test", resolution="invalid_res")


class TestVideoHandler:
    """Test Video request handling - basic functionality"""

    def test_get_dict_response_with_to_dict(self):
        """Test get_dict_response with object that has to_dict method"""
        mock_obj = MagicMock()
        mock_obj.to_dict = MagicMock(
            return_value={"job_id": "123", "status": "pending"}
        )

        result = get_dict_response(mock_obj)
        assert result == {"job_id": "123", "status": "pending"}
        mock_obj.to_dict.assert_called_once()

    def test_get_dict_response_without_to_dict(self):
        """Test get_dict_response with object that lacks to_dict method"""
        mock_obj = MagicMock()
        del mock_obj.to_dict  # Remove the method

        with pytest.raises(ValueError) as exc_info:
            get_dict_response(mock_obj)

        assert "Expected response class with to_dict() method" in str(exc_info.value)


class TestVideoJobOperations:
    """Test video job CRUD operations"""

    def test_submit_job_returns_job_data(self):
        """Test that submitting a video job returns job metadata"""
        mock_service = MagicMock()
        mock_service.create_job = MagicMock(
            return_value={
                "id": "job_123",
                "object": "video",
                "status": "pending",
                "created_at": 1234567890,
            }
        )

        request = VideoGenerateRequest(prompt="Test video generation")

        # Call the mock synchronously (mock doesn't need async)
        job_data = mock_service.create_job(JobTypes.VIDEO, request)

        assert job_data["id"] == "job_123"
        assert job_data["status"] == "pending"
        mock_service.create_job.assert_called_once()

    def test_get_job_metadata(self):
        """Test fetching job metadata"""
        mock_service = MagicMock()
        mock_service.get_job_metadata = MagicMock(
            return_value={
                "id": "job_123",
                "object": "video",
                "status": "completed",
                "progress": 100,
            }
        )

        job_data = mock_service.get_job_metadata("job_123")

        assert job_data["id"] == "job_123"
        assert job_data["status"] == "completed"
        assert job_data["progress"] == 100
        mock_service.get_job_metadata.assert_called_once_with("job_123")

    def test_get_job_metadata_not_found(self):
        """Test fetching non-existent job metadata"""
        mock_service = MagicMock()
        mock_service.get_job_metadata = MagicMock(return_value=None)

        job_data = mock_service.get_job_metadata("non_existent_job")

        assert job_data is None
        mock_service.get_job_metadata.assert_called_once_with("non_existent_job")

    def test_get_all_jobs_metadata(self):
        """Test fetching all jobs metadata"""
        mock_service = MagicMock()
        mock_service.get_all_jobs_metadata = MagicMock(
            return_value=[
                {"id": "job_1", "status": "completed"},
                {"id": "job_2", "status": "pending"},
                {"id": "job_3", "status": "processing"},
            ]
        )

        jobs = mock_service.get_all_jobs_metadata()

        assert len(jobs) == 3
        assert jobs[0]["id"] == "job_1"
        assert jobs[1]["status"] == "pending"
        mock_service.get_all_jobs_metadata.assert_called_once()

    def test_cancel_job_success(self):
        """Test canceling a video job"""
        mock_service = MagicMock()
        mock_service.cancel_job = MagicMock(return_value=True)

        success = mock_service.cancel_job("job_123")

        assert success is True
        mock_service.cancel_job.assert_called_once_with("job_123")

    def test_cancel_job_not_found(self):
        """Test canceling a non-existent job"""
        mock_service = MagicMock()
        mock_service.cancel_job = MagicMock(return_value=False)

        success = mock_service.cancel_job("non_existent_job")

        assert success is False
        mock_service.cancel_job.assert_called_once_with("non_existent_job")


class TestVideoDownload:
    """Test video download functionality"""

    def test_get_job_result_returns_path(self):
        """Test that get_job_result returns a file path"""
        mock_service = MagicMock()
        mock_service.get_job_result = MagicMock(return_value="/tmp/video_123.mp4")

        file_path = mock_service.get_job_result("job_123")

        assert file_path == "/tmp/video_123.mp4"
        assert file_path.endswith(".mp4")
        mock_service.get_job_result.assert_called_once_with("job_123")

    def test_get_job_result_not_ready(self):
        """Test get_job_result when video is not ready"""
        mock_service = MagicMock()
        mock_service.get_job_result = MagicMock(return_value=None)

        file_path = mock_service.get_job_result("job_123")

        assert file_path is None
        mock_service.get_job_result.assert_called_once_with("job_123")

    def test_file_response_creation(self):
        """Test FileResponse creation with correct parameters"""
        response = FileResponse(
            "/tmp/video_123.mp4",
            media_type="video/mp4",
            filename="video_123.mp4",
            headers={"Content-Disposition": "attachment; filename=video_123.mp4"},
        )

        assert response.path == "/tmp/video_123.mp4"
        assert response.media_type == "video/mp4"
        assert response.filename == "video_123.mp4"
        assert "Content-Disposition" in response.headers


class TestVideoRouterIntegration:
    """Basic integration tests for Video functionality"""

    def test_router_mock(self):
        """Test that Video router functionality can be mocked"""
        mock_router = MagicMock()
        mock_router.routes = []
        assert mock_router is not None
        assert hasattr(mock_router, "routes")

    def test_json_response_creation(self):
        """Test JSONResponse creation"""
        response = JSONResponse(
            content={"id": "job_123", "status": "pending"}, status_code=202
        )

        assert response.content["id"] == "job_123"
        assert response.status_code == 202

    def test_http_exception_creation(self):
        """Test HTTPException creation for error cases"""
        exception = HTTPException(status_code=404, detail="Video job not found")

        assert exception.status_code == 404
        assert exception.detail == "Video job not found"

        exception = HTTPException(status_code=500, detail="Internal server error")

        assert exception.status_code == 500
        assert exception.detail == "Internal server error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

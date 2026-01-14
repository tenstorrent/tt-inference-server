# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for CNN API endpoints."""

import base64
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock modules before importing the module under test
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()


class TestParseCNNRequest:
    """Tests for _parse_image_search_request function."""

    @pytest.fixture
    def mock_request_json(self):
        """Create a mock request with JSON content-type."""
        request = AsyncMock()
        request.headers = {"content-type": "application/json"}
        return request

    @pytest.fixture
    def mock_request_multipart(self):
        """Create a mock request with multipart content-type."""
        request = AsyncMock()
        request.headers = {"content-type": "multipart/form-data"}
        return request

    @pytest.fixture
    def valid_base64_image(self):
        """Create a valid base64 encoded image string."""
        # Simple 1x1 PNG image
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return base64.b64encode(png_bytes).decode("utf-8")

    @pytest.mark.asyncio
    async def test_parse_file_upload_creates_image_search_request(
        self, mock_request_multipart
    ):
        """Test that file upload is correctly parsed into ImageSearchRequest."""
        from open_ai_api.cnn import _parse_image_search_request

        # Create mock file
        file_content = b"fake image content for testing"
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=file_content)

        result = await _parse_image_search_request(
            request=mock_request_multipart,
            file=mock_file,
            response_format="json",
            top_k=3,
            min_confidence=70.0,
        )

        assert result.prompt == file_content
        assert result.top_k == 3
        assert result.min_confidence == 70.0
        assert result.response_format == "json"
        mock_file.read.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_request_raises_http_exception_when_no_file_or_json(
        self, mock_request_multipart
    ):
        """Test that HTTPException is raised when neither file nor JSON is provided."""
        from fastapi import HTTPException
        from open_ai_api.cnn import _parse_image_search_request

        with pytest.raises(HTTPException) as exc_info:
            await _parse_image_search_request(request=mock_request_multipart, file=None)

        assert exc_info.value.status_code == 400
        assert "multipart/form-data" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_parse_file_upload_with_custom_params(self, mock_request_multipart):
        """Test file upload with custom top_k and min_confidence."""
        from open_ai_api.cnn import _parse_image_search_request

        file_content = b"test image bytes"
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=file_content)

        result = await _parse_image_search_request(
            request=mock_request_multipart,
            file=mock_file,
            response_format="verbose_json",
            top_k=10,
            min_confidence=90.0,
        )

        assert result.top_k == 10
        assert result.min_confidence == 90.0
        assert result.response_format == "verbose_json"


class TestSearchImageEndpoint:
    """Tests for searchImage endpoint."""

    @pytest.fixture
    def mock_image_search_request(self):
        """Create a mock ImageSearchRequest."""
        request = MagicMock()
        request.prompt = b"fake image bytes"
        request.top_k = 3
        request.min_confidence = 70.0
        request.response_format = "json"
        return request

    @pytest.fixture
    def mock_service(self):
        """Create a mock BaseService."""
        service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_search_image_returns_success_response(
        self, mock_image_search_request, mock_service
    ):
        """Test that searchImage returns correct success response."""
        from open_ai_api.cnn import searchImage

        # Service returns List[List[ImagePrediction]] structure
        expected_result = [[{"object": "cat", "confidence_level": 95.5}]]
        mock_service.process_request = AsyncMock(return_value=expected_result)

        result = await searchImage(
            image_search_request=mock_image_search_request,
            service=mock_service,
            api_key="test-api-key",
        )

        assert result.status == "success"
        # Pydantic converts dicts to ImagePrediction objects, compare underlying data
        assert len(result.image_data) == 1
        assert len(result.image_data[0]) == 1
        assert result.image_data[0][0].object == "cat"
        assert result.image_data[0][0].confidence_level == 95.5
        mock_service.process_request.assert_called_once_with(mock_image_search_request)

    @pytest.mark.asyncio
    async def test_search_image_raises_http_exception_on_service_error(
        self, mock_image_search_request, mock_service
    ):
        """Test that searchImage raises HTTPException when service fails."""
        from fastapi import HTTPException
        from open_ai_api.cnn import searchImage

        mock_service.process_request = AsyncMock(
            side_effect=Exception("Model inference failed")
        )

        with pytest.raises(HTTPException) as exc_info:
            await searchImage(
                image_search_request=mock_image_search_request,
                service=mock_service,
                api_key="test-api-key",
            )

        assert exc_info.value.status_code == 500
        assert "Model inference failed" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_search_image_with_verbose_json_format(
        self, mock_image_search_request, mock_service
    ):
        """Test searchImage with verbose_json response format."""
        from open_ai_api.cnn import searchImage

        mock_image_search_request.response_format = "verbose_json"
        # Service returns List[str] structure for verbose format
        expected_result = ["cat,95.5,dog,80.2"]
        mock_service.process_request = AsyncMock(return_value=expected_result)

        result = await searchImage(
            image_search_request=mock_image_search_request,
            service=mock_service,
            api_key="test-api-key",
        )

        assert result.status == "success"
        assert result.image_data == expected_result

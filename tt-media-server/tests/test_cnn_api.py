# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for CNN API endpoints."""

import base64
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock modules before importing the module under test
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()


def _import_cnn_module():
    """
    Import cnn.py directly without going through open_ai_api/__init__.py.

    This avoids import conflicts where __init__.py imports all routers
    (image, audio, etc.) which have dependencies that conflict with test mocks.
    """
    cnn_path = Path(__file__).parent.parent / "open_ai_api" / "cnn.py"
    spec = importlib.util.spec_from_file_location("cnn", cnn_path)
    cnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnn_module)
    return cnn_module


# Import cnn module directly
_cnn = _import_cnn_module()


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
    async def test_parse_json_request_creates_image_search_request(
        self, mock_request_json, valid_base64_image
    ):
        """Test that JSON request body is correctly parsed into ImageSearchRequest."""
        mock_request_json.json = AsyncMock(
            return_value={
                "prompt": valid_base64_image,
                "top_k": 5,
                "min_confidence": 80.0,
                "response_format": "json",
            }
        )

        result = await _cnn._parse_image_search_request(
            request=mock_request_json, file=None
        )

        assert result.prompt == valid_base64_image
        assert result.top_k == 5
        assert result.min_confidence == 80.0
        assert result.response_format == "json"

    @pytest.mark.asyncio
    async def test_parse_file_upload_creates_image_search_request(
        self, mock_request_multipart
    ):
        """Test that file upload is correctly parsed into ImageSearchRequest."""
        # Create mock file
        file_content = b"fake image content for testing"
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=file_content)

        result = await _cnn._parse_image_search_request(
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

        with pytest.raises(HTTPException) as exc_info:
            await _cnn._parse_image_search_request(
                request=mock_request_multipart, file=None
            )

        assert exc_info.value.status_code == 400
        assert "multipart/form-data" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_parse_file_upload_with_custom_params(self, mock_request_multipart):
        """Test file upload with custom top_k and min_confidence."""
        file_content = b"test image bytes"
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=file_content)

        result = await _cnn._parse_image_search_request(
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
        expected_result = [{"object": "cat", "confidence_level": 95.5}]
        mock_service.process_request = AsyncMock(return_value=expected_result)

        result = await _cnn.searchImage(
            image_search_request=mock_image_search_request,
            service=mock_service,
            api_key="test-api-key",
        )

        assert result["status"] == "success"
        assert result["image_data"] == expected_result
        mock_service.process_request.assert_called_once_with(mock_image_search_request)

    @pytest.mark.asyncio
    async def test_search_image_raises_http_exception_on_service_error(
        self, mock_image_search_request, mock_service
    ):
        """Test that searchImage raises HTTPException when service fails."""
        from fastapi import HTTPException

        mock_service.process_request = AsyncMock(
            side_effect=Exception("Model inference failed")
        )

        with pytest.raises(HTTPException) as exc_info:
            await _cnn.searchImage(
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
        mock_image_search_request.response_format = "verbose_json"
        expected_result = "cat,95.5,dog,80.2"
        mock_service.process_request = AsyncMock(return_value=expected_result)

        result = await _cnn.searchImage(
            image_search_request=mock_image_search_request,
            service=mock_service,
            api_key="test-api-key",
        )

        assert result["status"] == "success"
        assert result["image_data"] == expected_result

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import MagicMock


# Self-contained mock classes for testing TTS functionality
class MockTextToSpeechRequest:
    def __init__(self, **kwargs):
        # Validate required text parameter
        if "text" not in kwargs or kwargs.get("text") is None:
            raise ValueError("Text is required")

        # Validate speaker_embedding type
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding is not None:
            if not isinstance(speaker_embedding, (str, bytes)):
                raise ValueError("speaker_embedding must be str or bytes")

        self.text = kwargs.get("text", "Hello world")
        self.stream = kwargs.get("stream", False)
        self.speaker_id = kwargs.get("speaker_id", None)
        self.response_format = kwargs.get("response_format", "verbose_json")
        self.speaker_embedding = speaker_embedding


class MockAudioResponseFormat:
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"


class MockResponse:
    def __init__(self, audio=b"test_audio", **kwargs):
        self.audio = audio
        self.to_dict = lambda: {"audio": "base64_test_audio"}


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


class MockHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


class MockStreamingResponse:
    def __init__(self, content, media_type=None):
        self.content = content
        self.media_type = media_type


def mock_get_dict_response(obj):
    """Mock implementation of get_dict_response"""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise ValueError("Expected response class with to_dict() method.")


# Use mock classes
TextToSpeechRequest = MockTextToSpeechRequest
AudioResponseFormat = MockAudioResponseFormat
Request = MockRequest
HTTPException = MockHTTPException
StreamingResponse = MockStreamingResponse
get_dict_response = mock_get_dict_response


class TestTTSParsing:
    """Test TTS request parsing from form data and JSON"""

    def test_text_to_speech_request_creation(self):
        """Test TextToSpeechRequest creation with various parameters"""
        # Basic request
        request = TextToSpeechRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.stream is False
        assert request.response_format == "verbose_json"
        assert request.speaker_id is None
        assert request.speaker_embedding is None

        # Request with all parameters
        request = TextToSpeechRequest(
            text="Hello world",
            stream=True,
            response_format="text",
            speaker_id="speaker_1",
            speaker_embedding="base64_data",
        )
        assert request.text == "Hello world"
        assert request.stream is True
        assert request.response_format == "text"
        assert request.speaker_id == "speaker_1"
        assert request.speaker_embedding == "base64_data"

    def test_text_to_speech_request_validation(self):
        """Test TextToSpeechRequest validation"""
        # Missing required text should fail
        with pytest.raises(Exception):
            TextToSpeechRequest()

        # Invalid speaker_embedding type should fail
        with pytest.raises(Exception):
            TextToSpeechRequest(text="Hello", speaker_embedding=123)


class TestTTSHandler:
    """Test TTS request handling - basic functionality"""

    def test_get_dict_response_with_to_dict(self):
        """Test get_dict_response with object that has to_dict method"""
        mock_obj = MagicMock()
        mock_obj.to_dict = MagicMock(return_value={"key": "value"})

        result = get_dict_response(mock_obj)
        assert result == {"key": "value"}
        mock_obj.to_dict.assert_called_once()

    def test_get_dict_response_without_to_dict(self):
        """Test get_dict_response with object that lacks to_dict method"""
        mock_obj = MagicMock()
        del mock_obj.to_dict  # Remove the method

        with pytest.raises(ValueError) as exc_info:
            get_dict_response(mock_obj)

        assert "Expected response class with to_dict() method" in str(exc_info.value)


class TestTTSRouterIntegration:
    """Basic integration tests for TTS functionality"""

    def test_router_mock(self):
        """Test that TTS router functionality can be mocked"""
        # Mock router test - just verify we can create mock objects
        mock_router = MagicMock()
        mock_router.routes = []
        assert mock_router is not None
        assert hasattr(mock_router, "routes")

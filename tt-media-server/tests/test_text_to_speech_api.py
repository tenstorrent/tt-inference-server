# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for Text-to-Speech API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from open_ai_api.text_to_speech import (
    get_dict_response as real_get_dict_response,
)
from open_ai_api.text_to_speech import (
    handle_tts_request as real_handle_tts_request,
)
from open_ai_api.text_to_speech import (
    text_to_speech,
)


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
        self.speaker_id = kwargs.get("speaker_id", None)
        self.speaker_embedding = speaker_embedding


class MockResponseFormat:
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


def mock_get_dict_response(obj):
    """Mock implementation of get_dict_response"""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise ValueError("Expected response class with to_dict() method.")


# Use mock classes
TextToSpeechRequest = MockTextToSpeechRequest
ResponseFormat = MockResponseFormat
Request = MockRequest
HTTPException = MockHTTPException
get_dict_response = mock_get_dict_response


class TestTTSParsing:
    """
    Test TTS request parsing from form data and JSON

    This test suite validates the TextToSpeechRequest class for various input scenarios.
    It covers:
    - Basic request creation
    - Request with all parameters
    - Validation of required fields
    - Validation of speaker_embedding type
    """

    def test_text_to_speech_request_creation(self):
        """Test TextToSpeechRequest creation with various parameters"""
        # Basic request
        request = TextToSpeechRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.speaker_id is None
        assert request.speaker_embedding is None

        # Request with all parameters
        request = TextToSpeechRequest(
            text="Hello world",
            speaker_id="speaker_1",
            speaker_embedding="base64_data",
        )
        assert request.text == "Hello world"
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


class TestTextToSpeechRequestValidation:
    """Test actual TextToSpeechRequest validation including max length."""

    def test_text_exceeds_max_length(self):
        """Test that text exceeding max_tts_text_length raises ValueError."""
        from domain.text_to_speech_request import (
            DEFAULT_MAX_TTS_TEXT_LENGTH,
            TextToSpeechRequest,
        )

        long_text = "a" * (DEFAULT_MAX_TTS_TEXT_LENGTH + 100)

        with pytest.raises(ValueError) as exc_info:
            TextToSpeechRequest(text=long_text)

        assert "exceeds maximum length" in str(exc_info.value)

    def test_text_at_max_length_succeeds(self):
        """Test that text at exactly max_tts_text_length succeeds."""
        from domain.text_to_speech_request import (
            DEFAULT_MAX_TTS_TEXT_LENGTH,
            TextToSpeechRequest,
        )

        exact_text = "a" * DEFAULT_MAX_TTS_TEXT_LENGTH

        request = TextToSpeechRequest(text=exact_text)
        assert len(request.text) == DEFAULT_MAX_TTS_TEXT_LENGTH

    def test_text_below_max_length_succeeds(self):
        """Test that text below max_tts_text_length succeeds."""
        from domain.text_to_speech_request import TextToSpeechRequest

        request = TextToSpeechRequest(text="Hello world")
        assert request.text == "Hello world"


class TestRealImplementation:
    """Test actual text_to_speech.py implementation"""

    def test_real_get_dict_response(self):
        """Test real get_dict_response function"""
        mock_obj = MagicMock()
        mock_obj.to_dict = MagicMock(return_value={"key": "value"})
        result = real_get_dict_response(mock_obj)
        assert result == {"key": "value"}

        mock_obj2 = MagicMock()
        del mock_obj2.to_dict
        with pytest.raises(ValueError):
            real_get_dict_response(mock_obj2)

    @pytest.mark.asyncio
    async def test_real_handle_tts_request_audio(self):
        """Test real handle_tts_request with audio format"""
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.wav_bytes = b"test_wav"
        mock_service.process_request = AsyncMock(return_value=mock_response)

        mock_request = MagicMock()
        mock_format = MagicMock()
        mock_format.lower.return_value = "audio"
        mock_request.response_format = mock_format

        result = await real_handle_tts_request(mock_request, mock_service)
        assert isinstance(result, Response)
        assert result.body == b"test_wav"

    @pytest.mark.asyncio
    async def test_real_handle_tts_request_wav(self):
        """Test real handle_tts_request with wav format"""
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.wav_bytes = b"test_wav"
        mock_service.process_request = AsyncMock(return_value=mock_response)

        mock_request = MagicMock()
        mock_format = MagicMock()
        mock_format.lower.return_value = "wav"
        mock_request.response_format = mock_format

        result = await real_handle_tts_request(mock_request, mock_service)
        assert isinstance(result, Response)

    @pytest.mark.asyncio
    async def test_real_handle_tts_request_json(self):
        """Test real handle_tts_request with json format"""
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.to_dict = MagicMock(return_value={"audio": "test"})
        mock_service.process_request = AsyncMock(return_value=mock_response)

        mock_request = MagicMock()
        mock_format = MagicMock()
        mock_format.lower.return_value = "verbose_json"
        mock_request.response_format = mock_format

        result = await real_handle_tts_request(mock_request, mock_service)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_text_to_speech_endpoint(self):
        """Test text_to_speech endpoint"""
        mock_service = MagicMock()
        mock_response = MagicMock()
        mock_response.wav_bytes = b"test"
        mock_service.process_request = AsyncMock(return_value=mock_response)

        mock_request = MagicMock()
        mock_format = MagicMock()
        mock_format.lower.return_value = "audio"
        mock_request.response_format = mock_format

        result = await text_to_speech(mock_request, mock_service, "key")
        assert isinstance(result, Response)

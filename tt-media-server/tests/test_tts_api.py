# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

# Import the TTS router and dependencies
from open_ai_api.tts import tts_router, parse_tts_request, handle_tts_request, get_dict_response
from domain.text_to_speech_request import TextToSpeechRequest
from config.constants import AudioResponseFormat


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
            speaker_embedding="base64_data"
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

    def test_router_import(self):
        """Test that the TTS router can be imported"""
        # This tests that all dependencies are properly set up
        from open_ai_api.tts import router
        assert router is not None
        assert hasattr(router, 'routes')

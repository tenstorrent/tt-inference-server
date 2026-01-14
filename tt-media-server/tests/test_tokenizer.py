# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Tests for tokenizer API endpoints."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# Mock modules before importing
sys.modules["utils.logger"] = MagicMock()

# Mock VLLMSettings before importing domain classes that use it
# The domain classes access VLLMSettings.model.value at class definition time
from config.constants import SupportedModels

mock_model = MagicMock()
mock_model.value = SupportedModels.QWEN_3_4B.value

mock_vllm_settings_class = MagicMock()
mock_vllm_settings_class.model = mock_model

sys.modules["config.vllm_settings"] = MagicMock()
sys.modules["config.vllm_settings"].VLLMSettings = mock_vllm_settings_class

from config.vllm_settings import VLLMSettings
from domain.detokenize_request import DetokenizeRequest
from domain.tokenize_request import TokenizeCompletionRequest
from open_ai_api.tokenizer import _resolve_model, detokenize, tokenize


class TestResolveModel:
    """Tests for _resolve_model function."""

    def test_resolve_model_with_valid_model_string(self):
        """Test that valid model string resolves to SupportedModels enum."""
        model_str = SupportedModels.LLAMA_3_2_3B.value
        result = _resolve_model(model_str)
        assert result == SupportedModels.LLAMA_3_2_3B

    def test_resolve_model_with_none_raises_exception(self):
        """Test that None model raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            _resolve_model(None)
        assert exc_info.value.status_code == 400
        assert "Model is required" in exc_info.value.detail

    def test_resolve_model_with_unsupported_model_raises_exception(self):
        """Test that unsupported model string raises HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            _resolve_model("unsupported/model")
        assert exc_info.value.status_code == 400
        assert "Unsupported model" in exc_info.value.detail

    def test_resolve_model_with_qwen_model(self):
        """Test resolving Qwen model."""
        model_str = SupportedModels.QWEN_3_4B.value
        result = _resolve_model(model_str)
        assert result == SupportedModels.QWEN_3_4B


class TestTokenizeEndpoint:
    """Tests for tokenize endpoint."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.model_max_length = 8192
        tokenizer.convert_ids_to_tokens = MagicMock(
            return_value=["<s>", "Hello", "world", "</s>", "<pad>"]
        )
        return tokenizer

    @pytest.fixture
    def mock_auto_tokenizer(self, mock_tokenizer):
        """Mock AutoTokenizer.from_pretrained."""
        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            yield mock_auto

    def test_tokenize_basic_request(self, mock_auto_tokenizer, mock_tokenizer):
        """Test basic tokenization request."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=True,
            return_token_strs=False,
        )

        response = tokenize(request)

        assert response.count == 5
        assert response.max_model_len == 8192
        assert response.tokens == [1, 2, 3, 4, 5]
        assert response.token_strs is None
        mock_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=True
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.LLAMA_3_2_3B.value
        )

    def test_tokenize_with_token_strs(self, mock_auto_tokenizer, mock_tokenizer):
        """Test tokenization with return_token_strs=True."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=True,
            return_token_strs=True,
        )

        response = tokenize(request)

        assert response.count == 5
        assert response.tokens == [1, 2, 3, 4, 5]
        assert response.token_strs == ["<s>", "Hello", "world", "</s>", "<pad>"]
        mock_tokenizer.convert_ids_to_tokens.assert_called_once_with([1, 2, 3, 4, 5])

    def test_tokenize_without_special_tokens(self, mock_auto_tokenizer, mock_tokenizer):
        """Test tokenization without special tokens."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=False,
            return_token_strs=False,
        )

        response = tokenize(request)

        assert response.count == 5
        mock_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=False
        )

    def test_tokenize_with_default_model(self, mock_auto_tokenizer, mock_tokenizer):
        """Test that default model is used when model is not specified."""
        request = TokenizeCompletionRequest(
            prompt="Hello world",
        )

        response = tokenize(request)

        assert response.count == 5
        mock_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=True
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )

    def test_tokenize_with_unsupported_model_raises_exception(self):
        """Test that unsupported model raises HTTPException."""
        request = TokenizeCompletionRequest(
            model="unsupported/model",
            prompt="Hello world",
        )

        with pytest.raises(HTTPException) as exc_info:
            tokenize(request)
        assert exc_info.value.status_code == 400

    def test_tokenize_with_tokenizer_error_raises_exception(
        self, mock_auto_tokenizer, mock_tokenizer
    ):
        """Test that tokenizer errors are handled properly."""
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
        )

        with pytest.raises(HTTPException) as exc_info:
            tokenize(request)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Tokenization failed"

    def test_tokenize_with_tokenizer_loading_error_raises_exception(self):
        """Test that tokenizer loading errors are handled properly."""
        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("Loading failed")
            request = TokenizeCompletionRequest(
                model=SupportedModels.LLAMA_3_2_3B.value,
                prompt="Hello world",
            )

            with pytest.raises(HTTPException) as exc_info:
                tokenize(request)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Tokenization failed"

    def test_tokenize_with_qwen_model(self, mock_auto_tokenizer, mock_tokenizer):
        """Test tokenization with Qwen model."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.QWEN_3_4B.value,
            prompt="Test prompt",
        )

        response = tokenize(request)

        assert response.count == 5
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )


class TestDetokenizeEndpoint:
    """Tests for detokenize endpoint."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="Hello world")
        return tokenizer

    @pytest.fixture
    def mock_auto_tokenizer(self, mock_tokenizer):
        """Mock AutoTokenizer.from_pretrained."""
        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            yield mock_auto

    def test_detokenize_basic_request(self, mock_auto_tokenizer, mock_tokenizer):
        """Test basic detokenization request."""
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[1, 2, 3, 4, 5],
        )

        response = detokenize(request)

        assert response.prompt == "Hello world"
        mock_tokenizer.decode.assert_called_once_with(
            [1, 2, 3, 4, 5], skip_special_tokens=False
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.LLAMA_3_2_3B.value
        )

    def test_detokenize_with_empty_tokens(self, mock_auto_tokenizer, mock_tokenizer):
        """Test detokenization with empty token list."""
        mock_tokenizer.decode.return_value = ""
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[],
        )

        response = detokenize(request)

        assert response.prompt == ""
        mock_tokenizer.decode.assert_called_once_with([], skip_special_tokens=False)

    def test_detokenize_with_default_model(self, mock_auto_tokenizer, mock_tokenizer):
        """Test that default model is used when model is not specified."""
        request = DetokenizeRequest(
            tokens=[1, 2, 3],
        )

        response = detokenize(request)

        assert response.prompt == "Hello world"
        mock_tokenizer.decode.assert_called_once_with(
            [1, 2, 3], skip_special_tokens=False
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )

    def test_detokenize_with_unsupported_model_raises_exception(self):
        """Test that unsupported model raises HTTPException."""
        request = DetokenizeRequest(
            model="unsupported/model",
            tokens=[1, 2, 3],
        )

        with pytest.raises(HTTPException) as exc_info:
            detokenize(request)
        assert exc_info.value.status_code == 400

    def test_detokenize_with_tokenizer_error_raises_exception(
        self, mock_auto_tokenizer, mock_tokenizer
    ):
        """Test that tokenizer errors are handled properly."""
        mock_tokenizer.decode.side_effect = Exception("Tokenizer error")
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[1, 2, 3],
        )

        with pytest.raises(HTTPException) as exc_info:
            detokenize(request)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Detokenization failed"

    def test_detokenize_with_tokenizer_loading_error_raises_exception(self):
        """Test that tokenizer loading errors are handled properly."""
        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = Exception("Loading failed")
            request = DetokenizeRequest(
                model=SupportedModels.LLAMA_3_2_3B.value,
                tokens=[1, 2, 3],
            )

            with pytest.raises(HTTPException) as exc_info:
                detokenize(request)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Detokenization failed"

    def test_detokenize_with_qwen_model(self, mock_auto_tokenizer, mock_tokenizer):
        """Test detokenization with Qwen model."""
        request = DetokenizeRequest(
            model=SupportedModels.QWEN_3_4B.value,
            tokens=[10, 20, 30],
        )

        response = detokenize(request)

        assert response.prompt == "Hello world"
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )

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

# Mock settings before importing domain classes that use it
# The domain classes access settings.vllm.model at class definition time
from config.constants import SupportedModels

mock_vllm_settings = MagicMock()
mock_vllm_settings.model = SupportedModels.QWEN_3_4B.value
mock_settings_instance = MagicMock()
mock_settings_instance.vllm = mock_vllm_settings

sys.modules["config.settings"].settings = mock_settings_instance

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
    def mock_llama_tokenizer(self):
        """Create a mock Llama tokenizer with real token values."""
        # Real values from Llama 3.2 3B for "Hello world"
        # with special tokens: [128000, 9906, 1917]
        # without special tokens: [9906, 1917]
        # token strs: ['<|begin_of_text|>', 'Hello', 'Ġworld']
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[128000, 9906, 1917])
        tokenizer.model_max_length = 131072
        tokenizer.convert_ids_to_tokens = MagicMock(
            return_value=["<|begin_of_text|>", "Hello", "Ġworld"]
        )
        return tokenizer

    @pytest.fixture
    def mock_qwen_tokenizer(self):
        """Create a mock Qwen tokenizer with real token values."""
        # Real values from Qwen 3 4B for "Hello world"
        # with special tokens: [9707, 1879]
        # without special tokens: [9707, 1879]
        # token strs: ['Hello', 'Ġworld']
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[9707, 1879])
        tokenizer.model_max_length = 131072
        tokenizer.convert_ids_to_tokens = MagicMock(return_value=["Hello", "Ġworld"])
        return tokenizer

    @pytest.fixture
    def mock_auto_tokenizer(self, mock_llama_tokenizer, mock_qwen_tokenizer):
        """Mock AutoTokenizer.from_pretrained to return appropriate tokenizer."""

        def from_pretrained_side_effect(model_name):
            if model_name == SupportedModels.LLAMA_3_2_3B.value:
                return mock_llama_tokenizer
            elif model_name == SupportedModels.QWEN_3_4B.value:
                return mock_qwen_tokenizer
            return mock_llama_tokenizer  # default

        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = from_pretrained_side_effect
            yield mock_auto

    def test_tokenize_basic_request(self, mock_auto_tokenizer, mock_llama_tokenizer):
        """Test basic tokenization request."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=True,
            return_token_strs=False,
        )

        response = tokenize(request)

        assert response.count == 3
        assert response.max_model_len == 131072
        assert response.tokens == [128000, 9906, 1917]
        assert response.token_strs is None
        mock_llama_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=True
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.LLAMA_3_2_3B.value
        )

    def test_tokenize_with_token_strs(self, mock_auto_tokenizer, mock_llama_tokenizer):
        """Test tokenization with return_token_strs=True."""
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=True,
            return_token_strs=True,
        )

        response = tokenize(request)

        assert response.count == 3
        assert response.tokens == [128000, 9906, 1917]
        assert response.token_strs == ["<|begin_of_text|>", "Hello", "Ġworld"]
        mock_llama_tokenizer.convert_ids_to_tokens.assert_called_once_with(
            [128000, 9906, 1917]
        )

    def test_tokenize_without_special_tokens(
        self, mock_auto_tokenizer, mock_llama_tokenizer
    ):
        """Test tokenization without special tokens."""
        # Update mock to return tokens without special tokens
        mock_llama_tokenizer.encode.return_value = [9906, 1917]
        request = TokenizeCompletionRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            prompt="Hello world",
            add_special_tokens=False,
            return_token_strs=False,
        )

        response = tokenize(request)

        assert response.count == 2
        assert response.tokens == [9906, 1917]
        mock_llama_tokenizer.encode.assert_called_once_with(
            "Hello world", add_special_tokens=False
        )

    def test_tokenize_with_default_model(
        self, mock_auto_tokenizer, mock_qwen_tokenizer
    ):
        """Test that default model is used when model is not specified."""
        request = TokenizeCompletionRequest(
            prompt="Hello world",
        )

        response = tokenize(request)

        assert response.count == 2
        assert response.tokens == [9707, 1879]
        mock_qwen_tokenizer.encode.assert_called_once_with(
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
        self, mock_auto_tokenizer, mock_llama_tokenizer
    ):
        """Test that tokenizer errors are handled properly."""
        mock_llama_tokenizer.encode.side_effect = Exception("Tokenizer error")
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

    def test_tokenize_with_qwen_model(self, mock_auto_tokenizer, mock_qwen_tokenizer):
        """Test tokenization with Qwen model."""
        # For "Test prompt", we'll use a different mock return value
        # But for consistency, let's test with "Hello world" which we have real values for
        request = TokenizeCompletionRequest(
            model=SupportedModels.QWEN_3_4B.value,
            prompt="Hello world",
        )

        response = tokenize(request)

        assert response.count == 2
        assert response.tokens == [9707, 1879]
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )


class TestDetokenizeEndpoint:
    """Tests for detokenize endpoint."""

    @pytest.fixture
    def mock_llama_tokenizer(self):
        """Create a mock Llama tokenizer with real decode values."""
        # Real decode from Llama 3.2 3B for [128000, 9906, 1917]: '<|begin_of_text|>Hello world'
        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="<|begin_of_text|>Hello world")
        return tokenizer

    @pytest.fixture
    def mock_qwen_tokenizer(self):
        """Create a mock Qwen tokenizer with real decode values."""
        # Real decode from Qwen 3 4B for [9707, 1879]: 'Hello world'
        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="Hello world")
        return tokenizer

    @pytest.fixture
    def mock_auto_tokenizer(self, mock_llama_tokenizer, mock_qwen_tokenizer):
        """Mock AutoTokenizer.from_pretrained to return appropriate tokenizer."""

        def from_pretrained_side_effect(model_name):
            if model_name == SupportedModels.LLAMA_3_2_3B.value:
                return mock_llama_tokenizer
            elif model_name == SupportedModels.QWEN_3_4B.value:
                return mock_qwen_tokenizer
            return mock_llama_tokenizer  # default

        with patch("open_ai_api.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = from_pretrained_side_effect
            yield mock_auto

    def test_detokenize_basic_request(self, mock_auto_tokenizer, mock_llama_tokenizer):
        """Test basic detokenization request."""
        # Use real token values from Llama
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[128000, 9906, 1917],
        )

        response = detokenize(request)

        assert response.prompt == "<|begin_of_text|>Hello world"
        mock_llama_tokenizer.decode.assert_called_once_with(
            [128000, 9906, 1917], skip_special_tokens=False
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.LLAMA_3_2_3B.value
        )

    def test_detokenize_with_empty_tokens(
        self, mock_auto_tokenizer, mock_llama_tokenizer
    ):
        """Test detokenization with empty token list."""
        mock_llama_tokenizer.decode.return_value = ""
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[],
        )

        response = detokenize(request)

        assert response.prompt == ""
        mock_llama_tokenizer.decode.assert_called_once_with(
            [], skip_special_tokens=False
        )

    def test_detokenize_with_default_model(
        self, mock_auto_tokenizer, mock_qwen_tokenizer
    ):
        """Test that default model is used when model is not specified."""
        # Use real token values from Qwen
        request = DetokenizeRequest(
            tokens=[9707, 1879],
        )

        response = detokenize(request)

        assert response.prompt == "Hello world"
        mock_qwen_tokenizer.decode.assert_called_once_with(
            [9707, 1879], skip_special_tokens=False
        )
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )

    def test_detokenize_with_unsupported_model_raises_exception(self):
        """Test that unsupported model raises HTTPException."""
        request = DetokenizeRequest(
            model="unsupported/model",
            tokens=[9707, 1879],
        )

        with pytest.raises(HTTPException) as exc_info:
            detokenize(request)
        assert exc_info.value.status_code == 400

    def test_detokenize_with_tokenizer_error_raises_exception(
        self, mock_auto_tokenizer, mock_llama_tokenizer
    ):
        """Test that tokenizer errors are handled properly."""
        mock_llama_tokenizer.decode.side_effect = Exception("Tokenizer error")
        request = DetokenizeRequest(
            model=SupportedModels.LLAMA_3_2_3B.value,
            tokens=[128000, 9906, 1917],
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
                tokens=[128000, 9906, 1917],
            )

            with pytest.raises(HTTPException) as exc_info:
                detokenize(request)
            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Detokenization failed"

    def test_detokenize_with_qwen_model(self, mock_auto_tokenizer, mock_qwen_tokenizer):
        """Test detokenization with Qwen model."""
        # Use real token values from Qwen
        request = DetokenizeRequest(
            model=SupportedModels.QWEN_3_4B.value,
            tokens=[9707, 1879],
        )

        response = detokenize(request)

        assert response.prompt == "Hello world"
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            SupportedModels.QWEN_3_4B.value
        )

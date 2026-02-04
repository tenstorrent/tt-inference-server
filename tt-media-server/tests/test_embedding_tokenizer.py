# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for utils.embedding_tokenizer.EmbeddingTokenizer."""

from unittest.mock import MagicMock, patch

from utils.embedding_tokenizer import EmbeddingTokenizer


class TestEmbeddingTokenizerProperty:
    """Tests for EmbeddingTokenizer.tokenizer property (lazy load)."""

    def test_tokenizer_property_loads_on_first_access(self):
        """First access to .tokenizer calls AutoTokenizer.from_pretrained and caches."""
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = MagicMock()
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")

            result = tok.tokenizer

            mock_auto.from_pretrained.assert_called_once_with("BAAI/bge-large-en-v1.5")
            assert tok._tokenizer is result
            assert result == mock_auto.from_pretrained.return_value

    def test_tokenizer_property_returns_cached_on_second_access(self):
        """Second access to .tokenizer returns cached instance without calling from_pretrained again."""
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = MagicMock()
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
            first = tok.tokenizer
            second = tok.tokenizer

            assert first is second
            mock_auto.from_pretrained.assert_called_once()


class TestEmbeddingTokenizerTokenize:
    """Tests for EmbeddingTokenizer.tokenize."""

    def test_tokenize_calls_tokenizer_with_args_and_returns_result(self):
        """tokenize passes text_inputs and max_length to tokenizer and returns result."""
        mock_tokenizer = MagicMock()
        mock_result = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.return_value = mock_result
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")
            text_inputs = ["Hello world", "Second sentence"]
            max_length = 384

            result = tok.tokenize(text_inputs, max_length)

            mock_tokenizer.assert_called_once_with(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            assert result == mock_result


class TestEmbeddingTokenizerCalculateTokenCounts:
    """Tests for EmbeddingTokenizer.calculate_token_counts."""

    def test_calculate_token_counts_uses_attention_mask_when_present(self):
        """When attention_mask is in tokenized, counts are sum per row truncated to num_requests."""
        mock_attention = MagicMock()
        mock_attention.sum.return_value.tolist.return_value = [3, 2]
        tokenized = {
            "input_ids": MagicMock(),
            "attention_mask": mock_attention,
        }
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = MagicMock()
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")

            result = tok.calculate_token_counts(tokenized, num_requests=2)

            mock_attention.sum.assert_called_once_with(dim=1)
            assert result == [3, 2]

    def test_calculate_token_counts_truncates_to_num_requests(self):
        """Counts list is truncated to num_requests."""
        mock_attention = MagicMock()
        mock_attention.sum.return_value.tolist.return_value = [3, 3, 3]
        tokenized = {
            "input_ids": MagicMock(),
            "attention_mask": mock_attention,
        }
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = MagicMock()
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")

            result = tok.calculate_token_counts(tokenized, num_requests=2)

            assert result == [3, 3]

    def test_calculate_token_counts_without_attention_mask_uses_pad_token_id(self):
        """When attention_mask is missing, counts non-pad tokens using pad_token_id."""
        mock_row0 = MagicMock()
        mock_row0.__ne__.return_value.sum.return_value.item.return_value = 2
        mock_row1 = MagicMock()
        mock_row1.__ne__.return_value.sum.return_value.item.return_value = 1
        mock_input_ids = MagicMock()
        mock_input_ids.__getitem__.side_effect = [mock_row0, mock_row1]
        tokenized = {"input_ids": mock_input_ids}
        mock_hf = MagicMock()
        mock_hf.pad_token_id = 0
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_hf
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")

            result = tok.calculate_token_counts(tokenized, num_requests=2)

            assert result == [2, 1]
        mock_row0.__ne__.assert_called_with(0)
        mock_row1.__ne__.assert_called_with(0)

    def test_calculate_token_counts_without_attention_mask_pad_token_id_none_uses_zero(
        self,
    ):
        """When attention_mask is missing and pad_token_id is None, use 0 as pad id."""
        mock_row0 = MagicMock()
        mock_row0.__ne__.return_value.sum.return_value.item.return_value = 2
        mock_row1 = MagicMock()
        mock_row1.__ne__.return_value.sum.return_value.item.return_value = 1
        mock_input_ids = MagicMock()
        mock_input_ids.__getitem__.side_effect = [mock_row0, mock_row1]
        tokenized = {"input_ids": mock_input_ids}
        mock_hf = MagicMock()
        mock_hf.pad_token_id = None
        with patch("utils.embedding_tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = mock_hf
            tok = EmbeddingTokenizer("BAAI/bge-large-en-v1.5")

            result = tok.calculate_token_counts(tokenized, num_requests=2)

            assert result == [2, 1]
        mock_row0.__ne__.assert_called_with(0)
        mock_row1.__ne__.assert_called_with(0)

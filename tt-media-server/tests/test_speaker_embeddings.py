# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import torch
import numpy as np
import base64
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from utils.speaker_embeddings import (
    SpeakerEmbeddingsManager,
    EmbeddingDimensionError,
    EmbeddingLoadError,
)


class TestSpeakerEmbeddingsManager:
    """Test SpeakerEmbeddingsManager functionality"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create SpeakerEmbeddingsManager with temp cache dir"""
        return SpeakerEmbeddingsManager(cache_dir=str(temp_cache_dir))


    def test_initialization(self, manager, temp_cache_dir):
        """Test manager initialization"""
        assert manager.cache_dir == temp_cache_dir
        assert manager.custom_embeddings_dir is None
        assert len(manager._embeddings_cache) == 0

    def test_initialization_with_custom_dir(self, temp_cache_dir):
        """Test initialization with custom embeddings directory"""
        custom_dir = temp_cache_dir / "custom"
        custom_dir.mkdir()
        manager = SpeakerEmbeddingsManager(
            cache_dir=str(temp_cache_dir),
            custom_embeddings_dir=str(custom_dir)
        )
        assert manager.custom_embeddings_dir == custom_dir

    def test_validate_embedding_valid(self, manager):
        """Test validation of valid embedding"""
        # Create a mock tensor that behaves like a 1D tensor with 512 elements
        embedding = MagicMock()
        embedding.dim.return_value = 1
        embedding.shape = [512]
        # Should not raise exception
        result = manager._validate_embedding(embedding)
        assert result is True

    def test_validate_embedding_wrong_dimension(self, manager):
        """Test validation of embedding with wrong dimensions"""
        # 2D embedding (should be 1D)
        embedding = torch.randn(10, 512)
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            manager._validate_embedding(embedding)
        assert "must be 1D tensor" in str(exc_info.value)

    def test_validate_embedding_wrong_size(self, manager):
        """Test validation of embedding with wrong size"""
        # Wrong size (not 512)
        embedding = MagicMock()
        embedding.dim.return_value = 1
        embedding.shape = [256]
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            manager._validate_embedding(embedding)
        assert "dimension must be 512" in str(exc_info.value)

    @patch('utils.speaker_embeddings.load_dataset')
    def test_load_default_embeddings_error_handling(self, mock_load_dataset, manager):
        """Test loading default embeddings with error"""
        mock_load_dataset.side_effect = Exception("Network error")

        # Should handle error gracefully
        with pytest.raises(EmbeddingLoadError):
            manager.load_default_embeddings()


    @patch('utils.speaker_embeddings.load_dataset')
    def test_load_default_embeddings_load_error(self, mock_load_dataset, manager):
        """Test loading default embeddings with network error"""
        mock_load_dataset.side_effect = Exception("Network error")

        with pytest.raises(EmbeddingLoadError) as exc_info:
            manager.load_default_embeddings()
        assert "Failed to load default embeddings" in str(exc_info.value)

    @patch('utils.speaker_embeddings.torch')
    def test_load_custom_embeddings(self, mock_torch, manager, temp_cache_dir):
        """Test loading custom embeddings from directory"""
        custom_dir = temp_cache_dir / "custom"
        custom_dir.mkdir()

        # Mock torch.load and torch.save
        mock_tensor = MagicMock()
        mock_tensor.shape = [512]
        mock_tensor.dim.return_value = 1
        mock_torch.load.return_value = mock_tensor

        # Create dummy file (torch.save is mocked)
        test_file = custom_dir / "custom_speaker.pt"
        test_file.touch()

        manager.custom_embeddings_dir = custom_dir
        manager.load_custom_embeddings()

        assert "custom_speaker" in manager._embeddings_cache

    @patch('utils.speaker_embeddings.torch')
    def test_load_custom_embeddings_pth_format(self, mock_torch, manager, temp_cache_dir):
        """Test loading custom embeddings with .pth extension"""
        custom_dir = temp_cache_dir / "custom"
        custom_dir.mkdir()

        # Mock torch.load
        mock_tensor = MagicMock()
        mock_tensor.shape = [512]
        mock_tensor.dim.return_value = 1
        mock_torch.load.return_value = mock_tensor

        # Create dummy file
        test_file = custom_dir / "custom_speaker.pth"
        test_file.touch()

        manager.custom_embeddings_dir = custom_dir
        manager.load_custom_embeddings()

        assert "custom_speaker" in manager._embeddings_cache

    def test_load_custom_embeddings_invalid_file(self, manager, temp_cache_dir):
        """Test loading custom embeddings with invalid file"""
        custom_dir = temp_cache_dir / "custom"
        custom_dir.mkdir()

        # Create invalid .pt file (wrong dimension)
        test_embedding = torch.randn(256, dtype=torch.float32)  # Wrong dimension
        torch.save(test_embedding, custom_dir / "invalid_speaker.pt")

        manager.custom_embeddings_dir = custom_dir
        manager.load_custom_embeddings()

        # Should skip invalid embedding
        assert len(manager._embeddings_cache) == 0

    def test_get_speaker_embedding_existing(self, manager):
        """Test getting existing speaker embedding"""
        test_embedding = torch.randn(1, 512, dtype=torch.float32)
        manager._embeddings_cache["test_speaker"] = test_embedding

        result = manager.get_speaker_embedding("test_speaker")
        assert torch.equal(result, test_embedding)

    def test_get_speaker_embedding_load_default(self, manager):
        """Test getting speaker embedding that triggers default loading"""
        with patch.object(manager, 'load_default_embeddings') as mock_load:
            with patch.object(manager, 'load_custom_embeddings') as mock_custom:
                # Set up mock to add embedding when called
                def add_embedding():
                    manager._embeddings_cache["new_speaker"] = torch.randn(1, 512)
                mock_load.side_effect = add_embedding

                result = manager.get_speaker_embedding("new_speaker")

                mock_load.assert_called_once()
                assert "new_speaker" in manager._embeddings_cache

    def test_get_speaker_embedding_not_found(self, manager):
        """Test getting non-existent speaker embedding"""
        with patch.object(manager, 'load_default_embeddings'):
            with patch.object(manager, 'load_custom_embeddings'):
                # Mock loading methods to not add the speaker
                with pytest.raises(EmbeddingLoadError) as exc_info:
                    manager.get_speaker_embedding("nonexistent_speaker")

                assert "Speaker 'nonexistent_speaker' not found" in str(exc_info.value)

    def test_process_user_embedding_invalid_base64(self, manager):
        """Test processing invalid base64 string"""
        with pytest.raises(EmbeddingLoadError):
            manager.process_user_embedding("invalid_base64!")

    def test_list_available_speakers(self, manager):
        """Test listing available speakers"""
        manager._embeddings_cache = {
            "speaker_1": torch.randn(1, 512),
            "speaker_2": torch.randn(1, 512),
            "speaker_3": torch.randn(1, 512),
        }

        speakers = manager.list_available_speakers()
        assert set(speakers) == {"speaker_1", "speaker_2", "speaker_3"}

    def test_preload_embeddings(self, manager):
        """Test preloading embeddings"""
        with patch.object(manager, 'load_default_embeddings') as mock_default:
            with patch.object(manager, 'load_custom_embeddings') as mock_custom:
                manager.preload_embeddings()

                mock_default.assert_called_once()
                if manager.custom_embeddings_dir:
                    mock_custom.assert_called_once()

    def test_preload_embeddings_with_error(self, manager):
        """Test preloading embeddings with error (should not crash)"""
        with patch.object(manager, 'load_default_embeddings', side_effect=Exception("Load error")):
            # Should not raise exception
            manager.preload_embeddings()


    def test_cache_corruption(self, manager, temp_cache_dir):
        """Test handling of corrupted cache file"""
        cache_file = temp_cache_dir / "speaker_embeddings_cache.pkl"

        # Create corrupted cache file
        with open(cache_file, 'w') as f:
            f.write("corrupted data")

        # Should handle gracefully
        new_manager = SpeakerEmbeddingsManager(cache_dir=str(temp_cache_dir))
        # Cache should be empty (not crash)
        assert len(new_manager._embeddings_cache) == 0

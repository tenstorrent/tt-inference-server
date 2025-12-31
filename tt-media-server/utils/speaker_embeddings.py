# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import base64
import torch
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import pickle
from datasets import load_dataset
from utils.logger import TTLogger

class SpeakerEmbeddingsError(Exception):
    """Base exception for speaker embedding errors"""
    pass

class EmbeddingDimensionError(SpeakerEmbeddingsError):
    """Raised when embedding has wrong dimensions"""
    pass

class EmbeddingLoadError(SpeakerEmbeddingsError):
    """Raised when embedding cannot be loaded"""
    pass

class SpeakerEmbeddingsManager:
    """
    Manages speaker embeddings for SpeechT5 TTS.

    Handles loading default embeddings from HuggingFace datasets,
    caching them, and processing user-provided embeddings.
    """

    SPEECHT5_EMBEDDING_DIM = 512  # SpeechT5 expects 512-dimensional embeddings
    DEFAULT_DATASET = "Matthijs/cmu-arctic-xvectors"
    CACHE_FILE = "speaker_embeddings_cache.pkl"

    def __init__(self, cache_dir: Optional[str] = None, custom_embeddings_dir: Optional[str] = None):
        """
        Initialize the speaker embeddings manager.

        Args:
            cache_dir: Directory to cache downloaded embeddings (default: ~/.cache/speecht5)
            custom_embeddings_dir: Directory containing custom speaker embeddings
        """
        self.logger = TTLogger()
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "speecht5"
        self.custom_embeddings_dir = Path(custom_embeddings_dir) if custom_embeddings_dir else None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded embeddings: speaker_id -> torch.Tensor
        self._embeddings_cache: Dict[str, torch.Tensor] = {}

        # Load cached embeddings if available
        self._load_cache()

    def _load_cache(self):
        """Load cached embeddings from disk"""
        cache_file = self.cache_dir / self.CACHE_FILE
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._embeddings_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._embeddings_cache)} cached speaker embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings cache: {e}")

    def _save_cache(self):
        """Save cached embeddings to disk"""
        cache_file = self.cache_dir / self.CACHE_FILE
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embeddings_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save embeddings cache: {e}")

    def _validate_embedding(self, embedding: torch.Tensor) -> bool:
        """
        Validate embedding dimensions and type.

        Args:
            embedding: Speaker embedding tensor

        Returns:
            True if valid

        Raises:
            EmbeddingDimensionError: If dimensions are incorrect
        """
        if embedding.dim() != 1:
            raise EmbeddingDimensionError(f"Embedding must be 1D tensor, got {embedding.dim()}D")

        if embedding.shape[0] != self.SPEECHT5_EMBEDDING_DIM:
            raise EmbeddingDimensionError(
                f"Embedding dimension must be {self.SPEECHT5_EMBEDDING_DIM}, got {embedding.shape[0]}"
            )

        return True

    def load_default_embeddings(self):
        """
        Load default speaker embeddings from HuggingFace dataset.

        Downloads and caches embeddings from Matthijs/cmu-arctic-xvectors dataset.
        """
        try:
            self.logger.info(f"Loading default speaker embeddings from {self.DEFAULT_DATASET}...")
            dataset = load_dataset(self.DEFAULT_DATASET, split="validation")

            for i, example in enumerate(dataset):
                # Use filename as speaker ID, or create a numeric ID
                speaker_id = example.get("speaker_id", f"speaker_{i:04d}")
                embedding = torch.tensor(example["xvector"], dtype=torch.float32).unsqueeze(0)  # Add batch dimension

                try:
                    # Validate the original embedding shape (without batch dim for validation)
                    self._validate_embedding(embedding.squeeze(0))
                    self._embeddings_cache[speaker_id] = embedding
                except EmbeddingDimensionError as e:
                    self.logger.warning(f"Skipping invalid embedding for speaker {speaker_id}: {e}")

            self._save_cache()
            self.logger.info(f"Loaded {len(self._embeddings_cache)} default speaker embeddings")

        except Exception as e:
            raise EmbeddingLoadError(f"Failed to load default embeddings: {e}")

    def load_custom_embeddings(self):
        """
        Load custom speaker embeddings from local directory.

        Expects .pt or .pth files containing torch.Tensor objects.
        """
        if not self.custom_embeddings_dir or not self.custom_embeddings_dir.exists():
            return

        self.logger.info(f"Loading custom embeddings from {self.custom_embeddings_dir}")

        for file_path in self.custom_embeddings_dir.glob("*.pt"):
            try:
                speaker_id = file_path.stem  # filename without extension
                embedding = torch.load(file_path, map_location='cpu')

                self._validate_embedding(embedding)
                self._embeddings_cache[speaker_id] = embedding
                self.logger.info(f"Loaded custom embedding for speaker: {speaker_id}")

            except Exception as e:
                self.logger.warning(f"Failed to load custom embedding {file_path}: {e}")

        for file_path in self.custom_embeddings_dir.glob("*.pth"):
            try:
                speaker_id = file_path.stem  # filename without extension
                embedding = torch.load(file_path, map_location='cpu')

                self._validate_embedding(embedding)
                self._embeddings_cache[speaker_id] = embedding
                self.logger.info(f"Loaded custom embedding for speaker: {speaker_id}")

            except Exception as e:
                self.logger.warning(f"Failed to load custom embedding {file_path}: {e}")

    def get_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        """
        Get speaker embedding by ID.

        Args:
            speaker_id: Speaker identifier

        Returns:
            Speaker embedding tensor

        Raises:
            EmbeddingLoadError: If speaker not found
        """
        if speaker_id not in self._embeddings_cache:
            # Try to load custom embeddings if not already loaded
            if self.custom_embeddings_dir:
                self.load_custom_embeddings()

            # Try to load default embeddings if not already loaded
            if speaker_id not in self._embeddings_cache:
                self.load_default_embeddings()

        if speaker_id not in self._embeddings_cache:
            available_speakers = list(self._embeddings_cache.keys())
            raise EmbeddingLoadError(
                f"Speaker '{speaker_id}' not found. Available speakers: {available_speakers[:10]}..."
                f"({len(available_speakers)} total)"
            )

        return self._embeddings_cache[speaker_id]

    def process_user_embedding(self, embedding_data: Union[str, bytes, np.ndarray]) -> torch.Tensor:
        """
        Process user-provided speaker embedding.

        Args:
            embedding_data: Base64 string, bytes, or numpy array

        Returns:
            Validated torch.Tensor embedding

        Raises:
            EmbeddingLoadError: If processing fails
        """
        try:
            # Handle different input types
            if isinstance(embedding_data, str):
                # Base64 encoded string
                embedding_bytes = base64.b64decode(embedding_data)
                embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            elif isinstance(embedding_data, bytes):
                # Raw bytes
                embedding_array = np.frombuffer(embedding_data, dtype=np.float32)
            elif isinstance(embedding_data, np.ndarray):
                # Already numpy array
                embedding_array = embedding_data
            else:
                raise EmbeddingLoadError(f"Unsupported embedding data type: {type(embedding_data)}")

            # Convert to torch tensor
            embedding = torch.tensor(embedding_array, dtype=torch.float32)

            # Validate
            self._validate_embedding(embedding)

            # Add batch dimension like the demo
            return embedding.unsqueeze(0)

        except Exception as e:
            raise EmbeddingLoadError(f"Failed to process user embedding: {e}")

    def list_available_speakers(self) -> list[str]:
        """
        List all available speaker IDs.

        Returns:
            List of speaker identifiers
        """
        return list(self._embeddings_cache.keys())

    def preload_embeddings(self):
        """
        Preload all available embeddings into cache.

        Useful for warmup to avoid loading during inference.
        """
        try:
            self.load_default_embeddings()
            if self.custom_embeddings_dir:
                self.load_custom_embeddings()
        except Exception as e:
            self.logger.warning(f"Failed to preload embeddings: {e}")

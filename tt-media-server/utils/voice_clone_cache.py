# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import TTLogger


class VoiceCloneCacheError(Exception):
    """Base exception for voice-clone cache errors"""

    pass


class VoiceCloneCacheManager:
    """
    Manages registered voice-clone VQ-code prompts for the Inworld TTS runner.

    Handles caching audio-prompt VQ codes (produced once by
    ``CachingAudioEncoder.encode()``) keyed by a caller-assigned ``voice_id``,
    so subsequent text-to-speech requests can reuse a previously-registered
    voice's prompt without re-encoding the reference audio every time.
    """

    CACHE_FILE = "voice_clone_cache.pkl"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the voice-clone cache manager.

        Args:
            cache_dir: Directory to cache registered voices (default: HF_HOME/inworld_tts or /tmp/inworld_tts)
        """
        self.logger = TTLogger()
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use HF_HOME if available (works in docker), otherwise fallback to /tmp
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                self.cache_dir = Path(hf_home) / "inworld_tts"
            else:
                self.cache_dir = Path("/tmp") / "inworld_tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for registered voices: voice_id -> list[int] (VQ codes)
        self._voice_cache: Dict[str, List[int]] = {}

        # Load cached voices if available
        self._load_cache()

    def _load_cache(self):
        """Load cached voices from disk"""
        cache_file = self.cache_dir / self.CACHE_FILE
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._voice_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._voice_cache)} cached voices")
            except Exception as e:
                self.logger.warning(f"Failed to load voice-clone cache: {e}")

    def _save_cache(self):
        """Save cached voices to disk"""
        cache_file = self.cache_dir / self.CACHE_FILE
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._voice_cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save voice-clone cache: {e}")

    def register_voice(self, voice_id: str, speech_ids: List[int]) -> None:
        """
        Register a voice-clone VQ-code prompt under ``voice_id``, saving immediately.

        Args:
            voice_id: Identifier to register the voice under.
            speech_ids: VQ codes (from ``CachingAudioEncoder.encode()``) for the voice prompt.
        """
        self._voice_cache[voice_id] = speech_ids
        self._save_cache()

    def get_voice(self, voice_id: str) -> List[int]:
        """
        Get a registered voice's VQ-code prompt by ID.

        Args:
            voice_id: Voice identifier.

        Returns:
            The registered VQ codes.

        Raises:
            VoiceCloneCacheError: If voice_id is not registered.
        """
        if voice_id not in self._voice_cache:
            available_voices = list(self._voice_cache.keys())
            raise VoiceCloneCacheError(
                f"Voice '{voice_id}' not found. Available voices: {available_voices[:10]}"
                f"...({len(available_voices)} total)"
            )
        return self._voice_cache[voice_id]

    def list_voices(self) -> List[str]:
        """
        List all registered voice IDs.

        Returns:
            List of voice identifiers.
        """
        return list(self._voice_cache.keys())

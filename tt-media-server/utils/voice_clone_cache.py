# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import fcntl
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        # Cache for registered voices. New value shape is a metadata-carrying
        # dict: {"speech_ids": list[int], "language": Optional[str],
        # "description": Optional[str]}. Old on-disk pickles stored a bare
        # list[int] per voice_id; _load_cache() transparently upgrades those in
        # memory (see below) so old caches keep working without re-registration.
        self._voice_cache: Dict[str, Dict[str, Any]] = {}

        # Load cached voices if available
        self._load_cache()

    @staticmethod
    def _normalize_entry(value: Any) -> Dict[str, Any]:
        """Coerce a cached value into the metadata-carrying dict shape.

        Old-format entries are a bare ``list[int]`` of VQ codes with no
        metadata -- wrap them as
        ``{"speech_ids": value, "language": None, "description": None}``.
        New-format entries are already dicts and are returned (defensively
        filled in) as-is.
        """
        if isinstance(value, dict):
            return {
                "speech_ids": value.get("speech_ids", []),
                "language": value.get("language"),
                "description": value.get("description"),
            }
        # Old format: bare list[int] of VQ codes.
        return {"speech_ids": value, "language": None, "description": None}

    def _load_cache(self):
        """Load cached voices from disk, upgrading any old bare-list entries.

        This is non-destructive on disk: the in-memory upgrade is only written
        back the next time ``_save_cache()`` runs (i.e. on the next
        ``register_voice`` call), so an old-format pickle is never rewritten
        merely by reading it.
        """
        cache_file = self.cache_dir / self.CACHE_FILE
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    raw = pickle.load(f)
                self._voice_cache = {
                    voice_id: self._normalize_entry(value)
                    for voice_id, value in raw.items()
                }
                self.logger.info(f"Loaded {len(self._voice_cache)} cached voices")
            except Exception as e:
                self.logger.warning(f"Failed to load voice-clone cache: {e}")

    def _save_cache(self):
        """Persist cached voices to disk, merging with what's already there.

        DP (data-parallel) fleet coherence: each of the N independent
        single-chip workers holds its own in-memory cache and shares one
        on-disk pickle. A blind full-overwrite here would make concurrent
        workers clobber each other -- e.g. registering 14 voices across the
        fleet, each landing on a different worker, would leave only the
        last-writer's voice on disk (every worker writes just its own 12
        warmup-loaded voices plus the one it registered). Instead, take an
        exclusive cross-process file lock, re-read the current on-disk state,
        merge this worker's in-memory entries on top (our own freshly
        registered voice wins), write the union back, and adopt the merged
        result in memory. Registrations from every worker thus accumulate.
        For TP=1/8 single-worker setups this is an equivalent, cheap no-op
        merge (disk and memory already agree).
        """
        cache_file = self.cache_dir / self.CACHE_FILE
        lock_file = self.cache_dir / (self.CACHE_FILE + ".lock")
        try:
            with open(lock_file, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    merged: Dict[str, Dict[str, Any]] = {}
                    if cache_file.exists():
                        try:
                            with open(cache_file, "rb") as f:
                                raw = pickle.load(f)
                            merged = {
                                voice_id: self._normalize_entry(value)
                                for voice_id, value in raw.items()
                            }
                        except Exception as e:
                            self.logger.warning(
                                f"Could not read existing voice-clone cache for merge: {e}"
                            )
                            merged = {}
                    merged.update(self._voice_cache)
                    self._voice_cache = merged
                    with open(cache_file, "wb") as f:
                        pickle.dump(self._voice_cache, f)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception as e:
            self.logger.warning(f"Failed to save voice-clone cache: {e}")

    def register_voice(
        self,
        voice_id: str,
        speech_ids: List[int],
        *,
        language: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a voice-clone VQ-code prompt under ``voice_id``, saving immediately.

        Args:
            voice_id: Identifier to register the voice under.
            speech_ids: VQ codes (from ``CachingAudioEncoder.encode()``) for the voice prompt.
            language: Optional BCP-47 language tag for the voice.
            description: Optional human-readable description of the voice.
        """
        self._voice_cache[voice_id] = {
            "speech_ids": speech_ids,
            "language": language,
            "description": description,
        }
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
            # DP (data-parallel) fleet coherence: with N independent single-chip
            # workers each holding their own in-memory cache, a voice registered
            # via POST /v1/audio/voices lands on ONE worker, but the follow-up
            # POST /v1/audio/speech may route to a DIFFERENT worker whose
            # in-memory cache predates the registration. register_voice() always
            # persists to the shared on-disk pickle immediately, and VQ codes are
            # host-side, chip-independent integers, so re-reading disk on a miss
            # lets any worker pick up voices registered by any other worker. This
            # is a cheap no-op for TP=1/8 single-worker setups (a just-registered
            # voice is already in-memory, so this reload path is never hit there).
            self._load_cache()
        if voice_id not in self._voice_cache:
            available_voices = list(self._voice_cache.keys())
            raise VoiceCloneCacheError(
                f"Voice '{voice_id}' not found. Available voices: {available_voices[:10]}"
                f"...({len(available_voices)} total)"
            )
        return self._voice_cache[voice_id]["speech_ids"]

    def list_voices(self) -> List[str]:
        """
        List all registered voice IDs.

        Returns:
            List of voice identifiers.
        """
        return list(self._voice_cache.keys())

    def reload_from_disk(self) -> None:
        """Re-read the on-disk pickle into the in-memory cache.

        DP (data-parallel) fleet coherence: with N independent single-chip
        workers each holding their own in-memory cache, a voice registered via
        POST /v1/audio/voices lands on ONE worker and is persisted to the
        shared on-disk pickle immediately, but a later GET /v1/audio/voices
        may route to a DIFFERENT worker whose in-memory cache predates that
        registration. Re-reading disk here (same pattern get_voice() uses on a
        miss) lets any worker report the fleet-wide set of registered voices.
        For TP=1/8 single-worker setups this is a cheap, harmless refresh.
        """
        self._load_cache()

    def list_voices_with_metadata(self) -> List[Dict[str, Any]]:
        """
        List all registered voices with their metadata.

        Returns:
            A list of dicts, one per cached voice, each with keys
            ``voice_id``, ``language``, ``description`` and ``num_codes``.
            Old-format (pre-metadata) voices naturally report
            ``language=None`` and ``description=None``.
        """
        voices: List[Dict[str, Any]] = []
        for voice_id, entry in self._voice_cache.items():
            speech_ids = entry.get("speech_ids") or []
            voices.append(
                {
                    "voice_id": voice_id,
                    "language": entry.get("language"),
                    "description": entry.get("description"),
                    "num_codes": len(speech_ids),
                }
            )
        return voices

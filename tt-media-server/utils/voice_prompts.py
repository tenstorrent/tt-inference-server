# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Voice prompt manager for Qwen3-TTS.

Pre-loads Mimi-encoded reference audio + transcript pairs at startup, keyed by
``voice_id``. Mirrors the cache/list/preload shape of
``utils/speaker_embeddings.py::SpeakerEmbeddingsManager`` but the schema
differs — Qwen3 stores ``(ref_codes, ref_text, audio_data)`` per voice (no
fixed-dim embedding), so it does not subclass.

Encoding is delegated to
``models.demos.qwen3_tts.demo.demo_full_ttnn_tts.encode_reference_audio``,
which already caches its result to ``<audio_path>.refcache.pt``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from utils.logger import TTLogger


@dataclass
class VoicePrompt:
    voice_id: str
    ref_codes: torch.Tensor   # [seq_len, 16]
    ref_text: str
    audio_data: torch.Tensor  # [num_samples] @ 24 kHz
    speaker_embedding: object = None  # optional cached TTNN speaker embedding (ECAPA)


# Built-in voices shipped with the runner. Paths are resolved relative to
# tt-metal at startup; missing paths are skipped with a warning so a partial
# install is not fatal.
DEFAULT_VOICE_PROMPTS: List[Tuple[str, str, str]] = [
    (
        "jim",
        "models/demos/qwen3_tts/demo/jim_reference.wav",
        "Jason, can you put up the high level overview slides.",
    ),
    (
        "ashley",
        "/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts/Ashley_en.wav",
        "Keeping my goals visible every day to stay focused on what matters most.",
    ),
    (
        "satoshi",
        "/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts/Satoshi_ja.wav",
        "ほう、あの試合で彼のサブスティテューションが決まっていたなんてまさかの展開でしたね",
    ),
]


class VoicePromptManager:
    """Pre-loads built-in voice prompts and looks them up by ID at request time."""

    def __init__(self, tt_metal_home: Optional[str] = None):
        self.logger = TTLogger()
        self._voices: Dict[str, VoicePrompt] = {}
        self._tt_metal_home = Path(tt_metal_home or os.environ.get("TT_METAL_HOME", "."))

    def preload(self, voices: Optional[List[Tuple[str, str, str]]] = None) -> None:
        """Encode each (voice_id, audio_path, ref_text) tuple via Mimi (cached)."""
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import encode_reference_audio

        for voice_id, audio_path, ref_text in (voices or DEFAULT_VOICE_PROMPTS):
            p = Path(audio_path)
            if not p.is_absolute():
                p = (self._tt_metal_home / audio_path).resolve()
            if not p.is_file():
                self.logger.warning(
                    f"VoicePromptManager: missing audio for {voice_id!r} at {p} — skipping"
                )
                continue
            try:
                ref_codes, audio_data = encode_reference_audio(str(p), main_weights=None)
            except Exception as e:
                self.logger.error(
                    f"VoicePromptManager: failed to encode {voice_id!r} ({p}): {e}"
                )
                continue
            self._voices[voice_id] = VoicePrompt(
                voice_id=voice_id, ref_codes=ref_codes, ref_text=ref_text, audio_data=audio_data
            )
            self.logger.info(
                f"VoicePromptManager: pre-loaded {voice_id!r} from {p.name} "
                f"({ref_codes.shape[0]} frames, {len(audio_data) / 24000:.2f}s)"
            )

    def list_available(self) -> List[str]:
        return sorted(self._voices.keys())

    def get(self, voice_id: str) -> Optional[VoicePrompt]:
        return self._voices.get(voice_id)

    def precompute_speaker_embeddings(self, model) -> None:
        """Run ECAPA on each preloaded voice once (avoids per-request conv2d
        re-preparation that can hang on subsequent requests)."""
        for voice_id, prompt in self._voices.items():
            if prompt.speaker_embedding is not None:
                continue
            try:
                prompt.speaker_embedding = model.extract_speaker_embedding(prompt.audio_data)
                self.logger.info(
                    f"VoicePromptManager: cached speaker embedding for {voice_id!r}"
                )
            except Exception as e:
                self.logger.error(
                    f"VoicePromptManager: speaker-embedding precompute failed for "
                    f"{voice_id!r}: {e}"
                )


# Late import to avoid hard dependency on os at module top
import os  # noqa: E402

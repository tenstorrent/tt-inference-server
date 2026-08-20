# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Voice prompt manager for Qwen3-TTS (ICL ref audio + transcript by voice_id).

Default voice: jim clip in tt-metal ``models/demos/qwen3_tts/demo/``.
Extra: ``QWEN3_TTS_REF_AUDIO`` / ``QWEN3_TTS_REF_TEXT`` as id ``custom``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from utils.logger import TTLogger

DEFAULT_VOICE_ID = "jim"


@dataclass
class VoicePrompt:
    voice_id: str
    ref_codes: torch.Tensor  # [seq_len, 16]
    ref_text: str
    audio_data: torch.Tensor  # [num_samples] @ 24 kHz
    speaker_embedding: object = None


def _jim_prompt() -> Tuple[str, Path, str]:
    import models.demos.qwen3_tts.demo as demo_pkg

    # demo/ is a namespace package (no __init__.py), so __file__ is None.
    demo_dir = Path(next(iter(demo_pkg.__path__))).resolve()
    wav = demo_dir / "jim_reference.wav"
    txt = wav.with_suffix(".txt")
    ref_text = (
        txt.read_text(encoding="utf-8").strip()
        if txt.is_file()
        else "So basically you put up the high level overview slides."
    )
    return DEFAULT_VOICE_ID, wav, ref_text


class VoicePromptManager:
    """Pre-loads built-in voice prompts and looks them up by ID at request time."""

    def __init__(self, tt_metal_home: Optional[str] = None):
        self.logger = TTLogger()
        self._voices: Dict[str, VoicePrompt] = {}
        self._tt_metal_home = Path(
            tt_metal_home or os.environ.get("TT_METAL_HOME", ".")
        )

    def _iter_voice_specs(self) -> List[Tuple[str, Path, str]]:
        specs = []
        try:
            specs.append(_jim_prompt())
        except Exception as e:
            self.logger.warning(f"VoicePromptManager: jim default unavailable: {e}")
        env_audio = os.environ.get("QWEN3_TTS_REF_AUDIO")
        if env_audio:
            p = Path(env_audio)
            env_text = os.environ.get("QWEN3_TTS_REF_TEXT")
            sibling = p.with_suffix(".txt")
            if env_text:
                ref_text = env_text
            elif sibling.is_file():
                ref_text = sibling.read_text(encoding="utf-8").strip()
            else:
                ref_text = "Reference audio transcript."
            specs.append(("custom", p, ref_text))
        return specs

    def preload(self) -> None:
        from models.demos.qwen3_tts.tt.server import encode_reference_audio

        for voice_id, audio_path, ref_text in self._iter_voice_specs():
            p = Path(audio_path)
            if not p.is_absolute():
                p = (self._tt_metal_home / audio_path).resolve()
            if not p.is_file():
                self.logger.warning(
                    f"VoicePromptManager: missing audio for {voice_id!r} at {p} — skipping"
                )
                continue
            try:
                ref_codes, audio_data = encode_reference_audio(
                    str(p), main_weights=None
                )
            except Exception as e:
                self.logger.error(
                    f"VoicePromptManager: failed to encode {voice_id!r} ({p}): {e}"
                )
                continue
            self._voices[voice_id] = VoicePrompt(
                voice_id=voice_id,
                ref_codes=ref_codes,
                ref_text=ref_text,
                audio_data=audio_data,
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
        """Run ECAPA once per voice so later requests skip on-device conv2d prep."""
        for voice_id, prompt in self._voices.items():
            if prompt.speaker_embedding is not None:
                continue
            try:
                prompt.speaker_embedding = model.extract_speaker_embedding(
                    prompt.audio_data
                )
                self.logger.info(
                    f"VoicePromptManager: cached speaker embedding for {voice_id!r}"
                )
            except Exception as e:
                self.logger.error(
                    f"VoicePromptManager: speaker-embedding precompute failed for "
                    f"{voice_id!r}: {e}"
                )

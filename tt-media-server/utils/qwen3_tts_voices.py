# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Qwen3-TTS voice ids and request checks that need no torch or device.

The worker's VoicePromptManager loads ``jim`` (and ``custom`` when QWEN3_TTS_REF_AUDIO
is set); the API process checks requests against the same list so a bad voice is a
400 before it reaches a worker, not a 500 after it.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional

DEFAULT_VOICE_ID = "jim"
CUSTOM_VOICE_ID = "custom"
# speaker_id may also name a language: default voice, that language.
KNOWN_LANGUAGES = (
    "english",
    "chinese",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "japanese",
    "korean",
    "french",
    "russian",
)


def preset_voice_ids() -> list[str]:
    ids = [DEFAULT_VOICE_ID]
    if os.environ.get("QWEN3_TTS_REF_AUDIO"):
        ids.append(CUSTOM_VOICE_ID)
    return ids


def request_error(request) -> Optional[str]:
    """Why ``request`` cannot be served by the Qwen3-TTS runner, or None."""
    clone_audio = request.voice_clone_audio
    clone_text = request.voice_clone_text
    if bool(clone_audio) != bool(clone_text and clone_text.strip()):
        return "voice_clone_audio and voice_clone_text must be given together"
    if clone_audio:
        try:
            base64.b64decode(clone_audio, validate=True)
        except (binascii.Error, ValueError):
            return "voice_clone_audio is not valid base64"
        return None
    speaker = request.speaker_id
    if speaker is None:
        return None
    if speaker.strip().lower() in KNOWN_LANGUAGES or speaker in preset_voice_ids():
        return None
    return (
        f"Unknown speaker_id={speaker!r}. Available voices: {preset_voice_ids()}; "
        f"languages: {list(KNOWN_LANGUAGES)}. "
        "Or pass voice_clone_audio + voice_clone_text for ad-hoc cloning."
    )

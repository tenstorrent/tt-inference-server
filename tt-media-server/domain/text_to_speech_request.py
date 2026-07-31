# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional, Union

import numpy as np
from config.constants import TTS_RESPONSE_FORMATS
from domain.base_request import BaseRequest
from pydantic import PrivateAttr, field_validator, model_validator

# Default max text length (runner handles chunking internally)
DEFAULT_MAX_TTS_TEXT_LENGTH = 20000


def _validate_text_content(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    if len(value) > DEFAULT_MAX_TTS_TEXT_LENGTH:
        raise ValueError(
            f"{field_name} exceeds maximum length of {DEFAULT_MAX_TTS_TEXT_LENGTH} characters. "
            f"Received {len(value)} characters."
        )
    return value


class TextToSpeechRequest(BaseRequest):
    # Input text to convert to speech. Optional at the field level so a
    # request carrying only ``input`` (below) doesn't 422 before the
    # text/input resolution below gets a chance to run; the model validator
    # enforces that at least one of the two is actually present.
    text: Optional[str] = None

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, text):
        if text is None:
            return None
        return _validate_text_content(text, field_name="Text")

    # OpenAI SDK-compatible alias for ``text`` (their client's TTS param is
    # ``input``, not ``text``). ``text`` wins when both are present (matches
    # what Console sends today); ``input`` is only used as a fallback when
    # ``text`` is absent. Resolved into ``text`` itself (see the model
    # validator below) so no other code needs to know ``input`` exists.
    input: Optional[str] = None

    @model_validator(mode="after")
    def resolve_text_input_alias(self):
        if self.text:
            return self
        if self.input:
            self.text = _validate_text_content(self.input, field_name="input")
            return self
        raise ValueError("Either 'text' or 'input' is required")

    # Optional fields for speaker embedding
    speaker_embedding: Optional[Union[str, bytes]] = (
        None  # Base64-encoded or raw bytes of speaker embedding
    )
    speaker_id: Optional[str] = None  # ID for pre-configured speaker embeddings

    # Optional voice-clone ID (Inworld TTS runner): a voice_id previously
    # registered via POST /v1/audio/voices. When set, synthesis uses that
    # voice's registered VQ-code prompt instead of TVD (no-reference-audio) mode.
    voice_id: Optional[str] = None

    # OpenAI SDK-compatible alias for voice_id (their client only has a
    # "voice" param, e.g. defaulting to "alloy"). Resolved against the
    # registered voice list at the endpoint layer (open_ai_api/text_to_speech.py)
    # before dispatch, NOT here -- matching needs a live registry lookup that
    # a pydantic validator can't perform. voice_id always wins when both are
    # present; an unmatched voice (e.g. an OpenAI default like "alloy") is
    # silently ignored rather than treated as an error. Does not change
    # voice_id's own matching/error behavior in any way.
    voice: Optional[str] = None

    @field_validator("voice", mode="before")
    @classmethod
    def unwrap_voice_object(cls, v):
        # OpenAI's custom-voice object form: voice: {"id": "Ashley"} (as
        # opposed to the plain-string form, voice: "Ashley"). Extract "id"
        # and fall through the normal string field from there; a dict with
        # no "id" key resolves to None, same as if "voice" were absent.
        if isinstance(v, dict):
            return v.get("id")
        return v

    # Response format: wav (default), mp3, ogg, json, or verbose_json
    response_format: str = "wav"

    # Streaming (Inworld TTS runner only, currently): stream audio chunks
    # progressively as they're decoded instead of waiting for the whole
    # utterance and returning one complete response, mirroring
    # ChatCompletionRequest.stream's convention.
    stream: bool | None = False

    @field_validator("response_format", mode="before")
    @classmethod
    def validate_response_format(cls, v):
        normalized = str(v).strip().lower() if v is not None else "wav"
        if normalized not in TTS_RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {sorted(TTS_RESPONSE_FORMATS)}"
            )
        return normalized

    # Private fields for internal processing
    _speaker_embedding_array: Optional[np.ndarray] = PrivateAttr(default=None)
    _estimated_duration: Optional[float] = PrivateAttr(default=None)

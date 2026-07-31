# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
Request schema for POST /tts/v1/voice:stream, mirroring Inworld's own public
API contract exactly (see
https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-stream)
so client code written against real Inworld endpoints is a drop-in against
this deployment.

Not every field in the real contract is backed by real behavior here --
see each field's comment. Fields we cannot honor are REJECTED (422) when set
to a non-default value that would silently produce wrong output if ignored
(e.g. speakingRate); fields that are purely descriptive or best-effort
enhancements are accepted and silently unused (e.g. language, deliveryMode).
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Encodings we can actually produce. LINEAR16/PCM/WAV all mean "uncompressed
# PCM" in Inworld's own convention (LINEAR16 and PCM are headerless raw PCM;
# WAV is PCM wrapped in a WAV container) -- MP3/OGG_OPUS are real encodes via
# ffmpeg. FLAC/ALAW/MULAW are real values in Inworld's enum this deployment
# does not implement.
SUPPORTED_AUDIO_ENCODINGS = frozenset({"LINEAR16", "PCM", "WAV", "MP3", "OGG_OPUS"})
ALL_DOCUMENTED_AUDIO_ENCODINGS = SUPPORTED_AUDIO_ENCODINGS | frozenset({"FLAC", "ALAW", "MULAW"})

_DELIVERY_MODES = frozenset({"STABLE", "BALANCED", "CREATIVE"})
_TIMESTAMP_TYPES = frozenset({"WORD", "CHARACTER"})
_TIMESTAMP_TRANSPORT_STRATEGIES = frozenset({"SYNC", "ASYNC"})
_TEXT_NORMALIZATION_VALUES = frozenset({"ON", "OFF"})

MAX_TEXT_LENGTH = 2000


class AudioConfig(BaseModel):
    audioEncoding: str = "MP3"
    sampleRateHertz: int = 48000
    bitRate: int = 128000
    speakingRate: float = 1.0

    @field_validator("audioEncoding", mode="before")
    @classmethod
    def normalize_encoding(cls, v):
        return str(v).strip().upper() if v is not None else "MP3"

    @field_validator("audioEncoding")
    @classmethod
    def validate_encoding(cls, v):
        if v not in ALL_DOCUMENTED_AUDIO_ENCODINGS:
            raise ValueError(
                f"audioEncoding must be one of {sorted(ALL_DOCUMENTED_AUDIO_ENCODINGS)}"
            )
        if v not in SUPPORTED_AUDIO_ENCODINGS:
            raise ValueError(
                f"audioEncoding={v!r} is a real Inworld value but not implemented on this "
                f"deployment -- supported: {sorted(SUPPORTED_AUDIO_ENCODINGS)}"
            )
        return v

    @field_validator("sampleRateHertz")
    @classmethod
    def validate_sample_rate(cls, v):
        if not (8000 <= v <= 48000):
            raise ValueError("sampleRateHertz must be in range [8000, 48000]")
        return v

    @field_validator("speakingRate")
    @classmethod
    def validate_speaking_rate(cls, v):
        if not (0.5 <= v <= 1.5):
            raise ValueError("speakingRate must be in range [0.5, 1.5]")
        # A real numeric contract we cannot honor (no time-stretch capability) --
        # reject rather than silently produce audio at the wrong rate.
        if abs(v - 1.0) > 1e-6:
            raise ValueError(
                "speakingRate != 1.0 is not supported on this deployment "
                "(no speech-rate control implemented)"
            )
        return v


class SynthesisContextPreviousRequest(BaseModel):
    text: str


class SynthesisContext(BaseModel):
    previousRequests: List[SynthesisContextPreviousRequest] = Field(default_factory=list)


class InworldVoiceStreamRequest(BaseModel):
    text: str
    voiceId: str
    modelId: str
    audioConfig: AudioConfig = Field(default_factory=AudioConfig)

    # Descriptive/best-effort fields: accepted, not wired to any behavior on
    # this deployment. Left unvalidated beyond type/enum shape so real client
    # payloads that set them (to defaults or otherwise) don't 422.
    language: Optional[str] = None
    deliveryMode: str = "BALANCED"
    applyTextNormalization: Optional[str] = None
    enhanceGeneration: bool = False

    # Real knob: wired through to the underlying generation's temperature.
    temperature: float = 1.0

    # Not implemented -- rejected outright if the caller actually asks for
    # either (word/phoneme timestamps and multi-turn continuation require
    # capabilities this pipeline doesn't have).
    timestampType: Optional[str] = None
    timestampTransportStrategy: Optional[str] = None
    synthesisContext: Optional[SynthesisContext] = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters")
        return v

    @field_validator("voiceId")
    @classmethod
    def validate_voice_id(cls, v):
        if not v or not v.strip():
            raise ValueError("voiceId is required")
        return v

    @field_validator("modelId")
    @classmethod
    def validate_model_id(cls, v):
        if not v or not v.strip():
            raise ValueError("modelId is required")
        return v

    @field_validator("deliveryMode", mode="before")
    @classmethod
    def normalize_delivery_mode(cls, v):
        return str(v).strip().upper() if v is not None else "BALANCED"

    @field_validator("deliveryMode")
    @classmethod
    def validate_delivery_mode(cls, v):
        if v not in _DELIVERY_MODES:
            raise ValueError(f"deliveryMode must be one of {sorted(_DELIVERY_MODES)}")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not (0 < v <= 2):
            raise ValueError("temperature must be in range (0, 2]")
        return v

    @field_validator("timestampType")
    @classmethod
    def validate_timestamp_type(cls, v):
        if v is None:
            return v
        if v not in _TIMESTAMP_TYPES:
            raise ValueError(f"timestampType must be one of {sorted(_TIMESTAMP_TYPES)}")
        raise ValueError("timestampType is not supported on this deployment (no word/phoneme alignment)")

    @field_validator("timestampTransportStrategy")
    @classmethod
    def validate_timestamp_transport_strategy(cls, v):
        if v is None:
            return v
        if v not in _TIMESTAMP_TRANSPORT_STRATEGIES:
            raise ValueError(
                f"timestampTransportStrategy must be one of {sorted(_TIMESTAMP_TRANSPORT_STRATEGIES)}"
            )
        return v

    @field_validator("applyTextNormalization")
    @classmethod
    def validate_text_normalization(cls, v):
        if v is None:
            return v
        if v not in _TEXT_NORMALIZATION_VALUES:
            raise ValueError(
                f"applyTextNormalization must be one of {sorted(_TEXT_NORMALIZATION_VALUES)}"
            )
        return v

    @model_validator(mode="after")
    def reject_synthesis_context(self):
        if self.synthesisContext and self.synthesisContext.previousRequests:
            raise ValueError(
                "synthesisContext.previousRequests (multi-turn continuation) is not "
                "supported on this deployment"
            )
        return self

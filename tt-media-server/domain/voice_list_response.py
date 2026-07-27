# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import List, Optional

from pydantic import BaseModel


class VoiceInfo(BaseModel):
    """Metadata for a single cached voice-clone voice."""

    voice_id: str
    language: Optional[str] = None
    description: Optional[str] = None
    num_codes: int


class VoiceListResponse(BaseModel):
    """Response for GET /v1/audio/voices -- all cached voices with metadata."""

    voices: List[VoiceInfo]

    def to_dict(self):
        return {
            "voices": [
                {
                    "voice_id": v.voice_id,
                    "language": v.language,
                    "description": v.description,
                    "num_codes": v.num_codes,
                }
                for v in self.voices
            ]
        }

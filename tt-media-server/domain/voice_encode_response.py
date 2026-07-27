# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from pydantic import BaseModel


class VoiceEncodeResponse(BaseModel):
    """Response for registering a voice-clone audio prompt (-> a reusable voice_id)."""

    voice_id: str  # Identifier the caller can pass as TextToSpeechRequest.voice_id
    num_codes: int  # Number of VQ codes the reference audio was encoded into
    language: Optional[str] = None  # Echoed-back BCP-47 language tag, if provided
    description: Optional[str] = None  # Echoed-back description, if provided

    def to_dict(self):
        return {
            "voice_id": self.voice_id,
            "num_codes": self.num_codes,
            "language": self.language,
            "description": self.description,
        }

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from pydantic import BaseModel


class VoiceEncodeResponse(BaseModel):
    """Response for registering a voice-clone audio prompt (-> a reusable voice_id)."""

    voice_id: str  # Identifier the caller can pass as TextToSpeechRequest.voice_id
    num_codes: int  # Number of VQ codes the reference audio was encoded into

    def to_dict(self):
        return {
            "voice_id": self.voice_id,
            "num_codes": self.num_codes,
        }

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional, Union

from domain.base_request import BaseRequest


class VoiceEncodeRequest(BaseRequest):
    # Required fields
    reference_audio: Union[str, bytes]  # Base64-encoded audio string OR raw audio bytes

    # Optional fields
    voice_id: Optional[str] = (
        None  # Caller-supplied voice ID to register under; a new one is generated if omitted
    )
    language: Optional[str] = None  # Optional BCP-47 language tag for the voice
    description: Optional[str] = None  # Optional human-readable description of the voice

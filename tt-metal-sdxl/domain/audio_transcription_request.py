# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    file: str  # Base64-encoded audio file
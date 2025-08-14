# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.base_request import BaseRequest

class AudioTranscriptionRequest(BaseRequest):
    file: bytes
    model: str = "whisper-1"
    
    def get_model_input(self):
        return self.file
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Union
from domain.base_request import BaseRequest
from starlette.datastructures import UploadFile

class AudioTranscriptionRequest(BaseRequest):
    file: Union[UploadFile, bytes]
    model: str = "whisper-1"
    
    def get_model_input(self):
        return self.file
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Union
from domain.base_request import BaseRequest
from starlette.datastructures import UploadFile

class AudioTranscriptionRequest(BaseRequest):
    file: Union[UploadFile, bytes]
    model: str = "whisper-1"
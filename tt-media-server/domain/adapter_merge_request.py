# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from domain.base_request import BaseRequest
from pydantic import PrivateAttr


class AdapterMergeRequest(BaseRequest):
    source_job_id: str
    checkpoint_id: str

    _adapter_path: str = PrivateAttr(default=None)
    _output_model_path: str = PrivateAttr(default=None)

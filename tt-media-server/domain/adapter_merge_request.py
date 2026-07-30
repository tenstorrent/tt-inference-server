# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from domain.base_request import BaseRequest
from pydantic import PrivateAttr


class AdapterMergeRequest(BaseRequest):
    """Request to merge a stored LoRA adapter checkpoint into its base model.

    Produces a full, standalone HuggingFace checkpoint servable by the vLLM
    inference container. Merging is CPU-only and does not use the accelerator.
    """

    source_job_id: str
    checkpoint_id: str

    # Resolved server-side before the job is scheduled.
    _adapter_path: str = PrivateAttr(default=None)
    _output_model_path: str = PrivateAttr(default=None)

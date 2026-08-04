# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Optional

from domain.base_request import BaseRequest
from pydantic import Field


# ``VideoShm.write_request`` packs the seed as a signed 64-bit int ("<q"), so a
# value outside this range raises ``struct.error`` inside the device worker.
# Bounding it here turns that into a 422 at the boundary instead of a 500 plus an
# increment on the worker's error count (#4811).
_MIN_SEED = -(2**63)
_MAX_SEED = 2**63 - 1


class VideoGenerateRequest(BaseRequest):
    # Required fields
    prompt: str

    # Optional fields
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=20, ge=12, le=50)
    seed: Optional[int] = Field(default=None, ge=_MIN_SEED, le=_MAX_SEED)

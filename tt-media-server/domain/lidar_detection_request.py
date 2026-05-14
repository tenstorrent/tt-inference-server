# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from domain.base_request import BaseRequest
from pydantic import Field


class LidarDetectionRequest(BaseRequest):
    """Request for 3D object detection from a raw LiDAR point cloud.

    The prompt field carries the raw binary content of a nuScenes-format
    LiDAR scan (.bin file), interpreted as a flat array of float32 values
    with 5 channels per point: x, y, z, intensity, ring.
    """

    prompt: bytes
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

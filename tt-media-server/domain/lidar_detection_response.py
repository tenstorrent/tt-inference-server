# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from typing import List

from pydantic import BaseModel

TASK_NAMES = [
    "car",
    "truck+construction_vehicle",
    "bus+trailer",
    "barrier",
    "motorcycle+bicycle",
    "pedestrian+traffic_cone",
]


class BoundingBox3D(BaseModel):
    """Single 3D bounding box prediction."""

    cx: float
    cy: float
    cz: float
    width: float
    length: float
    height: float
    yaw: float
    vx: float
    vy: float
    score: float
    label: int


class TaskDetections(BaseModel):
    """Detections for one CenterPoint task group."""

    task_name: str
    detections: List[BoundingBox3D]


class LidarDetectionResponse(BaseModel):
    """Full CenterPoint 3D detection response."""

    tasks: List[TaskDetections]
    num_detections: int

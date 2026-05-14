# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import logging

from domain.lidar_detection_request import LidarDetectionRequest
from domain.lidar_detection_response import LidarDetectionResponse
from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/detect", response_model=LidarDetectionResponse)
async def detect(
    file: UploadFile = File(...),
    score_threshold: float = Form(default=0.1),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
) -> LidarDetectionResponse:
    """
    Run CenterPoint 3D object detection on a LiDAR point cloud.

    Accepts a nuScenes-format .bin file (float32 array, 5 channels per point:
    x, y, z, intensity, ring).

    Returns per-task 3D bounding box detections.
    """
    try:
        file_content = await file.read()
        request = LidarDetectionRequest(
            prompt=file_content,
            score_threshold=score_threshold,
        )
        return await service.process_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

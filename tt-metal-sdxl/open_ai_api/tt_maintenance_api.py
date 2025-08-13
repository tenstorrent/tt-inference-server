# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver

router = APIRouter()


@router.get('/tt-liveness')
def liveness(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Check service liveness and model readiness.

    Returns:
        dict: Dictionary containing service status and model information.

    Raises:
        HTTPException: If service is unavailable or model check fails.
    """
    try:
        return {'status': 'alive', **service.check_is_model_ready()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {e}")

@router.post('/tt-deep-reset')
async def deep_reset(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Schedules a deep reset of the service and its model.

    Returns:
        dict: Status message indicating the result of the reset operation.

    Raises:
        HTTPException: If reset fails.
    """
    try:
        await service.deep_reset()
        return {'status': 'reset successful'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")
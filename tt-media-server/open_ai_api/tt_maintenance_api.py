# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver

router = APIRouter()

# Error kinds that mean a worker is not just failing but actively stuck. Any
# hit here flips overall_state to "hung" regardless of is_ready, because a
# worker can satisfy check_is_model_ready() and then hang on the next request.
_HANG_ERROR_KINDS = {"warmup_timeout", "request_timeout", "device_hang"}


def _overall_state(status: dict[str, Any]) -> str:
    """Collapse per-worker fields into a single summary string.

    Priority: hung > failed > starting > degraded > healthy. "hung" means we
    have a greppable timeout / tt-metal hang signature within the current
    worker generation; "failed" means a non-hang startup error.
    """
    workers: dict = status.get("worker_info") or {}
    if not workers:
        return "starting"

    any_hung = any(
        w.get("last_error_kind") in _HANG_ERROR_KINDS for w in workers.values()
    )
    if any_hung:
        return "hung"

    any_startup_error = any(
        w.get("last_error_kind") == "startup_error" for w in workers.values()
    )
    ready_flags = [bool(w.get("is_ready")) for w in workers.values()]

    if any_startup_error and not any(ready_flags):
        return "failed"
    if all(ready_flags):
        return "healthy"
    if any(ready_flags):
        return "degraded"
    return "starting"


@router.get("/tt-liveness")
def liveness(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Check service liveness and model readiness.

    Returns:
        dict: Dictionary containing service status and model information.

    Raises:
        HTTPException: If service is unavailable or model check fails.
    """
    try:
        return {"status": "alive", **service.check_is_model_ready()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {e}")


@router.get("/tt-status")
def tt_status(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Detailed health + hang-detection status for operators.

    Extends the data from /tt-liveness with per-worker last-error triage and a
    collapsed `overall_state` field ("healthy" | "degraded" | "starting" |
    "hung" | "failed"). "hung" fires when any worker has recorded a warmup
    timeout, request timeout, or native tt-metal hang signature; on that
    signal the operator can trigger a reset without having to scrape logs.

    Unlike /health and /tt-liveness, this endpoint never raises on a bad
    model state — it reports it. This is intentional so orchestrators can
    poll it even while the service is unhealthy.
    """
    try:
        status = service.check_is_model_ready()
    except Exception:
        return {"overall_state": "failed", "error": "status check failed"}

    return {"overall_state": _overall_state(status), **status}


@router.get("/health")
def health(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    OpenAI-compatible health check endpoint.
    Returns 200 OK when service and model are ready, compatible with vLLM health endpoint.
    Returns:
        dict: Empty dict when healthy (matches vLLM behavior).
    Raises:
        HTTPException: If service is unavailable or model is not ready.
    """
    try:
        status = service.check_is_model_ready()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

    if not status.get("model_ready", False):
        raise HTTPException(status_code=503, detail="Model not ready")
    return {}


@router.post("/tt-deep-reset")
async def deep_reset(
    service: BaseService = Depends(service_resolver),
) -> dict[str, Any]:
    """
    Schedules a deep reset of the service and its model.

    Returns:
        dict: Status message indicating the result of the reset operation.

    Raises:
        HTTPException: If reset fails.
    """
    try:
        await service.deep_reset()
        return {"status": "Reset scheduled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@router.post("/tt-reset-device")
async def reset_device(
    device_id: str, service: BaseService = Depends(service_resolver)
) -> dict[str, Any]:
    """
    Schedules a single device reset

    Returns:
        dict: Status message indicating the result of the reset operation.

    Raises:
        HTTPException: If reset fails.
    """
    try:
        await service.device_reset(device_id)
        return {"status": f"Reset of device {device_id} scheduled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

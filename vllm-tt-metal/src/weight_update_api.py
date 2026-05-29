# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Runtime weight-update control plane for the TT vLLM server.

Adds a small set of internal HTTP endpoints to the vLLM OpenAI API server so
an external RL trainer (e.g. tt-training-service) can hot-swap the policy
weights of a live inference server without restarting it:

    POST /v1/internal/weights/update          -> apply a new checkpoint
    GET  /v1/internal/weights/version         -> current weights/policy version
    POST /v1/internal/weights/reset_prefix_cache  -> flush prefix/KV cache

These endpoints reach the engine via ``app.state.engine_client`` (set by
vLLM's ``init_app_state``) and invoke the worker's ``update_weights`` method
through ``collective_rpc``. The actual in-place on-device overwrite lives in
the worker (tt-vllm-plugin ``TTWorker.update_weights``) and the tt-metal
model (``Generator.update_weights`` / ``Transformer.update``).

Wiring: ``install(...)`` monkeypatches ``vllm.entrypoints.openai.api_server``
``build_app`` so the router is mounted on the same FastAPI app (and behind the
same API-key auth) as the OpenAI routes. This avoids forking vLLM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/internal/weights", tags=["tt-weight-update"])


class WeightUpdateRequest(BaseModel):
    weights_path: str = Field(
        ...,
        description=(
            "Path to an HF-format checkpoint, reachable from the inference "
            "container's filesystem (e.g. a bind-mounted MODEL_WEIGHTS_DIR or "
            "a shared/NFS checkpoint directory written by the trainer)."
        ),
    )
    version: Optional[int] = Field(
        default=None,
        description=(
            "Optional caller-assigned weights/policy version to record. If "
            "omitted, the server increments its internal counter."
        ),
    )


class WeightUpdateResponse(BaseModel):
    status: str
    version: int
    workers: list[dict[str, Any]]


class WeightsVersionResponse(BaseModel):
    version: int


def _engine_client(request: Request):
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(
            status_code=503,
            detail="Engine client not initialized yet; server is still starting.",
        )
    if not hasattr(engine_client, "collective_rpc"):
        raise HTTPException(
            status_code=501,
            detail=(
                "Engine client does not support collective_rpc; runtime weight "
                "update is unavailable in this vLLM build."
            ),
        )
    return engine_client


@router.post("/update", response_model=WeightUpdateResponse)
async def update_weights(body: WeightUpdateRequest, request: Request):
    """Apply an in-place weight update to the live model.

    Note on quiescing: ``collective_rpc`` is serialized with engine scheduler
    steps, so the update never interleaves with a single ``execute_model``
    call. Requests already in flight will, however, finish decoding under a
    mix of old/new weights. For on-policy correctness the trainer should stop
    submitting rollouts and let the server drain before calling this endpoint;
    every response is tagged with the version returned here so stale samples
    can be detected.
    """
    engine_client = _engine_client(request)

    try:
        results = await engine_client.collective_rpc(
            "update_weights",
            kwargs={"weights_path": body.weights_path, "version": body.version},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface engine errors to caller
        logger.exception("Weight update failed")
        raise HTTPException(status_code=500, detail=f"Weight update failed: {exc}")

    results = results or []
    # The model-owning worker reports the authoritative new version.
    applied = [r for r in results if isinstance(r, dict) and r.get("updated")]
    if not applied:
        raise HTTPException(
            status_code=500,
            detail=(
                "No worker applied the weight update (model-owning rank did "
                f"not report success). Raw results: {results}"
            ),
        )
    version = applied[0].get("version")
    request.app.state.tt_weights_version = version
    logger.info("Weight update applied; weights_version=%s", version)
    return WeightUpdateResponse(status="ok", version=version, workers=results)


@router.get("/version", response_model=WeightsVersionResponse)
async def get_version(request: Request):
    engine_client = _engine_client(request)
    try:
        results = await engine_client.collective_rpc("get_weights_version")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to query weights version")
        raise HTTPException(
            status_code=500, detail=f"Failed to query weights version: {exc}"
        )
    version = next((r for r in (results or []) if isinstance(r, int)), 0)
    return WeightsVersionResponse(version=version)


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(request: Request):
    """Flush the prefix/KV cache across a weight-version boundary.

    On the TT backend prefix caching is currently disabled, so this is a
    best-effort no-op today, but it is exposed for forward compatibility and
    parity with other backends.
    """
    engine_client = _engine_client(request)
    reset = getattr(engine_client, "reset_prefix_cache", None)
    if reset is None:
        return {"status": "noop", "detail": "engine has no reset_prefix_cache"}
    try:
        await reset()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"reset_prefix_cache failed: {exc}"
        )
    return {"status": "ok"}


def install() -> None:
    """Monkeypatch vLLM's ``build_app`` to mount the weight-update router.

    Must be called before the OpenAI API server module is run (i.e. before
    ``runpy.run_module("vllm.entrypoints.openai.api_server", ...)``).
    Idempotent.
    """
    import vllm.entrypoints.openai.api_server as api_server

    if getattr(api_server.build_app, "_tt_weight_update_patched", False):
        return

    original_build_app = api_server.build_app

    def build_app_with_weight_update(args):
        app = original_build_app(args)
        app.include_router(router)
        app.state.tt_weights_version = 0
        logger.info(
            "Mounted TT weight-update routes under /v1/internal/weights"
        )
        return app

    build_app_with_weight_update._tt_weight_update_patched = True  # type: ignore[attr-defined]
    api_server.build_app = build_app_with_weight_update

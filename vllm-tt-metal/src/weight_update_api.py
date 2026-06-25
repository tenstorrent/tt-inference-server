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
through ``collective_rpc``. The worker (tt-vllm-plugin ``TTWorker``) receives
the new weights as an HF-keyed dict over tt-metal's ``WeightBridge`` (PR
#45734) and applies them in place via the model's
``update_weights(hf_dict, hf_rope=...)`` (tt-metal
``Transformer.update_weights`` / per-module ``update``).

Wiring: ``install(...)`` monkeypatches ``vllm.entrypoints.openai.api_server``
``build_app`` so the router is mounted on the same FastAPI app (and behind the
same API-key auth) as the OpenAI routes. This avoids forking vLLM.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/internal/weights", tags=["tt-weight-update"])

# Separate router for non-weights control-plane endpoints (e.g. shutdown) that
# live directly under /v1/internal rather than /v1/internal/weights.
control_router = APIRouter(prefix="/v1/internal", tags=["tt-control"])


class WeightUpdateRequest(BaseModel):
    sender_rank: int = Field(
        default=0,
        description=(
            "Distributed-context (MPI) rank of the training process that sends "
            "the weights -- the WeightBridge sender (TTML_RANK, default 0). The "
            "trainer streams the new weights device-to-device over TT-Fabric "
            "into the inference mesh; the worker is the bridge receiver "
            "(role='ttt', TTT_RANK)."
        ),
    )
    version: Optional[int] = Field(
        default=None,
        description=(
            "Optional caller-assigned weights/policy version to record. If "
            "omitted, the server increments its internal counter."
        ),
    )
    hf_rope: bool = Field(
        default=False,
        description=(
            "Forwarded to the model's update_weights(). False means Q/K rows "
            "are already in the inference model's RoPE convention (correct for "
            "the ttml -> tt-transformers Llama transfer)."
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
    """Apply an in-place weight update streamed over a device socket.

    The trainer (sender_rank) is the ``WeightBridge`` sender: it ships a
    JSON manifest over host MPI then streams every weight tensor over a fabric
    ``MeshSocket``. The inference worker is the bridge receiver; it
    materializes the HF-keyed dict and copies each tensor in place. This
    endpoint only triggers/awaits the receive on the worker side.

    Note on quiescing: ``collective_rpc`` is serialized with engine scheduler
    steps, so the update never interleaves with a single ``execute_model``
    call. Requests already in flight will, however, finish decoding under a
    mix of old/new weights. For on-policy correctness the trainer should stop
    submitting rollouts and let the server drain before calling this endpoint;
    every response is tagged with the version returned here so stale samples
    can be detected.

    The caller must coordinate timing: both ranks must reach
    ``bridge.connect()`` (and the final ``bridge.barrier()``) together, so the
    trainer should call its send side around the same time as this request.
    """
    engine_client = _engine_client(request)

    try:
        results = await engine_client.collective_rpc(
            "update_weights",
            kwargs={
                "sender_rank": body.sender_rank,
                "version": body.version,
                "hf_rope": body.hf_rope,
            },
        )
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


class ShutdownResponse(BaseModel):
    status: str


@control_router.post("/shutdown", response_model=ShutdownResponse)
async def shutdown(request: Request):
    """Gracefully stop this (co-located) vLLM server.

    The co-located trainer runs in the SAME single mpirun world as this server
    and calls this after its final weight push. Without it the server -- a
    long-running process with no self-exit -- keeps the mpirun world alive, so
    the launcher hangs until ``scancel``/walltime and leaks device-holding
    children. We send ``SIGTERM`` to this (APIServer) process AFTER the response
    is flushed; vLLM's signal handler then tears down the EngineCore + workers
    cleanly. Fired from a short-delay background thread so the caller still gets
    its ``200``.
    """
    delay_s = 0.5

    def _terminate() -> None:
        time.sleep(delay_s)
        logger.info(
            "Shutdown requested via /v1/internal/shutdown; sending SIGTERM to self (pid=%s)",
            os.getpid(),
        )
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_terminate, daemon=True).start()
    logger.info("Shutdown requested; server will stop in %.1fs", delay_s)
    return ShutdownResponse(status="stopping")


def install() -> None:
    """Mount the weight-update router on the vLLM OpenAI API server FastAPI app.

    Must be called before the OpenAI API server module is run. Idempotent.

    Implementation note -- why we hook ``fastapi.FastAPI.__init__`` rather than
    vLLM's ``build_app``:

    ``run_vllm_api_server.py`` launches the server via
    ``runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")``,
    which RE-EXECUTES that module's source in a brand-new ``__main__`` namespace.
    A monkeypatch of ``api_server.build_app`` on the already-imported module
    object is therefore shadowed by the freshly-redefined ``build_app`` and is
    never consulted -- the server mounts only its stock routes and every
    ``/v1/internal/weights/*`` call 404s. (At launch, ``runpy`` even warns:
    "'vllm.entrypoints.openai.api_server' found in sys.modules ... prior to
    execution ...; this may result in unpredictable behaviour".)

    ``fastapi`` is imported once and is NOT re-executed by ``runpy``, so a hook
    on ``FastAPI.__init__`` survives the re-exec. Every FastAPI app built in
    this process (in practice just the vLLM server app) gets the router mounted
    immediately after construction; the API-key auth middleware vLLM adds later
    is app-level (Starlette) and so still covers these routes.
    """
    import fastapi

    if getattr(fastapi.FastAPI.__init__, "_tt_weight_update_patched", False):
        return

    original_init = fastapi.FastAPI.__init__

    def __init___with_weight_update(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Guard against double-mounting (e.g. nested/sub-apps or a re-entrant
        # construction): only the first init per app installs the router.
        if getattr(self.state, "_tt_weight_update_mounted", False):
            return
        self.include_router(router)
        self.include_router(control_router)
        self.state._tt_weight_update_mounted = True
        self.state.tt_weights_version = 0
        logger.info(
            "Mounted TT weight-update routes under /v1/internal/weights "
            "and control routes under /v1/internal"
        )

    __init___with_weight_update._tt_weight_update_patched = True  # type: ignore[attr-defined]
    fastapi.FastAPI.__init__ = __init___with_weight_update

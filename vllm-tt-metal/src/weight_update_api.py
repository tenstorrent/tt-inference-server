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

Applying an update is quiesced server-side: an admission-gate middleware
rejects new inference requests with 503 while in-flight requests are drained,
so no request straddles the weight-version boundary (see ``update_weights``).

Wiring: ``install(...)`` monkeypatches ``vllm.entrypoints.openai.api_server``
``build_app`` so the router (and the admission-gate middleware) is mounted on
the same FastAPI app (and behind the same API-key auth) as the OpenAI routes.
This avoids forking vLLM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/internal/weights", tags=["tt-weight-update"])

# Requests to these path prefixes stay reachable while a weight update is in
# progress (control plane + liveness/metrics), so probes don't flap and the
# trainer can still drive/observe the update.
_GATE_EXEMPT_PREFIXES = ("/v1/internal", "/health", "/ping", "/metrics")

# How long to wait for in-flight requests to drain before applying the update.
# On timeout the stragglers are aborted so the update is never blocked forever.
_DRAIN_TIMEOUT_ENV = "TT_WEIGHT_UPDATE_DRAIN_TIMEOUT_SECONDS"
_DEFAULT_DRAIN_TIMEOUT_S = 60.0
_DRAIN_POLL_INTERVAL_S = 0.05


class _AdmissionGateMiddleware:
    """Reject new inference requests with 503 while a weight update is applied.

    Pure ASGI middleware (not ``BaseHTTPMiddleware``) so it never buffers or
    wraps response bodies -- SSE streaming from the OpenAI endpoints is left
    untouched. It only short-circuits *new* requests when the gate is closed;
    the control-plane / health paths in ``_GATE_EXEMPT_PREFIXES`` are always
    let through.
    """

    def __init__(self, app: ASGIApp, gate_state: Any) -> None:
        self.app = app
        self._gate_state = gate_state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and getattr(
            self._gate_state, "_tt_weight_update_in_progress", False
        ):
            path = scope.get("path", "")
            if not path.startswith(_GATE_EXEMPT_PREFIXES):
                response = JSONResponse(
                    {
                        "error": {
                            "message": "Weight update in progress; retry shortly.",
                            "type": "server_busy",
                        }
                    },
                    status_code=503,
                    headers={"Retry-After": "1"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _drain_inflight(engine_client: Any) -> None:
    """Wait for frontend-tracked in-flight requests to finish, then return.

    Uses ``AsyncLLM.output_processor.has_unfinished_requests()`` -- the only
    in-flight accounting exposed to the API process on vLLM V1. If the engine
    client does not expose it (older/other build), we skip draining: the
    ``collective_rpc`` between-steps guarantee still prevents mid-step
    corruption, we just can't wait out multi-step requests.
    """
    output_processor = getattr(engine_client, "output_processor", None)
    has_unfinished = getattr(output_processor, "has_unfinished_requests", None)
    if has_unfinished is None:
        logger.warning(
            "Engine client exposes no in-flight accounting; applying weight "
            "update without draining multi-step requests."
        )
        return

    try:
        timeout_s = float(os.getenv(_DRAIN_TIMEOUT_ENV, _DEFAULT_DRAIN_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout_s = _DEFAULT_DRAIN_TIMEOUT_S

    deadline = time.monotonic() + timeout_s
    while has_unfinished():
        if time.monotonic() >= deadline:
            request_states = getattr(output_processor, "request_states", {})
            stragglers = list(request_states)
            logger.warning(
                "Drain timed out after %.1fs with %d in-flight request(s); "
                "aborting them so the weight update can proceed.",
                timeout_s,
                len(stragglers),
            )
            if stragglers and hasattr(engine_client, "abort"):
                await engine_client.abort(stragglers)
            return
        await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)


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


async def _apply_weight_update(engine_client: Any, body: WeightUpdateRequest) -> list:
    """Run the worker-side weight swap and return the raw per-worker results."""
    try:
        results = await engine_client.collective_rpc(
            "update_weights",
            kwargs={
                "sender_rank": body.sender_rank,
                "hf_rope": body.hf_rope,
            },
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface engine errors to caller
        logger.exception("Weight update failed")
        raise HTTPException(status_code=500, detail=f"Weight update failed: {exc}")

    # Flush prefix/KV cache across the version boundary (best-effort; a no-op
    # on the TT backend today, but correct for caching backends).
    reset = getattr(engine_client, "reset_prefix_cache", None)
    if reset is not None:
        try:
            await reset()
        except Exception:  # noqa: BLE001 - cache flush is best-effort
            logger.exception("reset_prefix_cache after weight update failed")

    return results or []


@router.post("/update", response_model=WeightUpdateResponse)
async def update_weights(body: WeightUpdateRequest, request: Request):
    """Apply an in-place weight update streamed over a device socket.

    The trainer (sender_rank) is the ``WeightBridge`` sender: it ships a
    JSON manifest over host MPI then streams every weight tensor over a fabric
    ``MeshSocket``. The inference worker is the bridge receiver; it
    materializes the HF-keyed dict and copies each tensor in place. This
    endpoint only triggers/awaits the receive on the worker side.

    Quiescing is enforced server-side so no request spans the weight-version
    boundary: for the duration of this call new inference requests are rejected
    with 503 (admission gate), in-flight requests are drained (or aborted after
    ``TT_WEIGHT_UPDATE_DRAIN_TIMEOUT_SECONDS``), and only then is the swap
    applied via ``collective_rpc`` (which itself runs between engine steps, so
    it never interleaves a single ``execute_model``). Admission resumes once
    the update completes.

    """
    engine_client = _engine_client(request)
    app_state = request.app.state

    lock = getattr(app_state, "_tt_weight_update_lock", None)
    if lock is None:
        # Defensive: install() normally creates this; fall back so a stray build
        # still serializes updates.
        lock = asyncio.Lock()
        app_state._tt_weight_update_lock = lock

    async with lock:
        app_state._tt_weight_update_in_progress = True
        try:
            await _drain_inflight(engine_client)
            results = await _apply_weight_update(engine_client, body)
        finally:
            app_state._tt_weight_update_in_progress = False

    # The model-owning worker owns the version counter and reports the
    # authoritative new value.
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
    """Mount the weight-update router on the vLLM OpenAI API server FastAPI app.

    Must be called before the server module runs. Idempotent.

    We hook ``fastapi.FastAPI.__init__`` rather than vLLM's ``build_app``:
    ``runpy.run_module(..., run_name="__main__")`` re-executes the api_server
    module, so a ``build_app`` monkeypatch is shadowed by the redefinition and
    never runs. ``fastapi`` is not re-executed, so an ``__init__`` hook survives
    and mounts the router on the server app (auth middleware is app-level and
    still covers these routes).
    """
    # Defense-in-depth: these routes are strictly for the co-located RL trainer.
    # Even a stray/future call to install() must be inert off the RL path so the
    # process-wide fastapi.FastAPI.__init__ monkeypatch never happens (and the
    # internal weight-update routes never mount) on a normal server.
    if os.getenv("TT_COLOCATED_INFERENCE") != "1":
        return

    import fastapi

    if getattr(fastapi.FastAPI.__init__, "_tt_weight_update_patched", False):
        return

    # Capture the unpatched __init__ as a default arg so the wrapper can call it
    # without an enclosing-scope reference.
    def __init___with_weight_update(
        self, *args, _original_init=fastapi.FastAPI.__init__, **kwargs
    ):
        _original_init(self, *args, **kwargs)
        # Guard against double-mounting (e.g. nested/sub-apps or a re-entrant
        # construction): only the first init per app installs the router.
        if getattr(self.state, "_tt_weight_update_mounted", False):
            return
        self.include_router(router)
        self.state._tt_weight_update_mounted = True
        # Quiesce state: the gate flag the middleware reads, and a lock that
        # serializes concurrent /update calls.
        self.state._tt_weight_update_in_progress = False
        self.state._tt_weight_update_lock = asyncio.Lock()
        self.add_middleware(_AdmissionGateMiddleware, gate_state=self.state)
        logger.info(
            "Mounted TT weight-update routes under /v1/internal/weights "
            "(with weight-update admission gate)"
        )

    __init___with_weight_update._tt_weight_update_patched = True  # type: ignore[attr-defined]
    fastapi.FastAPI.__init__ = __init___with_weight_update

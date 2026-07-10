# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Parallel top-up ``vllm bench serve`` load during agentic evals.

Agentic tasks only occupy ``n_concurrent_trials`` request slots on the remote
SUPER_CLUSTER endpoint (bounded by runner CPU/Docker, not the model). To
exercise the endpoint at its real ``max_concurrency`` we run synthetic
``vllm bench serve`` load in parallel that tops the concurrency up to the
model ceiling:

    bench_concurrency = max_concurrency - n_concurrent_trials

The synthetic ISL is chosen to mirror the agentic load:

* Harbor tasks (terminal-bench, tau3): steered live from the running trial
  logs via :class:`~llm_module.agentic.live_isl.LiveISLTracker`.
* SWE-bench (mini-swe-agent, not Harbor): a fixed 200K ISL / 4K OSL point,
  clamped to fit ``max_context``.

The load runs as back-to-back bench segments on a background thread until the
agentic task finishes and the caller calls :meth:`ParallelBenchLoad.stop`.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from ..config import DriverContext, LLMRunConfig, ServerConnection
from ..drivers.vllm import VLLMBenchDriver
from .live_isl import LiveISLTracker

logger = logging.getLogger(__name__)

# Fixed synthetic point for SWE-bench (mini-swe-agent) parallel load.
SWEBENCH_PARALLEL_ISL = 200 * 1024  # 200K
SWEBENCH_PARALLEL_OSL = 4 * 1024  # 4K
# Short output for Harbor tasks so the load tracks input growth, not decode.
HARBOR_PARALLEL_OSL = 128
# ISL used before the live reader has its first sample.
HARBOR_DEFAULT_ISL = 8192


def _resolve_vllm_binary() -> str:
    """Locate the ``vllm`` CLI in the benchmarks venv, else fall back to PATH."""
    try:
        from workflows.workflow_types import WorkflowVenvType
        from workflows.workflow_venvs import VENV_CONFIGS

        venv_python = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_VLLM].venv_python
        candidate = Path(venv_python).parent / "vllm"
        if candidate.exists():
            return str(candidate)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not resolve BENCHMARKS_VLLM venv vllm binary: %s", e)
    return shutil.which("vllm") or "vllm"


def _clamp_isl(isl: int, osl: int, max_context: int) -> int:
    """Keep ``isl + osl <= max_context`` with at least 1 input token."""
    if isl + osl <= max_context:
        return max(1, isl)
    return max(1, max_context - osl)


class ParallelBenchLoad:
    """Runs back-to-back ``vllm bench serve`` segments on a background thread."""

    def __init__(
        self,
        *,
        driver: VLLMBenchDriver,
        server: ServerConnection,
        context: DriverContext,
        bench_concurrency: int,
        max_context: int,
        osl: int,
        fixed_isl: Optional[int],
        tracker: Optional[LiveISLTracker],
    ) -> None:
        self._driver = driver
        self._server = server
        self._context = context
        self._bench_concurrency = int(bench_concurrency)
        self._max_context = int(max_context)
        self._osl = int(osl)
        self._fixed_isl = fixed_isl
        self._tracker = tracker
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ParallelBenchLoad":
        if self._tracker is not None:
            self._tracker.start()
        self._thread = threading.Thread(
            target=self._loop, name="parallel-bench-load", daemon=True
        )
        self._thread.start()
        logger.info(
            "Started parallel bench load: concurrency=%d osl=%d (%s ISL)",
            self._bench_concurrency,
            self._osl,
            "fixed" if self._fixed_isl is not None else "live",
        )
        return self

    def stop(self, timeout_s: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None
        if self._tracker is not None:
            self._tracker.stop()
        logger.info("Stopped parallel bench load.")

    def __enter__(self) -> "ParallelBenchLoad":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _current_isl(self) -> int:
        if self._fixed_isl is not None:
            isl = self._fixed_isl
        elif self._tracker is not None:
            isl = self._tracker.current_isl()
        else:
            isl = HARBOR_DEFAULT_ISL
        return _clamp_isl(isl, self._osl, self._max_context)

    def _loop(self) -> None:
        # Overwrite one stable file each segment: we only keep the latest run,
        # not thousands of per-segment files. The ISL used is still recoverable
        # from the file (total_input_tokens / completed) and the concurrency
        # from max_concurrency.
        result_filename = self._context.output_dir / "parallel_bench.json"
        while not self._stop.is_set():
            isl = self._current_isl()
            config = LLMRunConfig(
                isl=isl,
                osl=self._osl,
                max_concurrency=self._bench_concurrency,
                num_prompts=self._bench_concurrency,
            )
            try:
                self._driver.run(
                    config,
                    self._server,
                    self._context,
                    result_filename=result_filename,
                )
            except Exception as e:  # pragma: no cover - defensive
                # A single failed segment must never break the agentic eval.
                logger.warning("Parallel bench segment failed (isl=%d): %s", isl, e)
                # Back off briefly so a hard failure loop doesn't spin.
                self._stop.wait(10.0)


def start_parallel_bench(
    ctx,
    task,
    *,
    watch_dir: Path,
    sidecar_dir: Path,
) -> Optional[ParallelBenchLoad]:
    """Build and start a :class:`ParallelBenchLoad` for ``task``, or ``None``.

    Returns ``None`` (no-op) when the feature does not apply: non
    SUPER_CLUSTER models, or when the agentic trials already saturate the
    model concurrency (``bench_concurrency < 1``).
    """
    from workflows.workflow_types import DeviceTypes

    device_type = getattr(ctx.model_spec, "device_type", None)
    if device_type != DeviceTypes.SUPER_CLUSTER:
        return None

    device_spec = ctx.model_spec.device_model_spec
    model_max = int(device_spec.max_concurrency)
    max_context = int(device_spec.max_context)

    is_swebench = getattr(task, "swebench_eval_config", None) is not None
    eval_cfg = task.agentic_eval_config or task.swebench_eval_config
    n_concurrent_trials = int(getattr(eval_cfg, "n_concurrent_trials", 0) or 0)

    bench_concurrency = model_max - n_concurrent_trials
    if bench_concurrency < 1:
        logger.info(
            "Skipping parallel bench for %s: n_concurrent_trials=%d >= "
            "max_concurrency=%d",
            task.task_name,
            n_concurrent_trials,
            model_max,
        )
        return None

    auth_token = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or ""
    server = ServerConnection(
        base_url=ctx.server_url if ctx.remote_server else ctx.server_host,
        service_port=ctx.server_port,
        model=ctx.model_spec.hf_model_repo,
        auth_token=auth_token,
        is_remote=ctx.remote_server,
    )
    driver_context = DriverContext(
        output_dir=Path(sidecar_dir),
        device=ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device),
    )
    driver = VLLMBenchDriver(vllm_binary=_resolve_vllm_binary())

    if is_swebench:
        osl = SWEBENCH_PARALLEL_OSL
        fixed_isl: Optional[int] = SWEBENCH_PARALLEL_ISL
        tracker: Optional[LiveISLTracker] = None
    else:
        osl = HARBOR_PARALLEL_OSL
        fixed_isl = None
        tracker = LiveISLTracker(
            watch_dir=watch_dir,
            hf_model_repo=ctx.model_spec.hf_model_repo,
            default_isl=_clamp_isl(HARBOR_DEFAULT_ISL, osl, max_context),
        )

    load = ParallelBenchLoad(
        driver=driver,
        server=server,
        context=driver_context,
        bench_concurrency=bench_concurrency,
        max_context=max_context,
        osl=osl,
        fixed_isl=fixed_isl,
        tracker=tracker,
    )
    return load.start()


__all__ = ["ParallelBenchLoad", "start_parallel_bench"]

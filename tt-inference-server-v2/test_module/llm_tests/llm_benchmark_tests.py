# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""LLM benchmark caller — select a perf-tool driver and run the sweep.

Bridges ``test_module`` to ``llm_module`` for the standard LLM performance
benchmark. The ``--tools`` value picks one driver; the sweep is built from
the resolved runtime model spec; the run is delegated to the existing
``run_llm_performance`` orchestrator, which forwards the resulting Blocks to
``workflow_module`` for the unified report.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from llm_module import (
    AIPerfDriver,
    GenAIPerfDriver,
    GuideLLMDriver,
    LLMDriver,
    VLLMBenchDriver,
)
from llm_module.runner import RunnerResult

from ..context import MediaContext
from .llm_performance_tests import run_llm_performance

logger = logging.getLogger(__name__)


def _make_driver(tools: str, venv_python: Optional[Path]) -> LLMDriver:
    """Instantiate the driver for ``tools``, wiring its interpreter/binary."""
    py = Path(venv_python) if venv_python else Path(sys.executable)
    if tools == "vllm":
        return VLLMBenchDriver(vllm_binary=str(py.parent / "vllm"))
    if tools == "aiperf":
        return AIPerfDriver(venv_python=py)
    if tools == "guidellm":
        # Pin the venv's guidellm; os.execv doesn't put the venv bin on PATH,
        # so shutil.which would otherwise pick up a stray ~/.local install.
        return GuideLLMDriver(
            venv_python=py, guidellm_binary=str(py.parent / "guidellm")
        )
    if tools in ("genai", "genai_perf"):
        # genai-perf runs via the NVIDIA Triton SDK Docker image, not a venv.
        return GenAIPerfDriver()
    raise ValueError(
        f"Unknown LLM benchmark tool {tools!r}. Expected one of: "
        "vllm, aiperf, genai, guidellm."
    )


def run_llm_bench(
    ctx: MediaContext,
    *,
    tools: str = "vllm",
    auth_token: str = "",
    venv_python: Optional[Path] = None,
) -> RunnerResult:
    """Run the LLM performance sweep for ``tools`` and emit Blocks.

    The launcher (``run_llm_bench.py``) has already put us inside the
    tool-specific venv, so ``sys.executable`` is the right interpreter and
    ``<venv>/bin/vllm`` is the right ``vllm`` CLI when ``venv_python`` is not
    given explicitly.
    """
    driver = _make_driver(tools, venv_python)

    if tools == "guidellm":
        #  --tools guidellm runs the dataset-driven scenario set
        # (multi_turn_chat / custom_dataset / omni_modal) Selection + per-scenario knobs ride on --workflow-args.
        from llm_module import build_guidellm_scenarios

        configs = build_guidellm_scenarios(
            ctx.model_spec,
            ctx.runtime_config,
            output_root=Path(ctx.output_path),
            auth_token=auth_token,
        )
    else:
        limit_samples_mode = getattr(ctx.runtime_config, "limit_samples_mode", None)
        from llm_module.benchmark_configs import get_llm_configs

        configs = get_llm_configs(
            ctx.model_spec, ctx.device, limit_samples_mode=limit_samples_mode
        )
    if not configs:
        logger.error(
            "No LLM benchmark configs for model=%s device=%s; nothing to run.",
            ctx.model_spec.model_name,
            getattr(ctx.device, "name", ctx.device),
        )
        return RunnerResult()

    logger.info(
        "Running LLM benchmark: tool=%s over %d sweep point(s)",
        driver.name,
        len(configs),
    )
    return run_llm_performance(
        ctx,
        driver=driver,
        configs=configs,
        auth_token=auth_token,
    )


__all__ = ["run_llm_bench"]

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the v2 LLM performance benchmark.

Selects/creates the per-tool virtual environment for the requested
``--tools`` driver and re-execs ``run.py`` inside it, forwarding every CLI
argument verbatim. Mirrors ``run_prefix_cache.py`` but picks the venv from
the tool so light tools (aiperf/guidellm) don't drag in the heavy vllm venv.

Usage (all flags pass straight through to run.py):
    python tt-inference-server-v2/run_llm_bench.py \
        --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
        --tools aiperf --service-port 8000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("tt_v2_llm_bench_launcher")


def _venv_type_for_tool(tools: str):
    """Map a ``--tools`` value to the WorkflowVenvType backing it."""
    from workflows.workflow_types import WorkflowVenvType

    return {
        "aiperf": WorkflowVenvType.V2_PREFIX_CACHE,  # already ships aiperf + v2 base
        "guidellm": WorkflowVenvType.V2_LLM_GUIDELLM,
        "vllm": WorkflowVenvType.V2_LLM_VLLM,
        "inferencemax": WorkflowVenvType.V2_LLM_VLLM,
        "genai": WorkflowVenvType.V2_RUN_SCRIPT,  # genai-perf runs via Docker
        "genai_perf": WorkflowVenvType.V2_RUN_SCRIPT,
    }.get(tools, WorkflowVenvType.V2_LLM_VLLM)


def _parse_tools(argv) -> str:
    """Extract --tools from argv without consuming the rest (forwarded to run.py)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tools", default="vllm")
    known, _ = parser.parse_known_args(argv)
    return known.tools


def _ensure_venv(tools: str) -> Path:
    """Materialize the tool's venv and return its interpreter path.

    Imports only the lightweight ``workflows.*`` helpers so the launcher runs
    from any base Python; the venv brings in everything ``run.py`` and the
    perf-tool subprocess actually need. ``VenvConfig.setup`` is idempotent.
    """
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_type = _venv_type_for_tool(tools)
    venv_config = VENV_CONFIGS[venv_type]
    setup_completed = venv_config.setup(model_spec=None)
    assert setup_completed, f"Failed to setup venv: {venv_type.name}"
    return venv_config.venv_python


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_py = _V2_ROOT / "run.py"
    tools = _parse_tools(sys.argv[1:])
    venv_python = _ensure_venv(tools)

    logger.info("Launching run.py for --tools %s inside venv: %s", tools, venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    # exec (not subprocess) so the benchmark inherits this process's stdio and
    # exit code directly. Targets run.py, never this launcher, so there is no
    # re-exec loop even when we're already inside the venv.
    os.execv(str(venv_python), [str(venv_python), str(run_py), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())

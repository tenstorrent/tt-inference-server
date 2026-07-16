#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the LLM performance benchmark.

Selects/creates the per-tool virtual environment for the requested
``--tools`` driver and re-execs ``run_workflows.py`` inside it, forwarding every CLI
argument verbatim. Picks the venv from the tool so light tools (aiperf/guidellm) don't drag in the heavy vllm venv.
"""

from __future__ import annotations

import argparse
import logging
import sys

from _launcher_common import setup_venv_and_exec

logger = logging.getLogger("tt_v2_llm_bench_launcher")


def _venv_type_for_tool(tools: str):
    """Map a ``--tools`` value to the WorkflowVenvType backing it."""
    from workflows.workflow_types import WorkflowVenvType

    return {
        "aiperf": WorkflowVenvType.V2_LLM_AIPERF,
        "guidellm": WorkflowVenvType.V2_LLM_GUIDELLM,
        "vllm": WorkflowVenvType.V2_LLM_VLLM,
        "genai": WorkflowVenvType.V2_RUN_SCRIPT,  # genai-perf runs via Docker
        "genai_perf": WorkflowVenvType.V2_RUN_SCRIPT,
    }.get(tools, WorkflowVenvType.V2_LLM_VLLM)


def _parse_tools(argv) -> str:
    """Extract --tools from argv without consuming the rest (forwarded to run_workflows.py)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tools", default="vllm")
    known, _ = parser.parse_known_args(argv)
    return known.tools


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    tools = _parse_tools(sys.argv[1:])
    return setup_venv_and_exec(_venv_type_for_tool(tools), logger, f"--tools {tools}")


if __name__ == "__main__":
    sys.exit(main())

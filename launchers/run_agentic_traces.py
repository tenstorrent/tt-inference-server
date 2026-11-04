#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the agentic trace-replay benchmark.

Selects/creates the dedicated ``AGENTIC_TRACES`` virtual environment and
re-execs ``run_workflows.py`` inside it, forwarding every CLI argument verbatim.

Usage (all flags are passed straight through to run_workflows.py):
    python launchers/run_agentic_traces.py \
        --model Kimi-K2.7-Code --workflow agentic_traces --device super_cluster \
        --agentic-traces-mode ci --service-port 8000 \
        --runtime-model-spec-json /tmp/kimi_agentic_traces.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from _launcher_common import setup_venv_and_exec

logger = logging.getLogger("tt_agentic_traces_launcher")


def _check_workflow(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workflow", required=True)
    args, _ = parser.parse_known_args(argv)
    if args.workflow != "agentic_traces":
        parser.error(
            "run_agentic_traces.py requires --workflow agentic_traces "
            f"(got --workflow {args.workflow})."
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from workflows.workflow_types import WorkflowVenvType

    _check_workflow(sys.argv[1:])
    return setup_venv_and_exec(
        WorkflowVenvType.AGENTIC_TRACES, logger, "agentic traces"
    )


if __name__ == "__main__":
    sys.exit(main())

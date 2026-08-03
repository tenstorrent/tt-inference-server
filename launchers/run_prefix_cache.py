#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the prefix-caching benchmark.

Selects/creates the dedicated ``PREFIX_CACHE`` virtual environment and
re-execs ``run_workflows.py`` inside it, then forwards every CLI argument verbatim.

Usage (all flags are passed straight through to run_workflows.py):
    python launchers/run_prefix_cache.py \
        --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
        --prefix-cache --prefix-cache-preset ci --service-port 8000 \
        --jwt-secret "$JWT_SECRET"
"""

from __future__ import annotations

import logging
import sys

from _launcher_common import setup_venv_and_exec

logger = logging.getLogger("tt_prefix_cache_launcher")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from workflows.workflow_types import WorkflowVenvType

    return setup_venv_and_exec(WorkflowVenvType.PREFIX_CACHE, logger, "prefix-cache")


if __name__ == "__main__":
    sys.exit(main())

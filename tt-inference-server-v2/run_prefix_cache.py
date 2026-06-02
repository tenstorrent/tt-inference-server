#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the v2 prefix-caching benchmark.

Selects/creates the dedicated ``V2_PREFIX_CACHE`` virtual environment and
re-execs ``run.py`` inside it, then forwards every CLI argument verbatim.

Usage (all flags are passed straight through to run.py):
    python tt-inference-server-v2/run_prefix_cache.py \
        --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
        --prefix-cache --prefix-cache-preset ci --service-port 8000 \
        --jwt-secret "$JWT_SECRET"
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("tt_v2_prefix_cache_launcher")


def _ensure_prefix_cache_venv() -> Path:
    """Materialize ``V2_PREFIX_CACHE`` and return its interpreter path.

    Imports only the lightweight ``workflows.*`` helpers so the launcher runs
    from any base Python (no aiohttp / torch / aiperf required); the venv
    brings in everything ``run.py`` and the AIPerf subprocess actually need.
    ``VenvConfig.setup`` is idempotent, so repeated launches are cheap once the
    venv exists.
    """
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_config = VENV_CONFIGS[WorkflowVenvType.V2_PREFIX_CACHE]
    # V2_PREFIX_CACHE declares only a requirements_file (no setup_function),
    # so model_spec is unused at runtime — passing None keeps the launcher
    # independent of the heavy v2 import chain.
    setup_completed = venv_config.setup(model_spec=None)
    assert setup_completed, "Failed to setup venv: V2_PREFIX_CACHE"
    return venv_config.venv_python


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_py = _V2_ROOT / "run.py"
    venv_python = _ensure_prefix_cache_venv()

    logger.info("Launching run.py inside V2_PREFIX_CACHE venv: %s", venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    # exec (not subprocess) so the benchmark inherits this process's stdio and
    # exit code directly. Targets run.py, never this launcher, so there is no
    # re-exec loop even when we're already inside the venv.
    os.execv(str(venv_python), [str(venv_python), str(run_py), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())

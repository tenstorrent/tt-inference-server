#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Thin launcher for the speculative-decoding benchmark.

Selects/creates the dedicated ``SPEC_DECODE`` virtual environment and
re-execs ``run_workflows.py`` inside it, then forwards every CLI argument
verbatim.

Usage (all flags are passed straight through to run_workflows.py):
    python launchers/run_spec_decode.py \
        --model Llama-3.1-8B-Instruct --workflow benchmarks --device gpu \
        --spec-decode --service-port 8000 --jwt-secret "$JWT_SECRET"
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# launchers/<this file> -> parent is launchers/, parent.parent is the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("tt_spec_decode_launcher")


def _ensure_spec_decode_venv() -> Path:
    """Materialize ``SPEC_DECODE`` and return its interpreter path.

    Imports only the lightweight ``workflows.*`` helpers so the launcher runs
    from any base Python (no aiohttp / torch / aiperf required); the venv
    brings in everything ``run_workflows.py`` and the AIPerf subprocess
    actually need. ``VenvConfig.setup`` is idempotent, so repeated launches
    are cheap once the venv exists.
    """
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_config = VENV_CONFIGS[WorkflowVenvType.SPEC_DECODE]
    # SPEC_DECODE declares only a requirements_file (no setup_function),
    # so model_spec is unused at runtime — passing None keeps the launcher
    # independent of the heavy workflow import chain.
    setup_completed = venv_config.setup(model_spec=None)
    assert setup_completed, "Failed to setup venv: SPEC_DECODE"
    return venv_config.venv_python


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_py = _REPO_ROOT / "run_workflows.py"
    venv_python = _ensure_spec_decode_venv()

    logger.info("Launching run_workflows.py inside SPEC_DECODE venv: %s", venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    # exec (not subprocess) so the benchmark inherits this process's stdio and
    # exit code directly. Targets run_workflows.py, never this launcher, so
    # there is no re-exec loop even when we're already inside the venv.
    os.execv(str(venv_python), [str(venv_python), str(run_py), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())

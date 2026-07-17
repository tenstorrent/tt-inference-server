# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared body for the thin benchmark launchers.

``run_prefix_cache.py`` and ``run_llm_bench.py`` both materialize a
per-tool venv and ``os.execv`` ``run_workflows.py`` inside it, forwarding
argv verbatim. The only difference is which ``WorkflowVenvType`` they pick,
so the sys.path bootstrap, venv setup, and re-exec live here once.
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


def setup_venv_and_exec(venv_type, logger: logging.Logger, label: str) -> int:
    """Materialize ``venv_type`` and re-exec ``run_workflows.py`` inside it.

    Imports only the lightweight ``workflows.*`` helpers so the launcher
    runs from any base Python; the venv brings in everything
    ``run_workflows.py`` and the perf-tool subprocess actually need.
    ``VenvConfig.setup`` is idempotent, so repeated launches are cheap once
    the venv exists.
    """
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_config = VENV_CONFIGS[venv_type]
    setup_completed = venv_config.setup(model_spec=None)
    assert setup_completed, f"Failed to setup venv: {venv_type.name}"
    venv_python = venv_config.venv_python

    run_py = _REPO_ROOT / "run_workflows.py"
    logger.info("Launching run_workflows.py for %s inside venv: %s", label, venv_python)
    sys.stdout.flush()
    sys.stderr.flush()
    # exec (not subprocess) so the benchmark inherits this process's stdio and
    # exit code directly. Targets run_workflows.py, never the launcher, so
    # there is no re-exec loop even when we're already inside the venv.
    os.execv(str(venv_python), [str(venv_python), str(run_py), *sys.argv[1:]])

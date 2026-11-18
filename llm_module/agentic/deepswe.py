# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""DeepSWE tasks, checked out from the benchmark repository.

DeepSWE (https://deepswe.datacurve.ai) is not in the Harbor registry: its 113
tasks ship in Harbor's task format in ``datacurve-ai/deep-swe``. The pinned
commit is v1.1, which grades in a separate verifier container and extracts the
agent's commits with a ``[[verifier.collect]]`` hook; the repository has no
v1.1 tag (only ``v1.0.0``), so the pin is a commit.

Every task sets ``[agent] network_mode = "no-network"``, so the eval config
must set ``allow_agent_endpoint`` for the agent to reach the inference server.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEEPSWE_DATASET = "datacurve/deep-swe"
_DEEPSWE_REPO = "https://github.com/datacurve-ai/deep-swe.git"
# master at "Update task timeout settings to 10800s" (2026-08-26): DeepSWE v1.1.
DEEPSWE_COMMIT = "0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea"
_CACHE_DIR_NAME = "harbor-datasets"


def generate_deepswe(python: Path) -> Path:
    """Task directory of DeepSWE at :data:`DEEPSWE_COMMIT`.

    The checkout is cached in the venv that owns *python*, keyed by the
    commit, so a run reuses it and a pin bump fetches the new revision.
    """
    # Not resolved: a venv interpreter is a symlink to the base Python, and
    # the cache belongs to the venv.
    venv_dir = Path(python).parent.parent
    dest = venv_dir / _CACHE_DIR_NAME / f"deep-swe-{DEEPSWE_COMMIT[:12]}"
    tasks = dest / "tasks"
    if tasks.is_dir():
        logger.info("Using cached %s tasks at %s", DEEPSWE_DATASET, tasks)
        return tasks

    logger.info("Checking out %s at %s into %s", DEEPSWE_DATASET, DEEPSWE_COMMIT, dest)
    tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            _DEEPSWE_REPO,
            str(tmp),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp), "checkout", "--quiet", DEEPSWE_COMMIT], check=True
    )
    try:
        tmp.rename(dest)
    except OSError:
        # A concurrent run published the same checkout first.
        if not dest.is_dir():
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    return tasks

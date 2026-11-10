# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""SWE-bench Pro tasks without the upstream fix in reach of the agent.

The registry tasks (``swebenchpro`` 1.0, ScaleAI's public set) leak the answer
two ways:

- History: the images (``jefzda/sweap-images``) keep the repository's full git
  history, including the commit that fixed the issue and everything after it,
  so ``git log --all`` / ``git show <fix>`` reveals the reference solution.
- Network: agents have open egress, so they can fetch the upstream fix from
  GitHub or a package index.

Both were seen on QB2 (tt-agentic-bringup-qb2 #89 and #90). The registry
verifier needs the history too: it restores the gold test files with
``git checkout <fix> -- <files>``.

Each generated task differs from the registry copy in three places:

- ``environment/Dockerfile``: after the base checkout, keep only the base
  commit on the current branch (no remotes, tags, other refs, reflog or
  unreachable objects). The build fails if the fix commit is still reachable.
- ``tests/``: the gold test files are restored from the dataset's
  ``test_patch``, which touches exactly the files the gold checkout names in
  every task (checked for all 731, and byte-identical to the checkout on the
  images we tried). ``tests/`` is uploaded only for verification, so the agent
  never sees it.
- ``task.toml``: ``[agent] network_mode = "no-network"``, so the eval config
  must set ``allow_agent_endpoint`` for the agent to reach the inference
  server. Agent setup and the verifier keep the task's network (some NodeBB
  verifiers run ``npm install``).

Scale's SWE-bench Pro V2 closes the same leaks, but it also rewrites the
instructions and verifiers and drops tasks, so scores are not comparable with
the public set that model cards report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SWEBENCH_PRO_DATASET = "swebenchpro"
_HARBOR_DATASETS_REPO = "https://github.com/laude-institute/harbor-datasets.git"
# The commit every task of the registry's swebenchpro@1.0 points at.
SWEBENCH_PRO_COMMIT = "c8e8f3fac7097accaacf261d74c3d6f441de45b1"
_TASKS_SUBDIR = "datasets/swebenchpro"
_CACHE_DIR_NAME = "harbor-datasets"

_DOCKERFILE_ANCHOR = "# Install uv for running Python scripts"
# {fix} is the commit the gold checkout reads the tests from.
_STRIP_HISTORY = """\
# No future git history: keep only the base commit, so the fix commit and
# everything after it are gone from the image (see swebench_pro.py).
RUN set -e; d=/app; [ -d /app/.git ] || d=/testbed; cd "$d"; \\
    b=$(git rev-parse HEAD); br=$(git symbolic-ref -q --short HEAD || true); \\
    git checkout -q --detach "$b"; \\
    for r in $(git remote); do git remote remove "$r"; done; \\
    git for-each-ref --format='%(refname)' | while read -r r; do git update-ref --no-deref -d "$r"; done; \\
    if [ -n "$br" ]; then git checkout -q -B "$br" "$b"; fi; \\
    rm -rf .git/logs .git/ORIG_HEAD .git/FETCH_HEAD .git/MERGE_HEAD .git/refs/original .git/shallow; \\
    git reflog expire --expire=now --all; git repack -a -d -q; git prune --expire=now; \\
    test "$(git rev-list --all | head -1)" = "$b"; test "$(git for-each-ref | wc -l)" -le 1; \\
    ! git cat-file -e {fix}^{{commit}} 2>/dev/null

"""

_GOLD_CHECKOUT_LINE = (
    '    eval "$GOLD_CHECKOUT" || { echo "ERROR: Gold tests checkout failed"; exit 1; }'
)
_GOLD_RESTORE_LINE = (
    "    bash /tests/restore_gold_tests.sh || "
    '{ echo "ERROR: Gold tests restore failed"; exit 1; }'
)
_GOLD_PATCH = "gold_tests.patch"
_GOLD_RESTORE = "restore_gold_tests.sh"
# Reset each gold file to the base commit first, as the checkout overwrote
# whatever the agent left there, then apply the gold test changes.
_RESTORE_TEMPLATE = """\
#!/bin/bash
# Restores the gold test files without the fix commit, which the image no
# longer has (see swebench_pro.py). Replaces: {checkout}
set -euo pipefail
for p in {paths}; do
    if git cat-file -e {base}:"$p" 2>/dev/null; then
        git checkout {base} -- "$p"
    else
        rm -f -- "$p"
    fi
done
git apply /tests/{patch}
"""


def generate_swebench_pro(python: Path) -> Path:
    """Leak-free SWE-bench Pro tasks from the registry's pinned source.

    Cached in the venv that owns *python*, keyed by the source commit and this
    module, so a change to either regenerates.
    """
    venv_dir = Path(python).parent.parent
    key = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    key_dir = (
        venv_dir / _CACHE_DIR_NAME / f"swebenchpro-{SWEBENCH_PRO_COMMIT[:12]}-{key}"
    )
    # Harbor reports a local dataset under its directory name.
    dest = key_dir / SWEBENCH_PRO_DATASET
    if dest.is_dir():
        logger.info("Using cached %s tasks at %s", SWEBENCH_PRO_DATASET, dest)
        return dest

    src = _checkout_source(
        venv_dir / _CACHE_DIR_NAME / f"harbor-datasets-{SWEBENCH_PRO_COMMIT[:12]}"
    )
    tmp = key_dir.with_name(f"{key_dir.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    logger.info("Generating leak-free %s tasks into %s", SWEBENCH_PRO_DATASET, dest)
    shutil.copytree(src / _TASKS_SUBDIR, tmp / SWEBENCH_PRO_DATASET)
    task_dirs = sorted(p for p in (tmp / SWEBENCH_PRO_DATASET).iterdir() if p.is_dir())
    for task_dir in task_dirs:
        sanitize_task(task_dir)
    try:
        tmp.rename(key_dir)
    except OSError:
        # A concurrent run published the same tasks first.
        if not dest.is_dir():
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    logger.info("Generated %d %s tasks", len(task_dirs), SWEBENCH_PRO_DATASET)
    return dest


def sanitize_task(task_dir: Path) -> None:
    """Rewrite one registry task in place (see the module docstring)."""
    config = json.loads((task_dir / "tests" / "config.json").read_text())
    checkout = config["before_repo_set_cmd"].strip().split("\n")[-1].strip()
    parts = shlex.split(checkout)
    if parts[:2] != ["git", "checkout"] or "--" not in parts:
        raise ValueError(f"{task_dir.name}: unexpected gold checkout {checkout!r}")
    fix, paths = parts[2], parts[parts.index("--") + 1 :]
    base = config["base_commit"]

    dockerfile = task_dir / "environment" / "Dockerfile"
    _replace_once(
        dockerfile,
        _DOCKERFILE_ANCHOR,
        _STRIP_HISTORY.format(fix=fix) + _DOCKERFILE_ANCHOR,
    )

    tests = task_dir / "tests"
    (tests / _GOLD_PATCH).write_text(config["test_patch"])
    (tests / _GOLD_RESTORE).write_text(
        _RESTORE_TEMPLATE.format(
            checkout=checkout,
            paths=" ".join(shlex.quote(p) for p in paths),
            base=base,
            patch=_GOLD_PATCH,
        )
    )
    _replace_once(tests / "test.sh", _GOLD_CHECKOUT_LINE, _GOLD_RESTORE_LINE)

    _replace_once(
        task_dir / "task.toml", "[agent]\n", '[agent]\nnetwork_mode = "no-network"\n'
    )


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected exactly one {old!r}")
    path.write_text(text.replace(old, new))


def _checkout_source(dest: Path) -> Path:
    if (dest / _TASKS_SUBDIR).is_dir():
        return dest
    tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Checking out %s at %s into %s", _TASKS_SUBDIR, SWEBENCH_PRO_COMMIT, dest
    )
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            _HARBOR_DATASETS_REPO,
            str(tmp),
        ],
        check=True,
    )
    # Only this dataset's blobs are fetched.
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp),
            "checkout",
            "--quiet",
            SWEBENCH_PRO_COMMIT,
            "--",
            _TASKS_SUBDIR,
        ],
        check=True,
    )
    try:
        tmp.rename(dest)
    except OSError:
        if not dest.is_dir():
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    return dest

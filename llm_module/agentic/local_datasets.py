# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Harbor datasets generated on the host instead of pulled from the registry.

The registry copy of ``sierra-research/tau3-bench`` predates the tau3 adapter
pinning tau2-bench: its task Dockerfiles clone tau2-bench ``main`` at build
time and never install ``websockets``, which the tau2 evaluator now imports,
so every verifier reports ``No module named 'websockets'`` and scores 0.
Generating the tasks from the adapter in the pinned Harbor checkout picks up
its fixed Dockerfiles. Generated tasks keep the registry task names in
``task.toml``, but Harbor filters a local dataset by directory name, so
configured task-name patterns are translated with :func:`local_task_pattern`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TAU3_BENCH_DATASET = "sierra-research/tau3-bench"
_TAU2_BENCH_REPO = "https://github.com/sierra-research/tau2-bench.git"
_TAU3_ADAPTER_SRC = Path("adapters") / "tau3-bench" / "src"
_TAU3_TEMPLATE_DOCKERFILE = (
    _TAU3_ADAPTER_SRC / "tau3_bench" / "task-template" / "environment" / "Dockerfile"
)
_TAU2_REF_RE = re.compile(r'^ARG TAU2_BENCH_REF="([0-9a-f]{40})"', re.MULTILINE)
_CACHE_DIR_NAME = "harbor-datasets"


def local_task_pattern(dataset: str, pattern: str) -> str:
    """Map a registry task-name pattern onto the generated directory names.

    ``sierra-research/tau3-bench__tau3-banking_knowledge-*`` becomes
    ``tau3-banking_knowledge-*``; patterns without the prefix pass through.
    """
    prefix = f"{dataset}__"
    return pattern[len(prefix) :] if pattern.startswith(prefix) else pattern


def prepare_local_dataset(dataset: str, python: Path) -> Optional[Path]:
    """Task directory for *dataset* when it is generated locally, else ``None``.

    *python* is the EVALS_AGENTIC interpreter, whose editable Harbor install
    carries the adapters. Output is cached next to that checkout, keyed by the
    adapter sources and the tau2-bench commit, so a changed adapter or pin
    regenerates and an unchanged one is reused.
    """
    if dataset != TAU3_BENCH_DATASET:
        return None
    return _generate_tau3_bench(Path(python))


def _generate_tau3_bench(python: Path) -> Path:
    harbor_root = _harbor_root(python)
    adapter_src = harbor_root / _TAU3_ADAPTER_SRC
    ref = tau2_bench_ref(harbor_root / _TAU3_TEMPLATE_DOCKERFILE)
    cache_dir = harbor_root.parent / _CACHE_DIR_NAME
    # Harbor reports a local dataset under its directory name, so the tasks sit
    # in a ``tau3-bench`` directory beneath the cache key.
    key_dir = cache_dir / f"tau3-bench-{tree_digest(adapter_src)[:12]}-{ref[:12]}"
    dest = key_dir / "tau3-bench"
    if dest.is_dir():
        logger.info("Using cached %s tasks at %s", TAU3_BENCH_DATASET, dest)
        return dest

    tau2_root = _checkout_tau2_bench(cache_dir / f"tau2-bench-{ref[:12]}", ref)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(adapter_src), env.get("PYTHONPATH")) if p
    )
    env["TAU2_BENCH_ROOT"] = str(tau2_root)
    logger.info(
        "Generating %s tasks into %s (tau2-bench %s, adapter %s)",
        TAU3_BENCH_DATASET,
        dest,
        ref,
        adapter_src,
    )
    tmp = _scratch_dir(key_dir)
    subprocess.run(
        [
            str(python),
            "-m",
            "tau3_bench.main",
            "--output-dir",
            str(tmp / dest.name),
            "--overwrite",
        ],
        check=True,
        env=env,
    )
    _move_into_place(tmp, key_dir)
    return dest


def _harbor_root(python: Path) -> Path:
    """Root of the editable Harbor checkout installed in *python*'s venv."""
    out = subprocess.run(
        [
            str(python),
            "-c",
            "import pathlib, harbor; "
            "print(pathlib.Path(harbor.__file__).resolve().parents[2])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def tau2_bench_ref(dockerfile: Path) -> str:
    """tau2-bench commit the adapter's task Dockerfile builds from.

    Generating from the same commit keeps the task data in step with the
    runtime the images install.
    """
    match = _TAU2_REF_RE.search(dockerfile.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(
            f"{dockerfile} does not pin TAU2_BENCH_REF. The pinned Harbor "
            "checkout predates the tau3 adapter fix, so generated tasks would "
            "fail their verifier the same way the registry copy does."
        )
    return match.group(1)


def tree_digest(root: Path) -> str:
    """sha256 over the relative paths and contents of every file under *root*."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkout_tau2_bench(dest: Path, ref: str) -> Path:
    if (dest / ".git").is_dir():
        return dest
    tmp = _scratch_dir(dest)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            _TAU2_BENCH_REPO,
            str(tmp),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp), "checkout", "--quiet", ref], check=True)
    _move_into_place(tmp, dest)
    return dest


def _scratch_dir(dest: Path) -> Path:
    tmp = dest.with_name(f"{dest.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    return tmp


def _move_into_place(tmp: Path, dest: Path) -> None:
    try:
        tmp.rename(dest)
    except OSError:
        # A concurrent run published the same directory first.
        if not dest.is_dir():
            raise
        shutil.rmtree(tmp, ignore_errors=True)

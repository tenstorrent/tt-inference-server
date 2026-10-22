# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""List the task names a Harbor dataset selection resolves to.

Run with the EVALS_AGENTIC venv's interpreter (it imports ``harbor``), as a
script, not a module: the caller is the agentic driver planning an
``--eval-shard``. It resolves the dataset exactly as ``harbor run`` does --
Harbor's own ``DatasetConfig`` with the run's ``task_names`` /
``exclude_task_names`` / ``n_tasks`` filters, in Harbor's order -- so the
names it prints are the task set an unsharded run evaluates.

    <venv>/bin/python harbor_task_list.py --spec spec.json --out names.json

``spec.json``: ``{"dataset", "dataset_ref", "task_names",
"exclude_task_names", "n_tasks"}``. ``names.json``: ``{"task_names": [...],
"resolved_ref": <the package content digest Harbor resolved, or null>}``.

Standard library plus ``harbor`` only; no repo imports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Run as a script, this file's directory is sys.path[0], and it holds the
# repo's own ``harbor.py`` runner, which would shadow the ``harbor`` package.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]


def resolve(spec: dict) -> dict:
    from harbor.models.job.config import DatasetConfig

    kwargs = {"name": spec["dataset"]}
    if spec.get("dataset_ref"):
        kwargs["ref"] = spec["dataset_ref"]
    if spec.get("task_names"):
        kwargs["task_names"] = list(spec["task_names"])
    if spec.get("exclude_task_names"):
        kwargs["exclude_task_names"] = list(spec["exclude_task_names"])
    if spec.get("n_tasks") is not None:
        kwargs["n_tasks"] = int(spec["n_tasks"])
    config = DatasetConfig(**kwargs)
    task_configs = asyncio.run(config.get_task_configs())
    return {
        "task_names": [tc.get_task_id().get_name() for tc in task_configs],
        # A package dataset records the content digest it resolved to.
        "resolved_ref": config.ref,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    listing = resolve(json.loads(args.spec.read_text()))
    args.out.write_text(json.dumps(listing, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

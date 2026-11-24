# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""List the Harbor task names a full agentic eval run would select.

    python3 -m report_module.list_agentic_tasks --model <hf_repo> \\
        --task <agentic eval task> [--out tasks.json]

Resolves the model's EvalConfig task and asks Harbor for its dataset's tasks
through the same job-config ``datasets`` entry a full run hands Harbor: the
registry dataset (and ``dataset_ref``) or the locally generated copy
(``local_datasets``), the configured ``task_names`` / ``exclude_task_names``
filters and ``n_tasks``. Limit modes (``task_names_map``, ``limit_samples_map``)
are not applied: this is the full run's set. Each name is the task id Harbor
filters on, so a slice of the list can be passed as is to ``run.py
--agentic-task-names`` and the whole list to ``report_module.merge_agentic_runs
--expected-tasks``.

Writes a JSON list of names to ``--out`` (or stdout). Needs the Harbor CLI's
Python: run from any Python, the tool sets up the ``EVALS_AGENTIC`` venv
(``.workflow_venvs/.venv_evals_agentic``, the one ``run.py --workflow agentic``
uses) and re-runs itself there. Registry datasets need network access to
Harbor's registry; locally generated ones (tau3-bench, DeepSWE) clone their
sources on first use and are cached in that venv.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REEXEC_ENV = "TT_LIST_AGENTIC_TASKS_IN_VENV"
_REPO_ROOT = Path(__file__).resolve().parent.parent


class ListError(Exception):
    """The task cannot be listed."""


def eval_task(model: str, task_name: str):
    """The EvalConfig agentic task to list."""
    from reference_config.evals.eval_config import _eval_config_map

    config = _eval_config_map.get(model)
    tasks = {t.task_name: t for t in (config.tasks if config else [])}
    task = tasks.get(task_name)
    if task is None or getattr(task, "agentic_eval_config", None) is None:
        agentic = sorted(
            name for name, t in tasks.items() if getattr(t, "agentic_eval_config", None)
        )
        raise ListError(
            f"{model} has no agentic EvalConfig task {task_name!r}; "
            f"its agentic tasks: {agentic}"
        )
    return task


def dataset_config_for(task, python: Path) -> Dict[str, Any]:
    """The Harbor ``datasets[]`` entry a full run of ``task`` would use."""
    from llm_module.agentic.harbor import harbor_dataset_config
    from llm_module.agentic.local_datasets import prepare_local_dataset

    cfg = task.agentic_eval_config
    return harbor_dataset_config(
        cfg.dataset,
        dataset_ref=getattr(cfg, "dataset_ref", None),
        dataset_path=prepare_local_dataset(cfg.dataset, python),
        task_names=list(cfg.task_names or []),
        exclude_task_names=list(cfg.exclude_task_names or []),
        n_tasks=cfg.n_tasks,
    )


def harbor_task_names(dataset_config: Dict[str, Any]) -> List[str]:
    """Harbor's own selection for ``dataset_config``, as filterable task ids."""
    from harbor.models.job.config import DatasetConfig

    dataset = DatasetConfig.model_validate(dataset_config)
    task_configs = asyncio.run(dataset.get_task_configs())
    return [tc.get_task_id().get_name() for tc in task_configs]


def list_task_names(model: str, task_name: str, python: Path) -> List[str]:
    task = eval_task(model, task_name)
    dataset_config = dataset_config_for(task, python)
    names = harbor_task_names(dataset_config)
    if not names:
        raise ListError(f"Harbor selected no tasks from {dataset_config}")
    print(
        f"{task_name}: {len(names)} task(s) from {dataset_config}",
        file=sys.stderr,
    )
    return names


def _reexec_in_agentic_venv() -> None:
    """Set up the EVALS_AGENTIC venv and re-run this module inside it."""
    if os.environ.get(_REEXEC_ENV):
        raise ListError(
            f"harbor is not importable in {sys.executable} after the venv setup"
        )
    from workflows.workflow_types import WorkflowVenvType
    from workflows.workflow_venvs import VENV_CONFIGS

    venv_config = VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC]
    try:
        ready = venv_config.setup(model_spec=None)
    except RuntimeError as exc:
        raise ListError(f"EVALS_AGENTIC venv setup failed: {exc}") from exc
    if not ready:
        raise ListError("EVALS_AGENTIC venv setup failed")
    python = str(venv_config.venv_python)
    env = dict(os.environ, **{_REEXEC_ENV: "1"})
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os.chdir(_REPO_ROOT)
    os.execve(
        python, [python, "-m", "report_module.list_agentic_tasks", *sys.argv[1:]], env
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="HF repo id, e.g. zai-org/GLM-5.2"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="agentic eval task, e.g. terminal_bench_2_1, tau3_bench_banking, deepswe",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="JSON file to write (default stdout)"
    )
    args = parser.parse_args(argv)
    try:
        # Fail on a bad model/task before any venv setup.
        eval_task(args.model, args.task)
        if importlib.util.find_spec("harbor") is None:
            _reexec_in_agentic_venv()
        names = list_task_names(args.model, args.task, Path(sys.executable))
    except ListError as exc:
        print(f"list refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(names, indent=2) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Every ``--eval-tasks`` in models-ci-config.json names a task the model runs.

A misspelled task name in a CI entry's ``additional-args`` passes schema
validation and every spec test, and only fails on a device run, where
``get_llm_eval_tasks`` raises ValueError. This walks the CI config, resolves
each entry's model spec the way run.py does in CI (``--dev-mode``, so
``MODEL_SPECS_ENV=dev``) and runs the same task selection on CPU.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "models-ci-config.json"


def _flag_value(argv, flag):
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _eval_task_entries():
    """(model, engine, impl, device, schedule, eval_tasks) per CI leaf that
    passes --eval-tasks."""
    from scripts.release.model_spec_resolver import iter_implementations

    config = json.loads(CI_CONFIG.read_text())
    entries = []
    for model, model_entry in config["models"].items():
        for implementation in iter_implementations(model_entry):
            for schedule, sched in (implementation.get("ci") or {}).items():
                for device, dargs in (sched.get("device-args") or {}).items():
                    argv = shlex.split(dargs.get("additional-args") or "")
                    tasks = _flag_value(argv, "--eval-tasks")
                    if tasks is None:
                        continue
                    impl = _flag_value(argv, "--impl") or implementation.get("impl")
                    entries.append(
                        (
                            model,
                            implementation.get("inference_engine"),
                            impl,
                            device,
                            schedule,
                            tasks,
                        )
                    )
    return entries


def test_ci_eval_task_names_resolve_for_their_model():
    entries = _eval_task_entries()
    if not entries:
        return

    script = textwrap.dedent(
        """
        import json, sys
        from types import SimpleNamespace
        from llm_module.eval_configs import get_llm_eval_tasks
        from scripts.release.model_spec_resolver import _release_device
        from workflows.model_spec import get_runtime_model_spec
        from workflows.workflow_types import InferenceEngine

        failures = []
        for model, engine, impl, device, schedule, tasks in json.loads(sys.argv[1]):
            where = f"{model} impl={impl} {schedule} {device} --eval-tasks {tasks}"
            dev = _release_device(device)
            try:
                spec, _, _ = get_runtime_model_spec(
                    model=model,
                    device=dev.to_string(),
                    engine=InferenceEngine.from_string(engine).value,
                    impl=impl,
                )
                rc = SimpleNamespace(
                    eval_tasks=tasks, eval_samples=None, limit_samples_mode=None
                )
                selected = [t.task_name for t in get_llm_eval_tasks(spec, rc, device=dev)]
                wanted = [n.strip() for n in tasks.split(",") if n.strip()]
                if sorted(selected) != sorted(wanted):
                    failures.append(f"{where}: selected {selected}")
            except Exception as exc:
                failures.append(f"{where}: {type(exc).__name__}: {exc}")
        print(json.dumps(failures))
        """
    )
    env = {**os.environ, "MODEL_SPECS_ENV": "dev", "PYTHONDONTWRITEBYTECODE": "1"}
    output = subprocess.check_output(
        [sys.executable, "-c", script, json.dumps(entries)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    )
    failures = json.loads(output.strip().splitlines()[-1])
    assert failures == [], "\n".join(failures)

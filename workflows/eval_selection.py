# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Select complete evaluation tasks independently of sample-limit controls."""


def select_full_eval_tasks(tasks, runtime_config=None):
    """Return configured tasks named by --eval-tasks, without changing budgets.

    An explicit selection is strict: typos, empty entries, duplicates and
    combinations with sample-limit flags are errors, never a fallback to all
    tasks. Preserve configured order and the original task objects.
    """
    raw = getattr(runtime_config, "eval_tasks", None)
    if raw is None:
        return tasks
    if not isinstance(raw, str):
        raise ValueError("--eval-tasks must be a comma-separated string")
    if any(
        getattr(runtime_config, field, None)
        for field in ("eval_samples", "limit_samples_mode", "ci_mode")
    ):
        raise ValueError(
            "--eval-tasks selects full tasks; sample-limit flags are incompatible"
        )
    requested = [name.strip() for name in raw.split(",")]
    if not all(requested) or len(set(requested)) != len(requested):
        raise ValueError("--eval-tasks requires nonempty, unique task names")
    available = {task.task_name for task in tasks}
    unknown = set(requested) - available
    if unknown:
        raise ValueError(
            f"--eval-tasks unknown or unavailable in this workflow: {sorted(unknown)}; "
            f"available: {sorted(available)}"
        )
    return [task for task in tasks if task.task_name in requested]

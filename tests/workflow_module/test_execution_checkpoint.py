# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``WorkflowExecution.run`` checkpoints the report every time a Block lands.

A cancelled job never reaches the end-of-run report phase, so whatever the
runners accepted must already be on disk -- and carry the same workflow
metadata (``workflow``, ``run_command``) the final report would, because the
exabox merged report and ``exabox_unreported_tests`` key on those fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from report_module.schema import Block
from workflow_module import BlockAccumulator
from workflow_module.execution import (
    OrchestratorMetadata,
    TaskOutcome,
    WorkflowExecution,
)

ENVELOPE = {
    "model_name": "GLM-5.3",
    "model_repo": "zai-org/GLM-5.3",
    "device": "SUPER_CLUSTER",
    "generated_at": "2026-09-23 10:00:00",
}
RUN_COMMAND = "run.py --model GLM-5.3 --workflow agentic --agentic-benchmark tau3"


def _block(task_name: str) -> Block:
    return Block(
        kind="evals",
        title="Agentic Eval",
        id="zai-org__GLM-5.3_SUPER_CLUSTER",
        data={"task_name": task_name, "accuracy": 0.5},
    )


def _ctx(tmp_path: Path) -> MagicMock:
    ctx = MagicMock()
    ctx.output_path = str(tmp_path / "reports_output" / "agentic" / "out")
    ctx.model_spec.model_name = "GLM-5.3"
    ctx.device.name = "SUPER_CLUSTER"
    ctx.service_port = 8000
    return ctx


def _report_dir(tmp_path: Path) -> Path:
    return tmp_path / "reports_output" / "agentic"


def _only_data_file(tmp_path: Path) -> dict:
    files = sorted((_report_dir(tmp_path) / "data").glob("report_data_*.json"))
    assert len(files) == 1, files
    return json.loads(files[0].read_text())


class _Workflow(WorkflowExecution):
    """Accepts one Block per task name, snapshotting the report in between."""

    name = "agentic"

    def __init__(self, *args, tasks: List[str], die_after: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.die_after = die_after
        self.snapshots: List[dict] = []

    def run_tasks(self) -> List[TaskOutcome]:
        for i, task in enumerate(self.tasks, start=1):
            self.accumulator.accept([_block(task)], envelope=ENVELOPE)
            self.snapshots.append(_only_data_file(self.tmp_path))
            if i == self.die_after:
                raise RuntimeError("job cancelled")
        return [TaskOutcome("evaluation", 0, 1.0, "evals")]


def _workflow(tmp_path: Path, acc: BlockAccumulator, **kwargs) -> _Workflow:
    wf = _Workflow(
        _ctx(tmp_path),
        accumulator=acc,
        orchestrator_metadata=OrchestratorMetadata(run_command=RUN_COMMAND),
        **kwargs,
    )
    wf.tmp_path = tmp_path
    return wf


def test_each_accept_writes_a_partial_report_with_workflow_metadata(tmp_path):
    acc = BlockAccumulator()
    wf = _workflow(tmp_path, acc, tasks=["tau3", "terminal_bench_2_1"])

    wf.run()

    first, second = wf.snapshots
    for snapshot, n in ((first, 1), (second, 2)):
        meta = snapshot["metadata"]
        assert meta["report_partial"] is True
        assert meta["report_blocks"] == n
        assert meta["workflow"] == "agentic"
        assert meta["run_command"] == RUN_COMMAND
        assert len(snapshot["sections"]) == n


def test_finished_run_replaces_the_checkpoint_with_the_final_report(tmp_path):
    acc = BlockAccumulator()
    _workflow(tmp_path, acc, tasks=["tau3", "terminal_bench_2_1"]).run()

    final = _only_data_file(tmp_path)
    assert final["metadata"]["report_partial"] is False
    assert final["metadata"]["report_blocks"] == 2


def test_a_run_killed_mid_sweep_leaves_the_finished_tasks_on_disk(tmp_path):
    acc = BlockAccumulator()
    wf = _workflow(tmp_path, acc, tasks=["tau3", "terminal_bench_2_1"], die_after=1)

    result = wf.run()

    assert result.return_code == 1
    left = _only_data_file(tmp_path)
    assert left["metadata"]["report_partial"] is True
    assert left["metadata"]["workflow"] == "agentic"
    assert [s["data"]["task_name"] for s in left["sections"]] == ["tau3"]


def test_hook_is_removed_after_the_run_and_the_previous_one_restored(tmp_path):
    acc = BlockAccumulator()
    outer_calls: List[int] = []

    def outer() -> None:
        outer_calls.append(1)

    acc.set_on_accept(outer)
    _workflow(tmp_path, acc, tasks=["tau3"]).run()

    assert outer_calls == []  # the workflow's hook replaced it during the run
    assert acc.set_on_accept(None) is outer

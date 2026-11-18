# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.checkpoint.checkpoint_report``.

The point of a checkpoint is that a run killed mid-sweep still leaves a
readable report, so these assert the three properties that guarantee it:
something is written after the first Block, later checkpoints overwrite the
same paths (no file pile-up), and every checkpoint is marked partial.
"""

from __future__ import annotations

import json
from pathlib import Path

from report_module.schema import Block
from workflow_module import BlockAccumulator
from workflow_module.checkpoint import checkpoint_report


ENVELOPE = {
    "model_name": "GLM-5.3",
    "model_repo": "zai-org/GLM-5.3",
    "device": "SUPER_CLUSTER",
    "generated_at": "2026-09-16 12-26-46",
}


def _block(isl: int) -> Block:
    return Block(
        kind="benchmarks",
        title="vLLM Benchmark",
        id="zai-org__GLM-5.3_SUPER_CLUSTER",
        data={
            "input_sequence_length": isl,
            "output_sequence_length": 128,
            "concurrency": 80,
            "num_requests": 160,
            "status": "na",
        },
    )


def _data_files(report_dir: Path) -> list[Path]:
    return sorted((report_dir / "data").glob("report_data_*.json"))


def test_empty_accumulator_writes_nothing(tmp_path: Path) -> None:
    assert checkpoint_report(tmp_path, accumulator=BlockAccumulator()) is False
    assert not (tmp_path / "data").exists()


def test_checkpoint_after_each_point_overwrites_one_file(tmp_path: Path) -> None:
    acc = BlockAccumulator()
    for isl in (128, 1024, 8192):
        acc.accept([_block(isl)], envelope=ENVELOPE)
        assert checkpoint_report(tmp_path, accumulator=acc) is True
        # report_id is synthesised once from generated_at, so every checkpoint
        # lands on the same path instead of leaving one file per sweep point.
        assert len(_data_files(tmp_path)) == 1

    payload = json.loads(_data_files(tmp_path)[0].read_text())
    assert payload["metadata"]["report_partial"] is True
    assert payload["metadata"]["report_blocks"] == 3
    assert len(payload["sections"]) == 3
    assert [s["data"]["input_sequence_length"] for s in payload["sections"]] == [
        128,
        1024,
        8192,
    ]
    assert list((tmp_path).glob("report_*.md"))


def test_checkpoint_failure_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    acc = BlockAccumulator()
    acc.accept([_block(128)], envelope=ENVELOPE)

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "report_module.generator.ReportGenerator.generate", boom, raising=True
    )
    # A sweep still producing data must never die because a render failed.
    assert checkpoint_report(tmp_path, accumulator=acc) is False


def test_prepare_runs_before_the_partial_flag_is_set(tmp_path: Path) -> None:
    """``prepare`` is where the workflow injects its metadata.

    ``WorkflowExecution.inject_metadata`` writes ``report_partial=False`` (it is
    also the end-of-run path), so the checkpoint must mark itself partial after
    ``prepare`` -- otherwise every checkpoint would claim to be finished.
    """
    acc = BlockAccumulator()
    acc.accept([_block(128)], envelope=ENVELOPE)

    def prepare(schema) -> None:
        schema.metadata["workflow"] = "benchmarks"
        schema.metadata["run_command"] = "run.py --workflow benchmarks"
        schema.metadata["report_partial"] = False

    assert checkpoint_report(tmp_path, accumulator=acc, prepare=prepare) is True

    meta = json.loads(_data_files(tmp_path)[0].read_text())["metadata"]
    assert meta["workflow"] == "benchmarks"
    assert meta["run_command"] == "run.py --workflow benchmarks"
    assert meta["report_partial"] is True


def test_prepare_does_not_leak_into_the_accumulator(tmp_path: Path) -> None:
    acc = BlockAccumulator()
    acc.accept([_block(128)], envelope=ENVELOPE)

    def prepare(schema) -> None:
        schema.metadata["workflow"] = "benchmarks"

    checkpoint_report(tmp_path, accumulator=acc, prepare=prepare)

    assert "workflow" not in acc.envelope
    assert "report_partial" not in acc.envelope

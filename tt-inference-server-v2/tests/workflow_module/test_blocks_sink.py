# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end tests for the Block accumulator + ReportGenerator pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from report_module import ReportGenerator
from report_module.schema import Block
from workflow_module import BlockAccumulator


SWEEP_ENVELOPE = {
    "model_name": "tt-sdxl",
    "device": "n300",
    "generated_at": "2026-05-05 12:00:00",
}


def _benchmark_block() -> Block:
    return Block(
        kind="image_benchmark",
        id="tt-sdxl_n300",
        data={
            "Benchmarks": {
                "num_requests": 5,
                "ttft": 1.23,
                "inference_steps_per_second": 16.6,
            },
        },
    )


def _eval_block() -> Block:
    return Block(
        kind="image_eval",
        id="tt-sdxl_n300",
        data={
            "task_name": "sdxl-prompts",
            "tolerance": 0.05,
            "fid_score": 12.3,
            "average_clip": 0.31,
            "score": None,
        },
    )


def test_accumulator_preserves_insertion_order():
    acc = BlockAccumulator()
    acc.accept([_benchmark_block()], envelope=SWEEP_ENVELOPE)
    acc.accept([_eval_block()])
    kinds = [b.kind for b in acc.blocks]
    assert kinds == ["image_benchmark", "image_eval"]


def test_accumulator_rejects_non_blocks():
    acc = BlockAccumulator()
    with pytest.raises(TypeError):
        acc.accept([{"kind": "image_benchmark"}])  # type: ignore[list-item]


def test_envelope_first_write_wins():
    acc = BlockAccumulator()
    acc.accept([_benchmark_block()], envelope=SWEEP_ENVELOPE)
    acc.accept([_eval_block()], envelope={"model_name": "different-model"})
    assert acc.envelope["model_name"] == "tt-sdxl"


def test_build_schema_uses_recorded_envelope():
    acc = BlockAccumulator()
    acc.accept([_benchmark_block(), _eval_block()], envelope=SWEEP_ENVELOPE)

    schema = acc.build_schema()
    assert schema.metadata["model_name"] == "tt-sdxl"
    assert schema.metadata["device"] == "n300"
    assert schema.metadata["generated_at"] == "2026-05-05 12:00:00"
    assert schema.metadata["report_id"]  # synthesised, non-empty
    assert [b.kind for b in schema.sections] == ["image_benchmark", "image_eval"]


def test_build_schema_round_trips_through_report_generator(tmp_path: Path):
    acc = BlockAccumulator()
    acc.accept([_benchmark_block(), _eval_block()], envelope=SWEEP_ENVELOPE)
    schema = acc.build_schema()

    result = ReportGenerator().generate(schema, tmp_path)

    assert result.markdown_path.exists()
    assert result.json_path.exists()
    md = result.markdown_path.read_text()
    # The release header carries the model + device from accumulator metadata.
    assert "tt-sdxl on n300" in md
    # Each kind gets its own H3 section heading via render_generic_table.
    assert "### Image Benchmark" in md
    assert "### Image Eval" in md


def test_sweep_metadata_lives_only_at_top_level(tmp_path: Path):
    """Sweep-level model/device/timestamp live once in top-level metadata —
    never duplicated onto every block's serialised payload."""
    acc = BlockAccumulator()
    acc.accept([_benchmark_block()], envelope=SWEEP_ENVELOPE)
    schema = acc.build_schema()
    result = ReportGenerator().generate(schema, tmp_path)

    payload = result.json_path.read_text()
    assert payload.count('"model"') == 0
    assert payload.count('"device"') == 1  # top-level metadata.device
    assert payload.count('"timestamp"') == 0

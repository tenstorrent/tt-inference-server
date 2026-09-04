# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Multi-row benchmark blocks must still render the parent target_checks table."""

from report_module.renderers import render_generic_table
from report_module.schema import Block


def test_multi_row_sweep_renders_parent_target_checks():
    block = Block(
        kind="benchmarks",
        title="Audio Benchmark",
        data={
            "records": [
                {"name": "Benchmarks 30s", "rtr": 6.2, "ttft": 1.5},
                {"name": "Benchmarks 60s", "rtr": 13.4, "ttft": 0.86},
            ],
            "target_checks": {
                "functional": {
                    "ttft": 8000,
                    "ttft_check": "pass",
                    "rtr": 0.95,
                    "rtr_check": "pass",
                },
                "target": {
                    "ttft": 800,
                    "ttft_check": "fail",
                    "rtr": 9.5,
                    "rtr_check": "pass",
                },
            },
        },
    )
    rendered = render_generic_table(
        block, {"model_name": "Qwen3-ASR-1.7B", "device": "galaxy"}
    )
    assert "Target Checks" in rendered
    assert "functional" in rendered
    assert "target" in rendered
    assert "Benchmarks 30s" in rendered
    assert "Benchmarks 60s" in rendered

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for the spec_decode reporting integration in summary_report.py."""

import json
from pathlib import Path

import pytest

from benchmarking.summary_report import (
    create_spec_decode_display_dict,
    extract_params_from_filename,
    process_benchmark_file,
    render_spec_decode_sections,
)


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def test_extract_params_spec_decode():
    name = (
        "benchmark_spec_decode_id_tt-transformers_Llama-3.1-8B-Instruct_"
        "gpu_2026-05-20_10-00-00_spec_bench_osl-128_maxcon-4_n-16.json"
    )
    params = extract_params_from_filename(name)
    assert params["task_type"] == "spec_decode"
    assert params["public_dataset"] == "spec_bench"
    assert params["device"] == "gpu"
    assert params["output_sequence_length"] == 128
    assert params["max_con"] == 4
    assert params["num_requests"] == 16


def test_extract_params_spec_decode_natural_length():
    # osl and n omitted: natural EOS decode over the whole dataset.
    name = (
        "benchmark_spec_decode_modelid_gpu_2026-05-20_10-00-00_spec_bench_maxcon-1.json"
    )
    params = extract_params_from_filename(name)
    assert params["task_type"] == "spec_decode"
    assert params["output_sequence_length"] is None
    assert params["num_requests"] is None


def test_extract_params_speed_bench_throughput_slug():
    # Multi-token slugs (speed_bench_throughput_1k) must be captured whole.
    name = (
        "benchmark_spec_decode_modelid_gpu_"
        "2026-05-20_10-00-00_speed_bench_throughput_1k_"
        "osl-512_maxcon-4_n-16.json"
    )
    params = extract_params_from_filename(name)
    assert params["public_dataset"] == "speed_bench_throughput_1k"


def test_process_spec_decode_file_lifts_metrics(tmp_path):
    name = (
        "benchmark_spec_decode_mid_gpu_"
        "2026-05-20_10-00-00_spec_bench_osl-128_maxcon-4_n-16.json"
    )
    path = tmp_path / name
    _write(
        path,
        {
            "mean_ttft_ms": 50.0,
            "mean_tpot_ms": 10.0,
            "mean_e2el_ms": 200.0,
            "total_input_tokens": 1600,
            "total_output_tokens": 2048,
            "total_token_throughput": 300.0,
            "num_prompts": 16,
            "spec_decode_metrics": {
                "acceptance_rate": 0.82,
                "mean_accepted_length": 2.7,
                "accepted_tokens": 1300,
                "draft_tokens": 1585,
                "num_drafts": 480,
                "public_dataset": "spec_bench",
            },
        },
    )
    metrics = process_benchmark_file(str(path))
    assert metrics["task_type"] == "spec_decode"
    assert metrics["public_dataset"] == "spec_bench"
    assert metrics["acceptance_rate"] == pytest.approx(0.82)
    assert metrics["mean_accepted_length"] == pytest.approx(2.7)
    assert metrics["accepted_tokens"] == 1300
    assert metrics["draft_tokens"] == 1585
    # Falls through the default vLLM-bench branch → mean_e2el_ms is preserved
    assert metrics["mean_e2el_ms"] == pytest.approx(200.0, rel=1e-3)


def test_process_spec_decode_handles_missing_annotation(tmp_path):
    # If the Prometheus scrape failed, the JSON won't have spec_decode_metrics.
    # The processor should still emit a spec_decode result with None fields
    # rather than crashing.
    name = (
        "benchmark_spec_decode_mid_gpu_"
        "2026-05-20_10-00-00_spec_bench_osl-128_maxcon-1_n-4.json"
    )
    path = tmp_path / name
    _write(
        path,
        {
            "mean_ttft_ms": 50.0,
            "mean_tpot_ms": 10.0,
            "mean_e2el_ms": 200.0,
            "total_input_tokens": 400,
            "num_prompts": 4,
        },
    )
    metrics = process_benchmark_file(str(path))
    assert metrics["task_type"] == "spec_decode"
    # No annotation → acceptance_rate should be None/n/a
    assert metrics.get("acceptance_rate") in (None, "n/a")


def test_create_spec_decode_display_dict_columns():
    result = {
        "public_dataset": "spec_bench",
        "output_sequence_length": 128,
        "max_con": 4,
        "num_requests": 16,
        "acceptance_rate": 0.82,
        "mean_accepted_length": 2.7,
        "mean_ttft_ms": 50.0,
        "mean_tpot_ms": 10.0,
        "mean_e2el_ms": 200.0,
        "total_token_throughput": 300.0,
    }
    d = create_spec_decode_display_dict(result)
    assert d["Dataset"] == "spec_bench"
    assert d["Accept Rate"] == "0.82"
    assert d["E2EL (ms)"] == "200.0"


def test_render_returns_empty_when_no_results():
    sections = render_spec_decode_sections([], "model", "gpu")
    assert sections == []


def test_render_per_run_section():
    per_run = [
        {
            "task_type": "spec_decode",
            "public_dataset": "speed_bench_coding",
            "output_sequence_length": 128,
            "max_con": 1,
            "mean_e2el_ms": 100.0,
            "acceptance_rate": 0.8,
        },
        {
            "task_type": "spec_decode",
            "public_dataset": "spec_bench",
            "output_sequence_length": 128,
            "max_con": 1,
            "mean_e2el_ms": 200.0,
        },
    ]
    sections = render_spec_decode_sections(per_run, "model", "gpu")
    assert len(sections) == 1
    assert "Speculative Decoding Per-Run" in sections[0]
    # Rows sorted by dataset: spec_bench before speed_bench_coding.
    lines = sections[0].splitlines()
    spec_bench_line = next(i for i, line in enumerate(lines) if "| spec_bench" in line)
    coding_line = next(
        i for i, line in enumerate(lines) if "speed_bench_coding" in line
    )
    assert spec_bench_line < coding_line

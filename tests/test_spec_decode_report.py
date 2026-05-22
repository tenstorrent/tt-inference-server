# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for the spec_decode reporting integration in summary_report.py."""

import json
from pathlib import Path

import pytest

from benchmarking.summary_report import (
    create_spec_decode_display_dict,
    create_spec_decode_pair_display_dict,
    extract_params_from_filename,
    process_benchmark_file,
    render_spec_decode_sections,
)


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Filename regex
# ---------------------------------------------------------------------------


def test_extract_params_spec_decode_spec_role():
    name = (
        "benchmark_spec_decode_spec_id_tt-transformers_Llama-3.1-8B-Instruct_"
        "gpu_2026-05-20_10-00-00_spec_bench_writing_osl-128_maxcon-4_n-16.json"
    )
    params = extract_params_from_filename(name)
    assert params["task_type"] == "spec_decode"
    assert params["endpoint_role"] == "spec"
    assert params["dataset_kind"] == "spec_bench"
    assert params["category"] == "writing"
    assert params["device"] == "gpu"
    assert params["output_sequence_length"] == 128
    assert params["max_con"] == 4
    assert params["num_requests"] == 16


def test_extract_params_spec_decode_baseline_role():
    name = (
        "benchmark_spec_decode_baseline_modelid_gpu_"
        "2026-05-20_10-00-00_spec_bench_writing_osl-512_maxcon-1_n-4.json"
    )
    params = extract_params_from_filename(name)
    assert params["task_type"] == "spec_decode"
    assert params["endpoint_role"] == "baseline"
    assert params["dataset_kind"] == "spec_bench"


def test_extract_params_spec_decode_pair_role():
    name = (
        "benchmark_spec_decode_pair_modelid_gpu_"
        "2026-05-20_10-00-00_spec_bench_writing_osl-128_maxcon-4_n-16.json"
    )
    params = extract_params_from_filename(name)
    assert params["task_type"] == "spec_decode_pair"
    assert params["endpoint_role"] == "pair"


def test_extract_params_speed_bench_with_subset_in_category():
    # The slug encodes the subset inside the category capture because the
    # filename collapses them with underscores. The structured fields live
    # in the JSON body; the regex just needs to anchor on osl-/maxcon-/n-.
    name = (
        "benchmark_spec_decode_spec_modelid_gpu_"
        "2026-05-20_10-00-00_speed_bench_all_throughput_1k_"
        "osl-512_maxcon-4_n-16.json"
    )
    params = extract_params_from_filename(name)
    assert params["dataset_kind"] == "speed_bench"
    assert params["category"] == "all_throughput_1k"


# ---------------------------------------------------------------------------
# process_benchmark_file
# ---------------------------------------------------------------------------


def test_process_spec_decode_pair_file(tmp_path):
    name = (
        "benchmark_spec_decode_pair_mid_gpu_"
        "2026-05-20_10-00-00_spec_bench_writing_osl-128_maxcon-4_n-16.json"
    )
    path = tmp_path / name
    _write(
        path,
        {
            "benchmark_kind": "spec_decode_pair",
            "slug": "spec_bench_writing_osl-128_maxcon-4_n-16",
            "speedup_p50_e2el": 1.8,
            "speedup_p95_e2el": 1.6,
            "speedup_p99_e2el": 1.5,
            "output_tput_ratio": 1.75,
            "spec_acceptance_rate": 0.82,
            "baseline_acceptance_rate": None,
            "dataset_kind": "spec_bench",
            "category": "writing",
        },
    )
    metrics = process_benchmark_file(str(path))
    assert metrics["task_type"] == "spec_decode_pair"
    assert metrics["endpoint_role"] == "pair"
    assert metrics["speedup_p50_e2el"] == pytest.approx(1.8)
    assert metrics["output_tput_ratio"] == pytest.approx(1.75)
    assert metrics["spec_acceptance_rate"] == pytest.approx(0.82)
    assert metrics["category"] == "writing"


def test_process_spec_decode_spec_file_lifts_metrics(tmp_path):
    name = (
        "benchmark_spec_decode_spec_mid_gpu_"
        "2026-05-20_10-00-00_spec_bench_writing_osl-128_maxcon-4_n-16.json"
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
                "dataset_kind": "spec_bench",
                "category": "writing",
                "endpoint_role": "spec",
            },
        },
    )
    metrics = process_benchmark_file(str(path))
    assert metrics["task_type"] == "spec_decode"
    assert metrics["endpoint_role"] == "spec"
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
        "benchmark_spec_decode_spec_mid_gpu_"
        "2026-05-20_10-00-00_spec_bench_writing_osl-128_maxcon-1_n-4.json"
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


# ---------------------------------------------------------------------------
# Display dicts
# ---------------------------------------------------------------------------


def test_create_spec_decode_display_dict_columns():
    result = {
        "endpoint_role": "spec",
        "dataset_kind": "spec_bench",
        "category": "writing",
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
    assert d["Role"] == "spec"
    assert d["Dataset"] == "spec_bench"
    assert d["Category"] == "writing"
    assert d["Accept Rate"] == "0.82"
    assert d["E2EL (ms)"] == "200.0"


def test_create_spec_decode_pair_display_dict_columns():
    result = {
        "dataset_kind": "spec_bench",
        "category": "writing",
        "output_sequence_length": 128,
        "max_con": 4,
        "speedup_p50_e2el": 1.8,
        "speedup_p95_e2el": 1.6,
        "speedup_p99_e2el": 1.5,
        "output_tput_ratio": 1.75,
        "spec_acceptance_rate": 0.82,
    }
    d = create_spec_decode_pair_display_dict(result)
    assert d["Speedup p50"] == "1.8"
    assert d["Speedup p95"] == "1.6"
    assert d["Output Tput Ratio"] == "1.75"
    assert d["Accept Rate"] == "0.82"


# ---------------------------------------------------------------------------
# render_spec_decode_sections
# ---------------------------------------------------------------------------


def test_render_returns_empty_when_no_results():
    sections = render_spec_decode_sections([], [], "model", "gpu")
    assert sections == []


def test_render_per_run_section_only():
    per_run = [
        {
            "task_type": "spec_decode",
            "endpoint_role": "baseline",
            "dataset_kind": "spec_bench",
            "category": "writing",
            "output_sequence_length": 128,
            "max_con": 1,
            "mean_e2el_ms": 200.0,
        },
        {
            "task_type": "spec_decode",
            "endpoint_role": "spec",
            "dataset_kind": "spec_bench",
            "category": "writing",
            "output_sequence_length": 128,
            "max_con": 1,
            "mean_e2el_ms": 100.0,
            "acceptance_rate": 0.8,
        },
    ]
    sections = render_spec_decode_sections(per_run, [], "model", "gpu")
    assert len(sections) == 1
    assert "Speculative Decoding Per-Run" in sections[0]
    # baseline row should come before spec row in the rendered table
    lines = sections[0].splitlines()
    baseline_line = next(i for i, line in enumerate(lines) if "baseline" in line)
    # match "spec" as a cell value, not as "spec_bench" in the Dataset column
    spec_line = next(
        i
        for i, line in enumerate(lines)
        if line.lstrip().startswith("|") and " spec " in line[: line.find("spec_bench")]
    )
    assert baseline_line < spec_line


def test_render_includes_pair_section_when_pair_results_present():
    pair = [
        {
            "task_type": "spec_decode_pair",
            "dataset_kind": "spec_bench",
            "category": "writing",
            "output_sequence_length": 128,
            "max_con": 1,
            "speedup_p50_e2el": 1.8,
            "speedup_p95_e2el": 1.6,
            "speedup_p99_e2el": 1.5,
            "output_tput_ratio": 1.75,
            "spec_acceptance_rate": 0.82,
        }
    ]
    sections = render_spec_decode_sections([], pair, "model", "gpu")
    assert len(sections) == 1
    assert "Speedup vs Baseline" in sections[0]
    assert "Speedup p50" in sections[0]

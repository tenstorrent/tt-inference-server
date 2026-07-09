# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the LLM benchmark parsers (aiperf / genai-perf / vLLM).

Each parser turns one tool's raw result into a flat, scalar report record
keyed with the shared display names. The key invariant — and the reason
these parsers were rewritten — is that a record must be *flat*: a sweep
of N points is merged into one multi-row table, and any nested dict/list
value would render as a giant JSON cell instead of a column.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import pytest

from llm_module.parsers.aiperf import AIPerfParser
from llm_module.parsers.genai_perf import GenAIPerfParser
from llm_module.parsers.vllm import VLLMBenchParser
from report_module.display import display_name

MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# NVIDIA aiperf / genai-perf share a schema: metrics are top-level mappings
# of stat -> value. A clean run reports no errors (error_request_count None).
AIPERF_RAW: Dict[str, Any] = {
    "schema_version": "1.0",
    "aiperf_version": "0.5.0",
    "benchmark_id": "abc",
    "start_time": "2026-06-17T16:08:57.327203",
    "end_time": "2026-06-17T16:09:00.000000",
    "input_config": {
        "endpoint": {"type": "chat", "model_names": [MODEL]},
        "loadgen": {"concurrency": 32},
    },
    "request_count": {"avg": 256, "unit": "requests"},
    "error_request_count": None,
    "input_sequence_length": {"avg": 16384.0, "unit": "tokens"},
    "output_sequence_length": {"avg": 126.5, "unit": "tokens"},
    "time_to_first_token": {"avg": 1240.09, "p50": 786.0, "p99": 2093.6, "unit": "ms"},
    "inter_token_latency": {"avg": 129.8, "unit": "ms"},
    "request_latency": {"avg": 17525.9, "unit": "ms"},
    "output_token_throughput_per_user": {"avg": 7.71, "unit": "tokens/sec/user"},
    "output_token_throughput": {"avg": 228.47, "unit": "tokens/sec"},
    "request_throughput": {"avg": 1.81, "unit": "requests/sec"},
}

GENAI_RAW: Dict[str, Any] = {
    "input_config": {"model": MODEL, "concurrency": [16]},
    "request_count": {"avg": 64},
    "input_sequence_length": {"avg": 2048.0},
    "output_sequence_length": {"avg": 128.0},
    "time_to_first_token": {"avg": 300.0, "p50": 250.0, "p99": 900.0},
    "inter_token_latency": {"avg": 30.0},
    "request_latency": {"avg": 4000.0},
    "output_token_throughput_per_user": {"avg": 33.3},
    "output_token_throughput": {"avg": 500.0},
    "request_throughput": {"avg": 2.5},
}

# vLLM `bench serve --save-result` is flat: stat-prefixed keys, token totals.
VLLM_RAW: Dict[str, Any] = {
    "model_id": MODEL,
    "date": "20260617-164242",
    "max_concurrency": 32,
    "completed": 128,
    "failed": 0,
    "num_prompts": 128,
    "total_input_tokens": 262144,
    "total_output_tokens": 16384,
    "mean_ttft_ms": 350.5,
    "median_ttft_ms": 300.1,
    "p99_ttft_ms": 1200.9,
    "mean_tpot_ms": 28.7,
    "mean_e2el_ms": 4200.0,
    "request_throughput": 2.1,
    "output_throughput": 480.0,
    "total_token_throughput": 900.0,
}


def _assert_flat(data: Mapping[str, Any]) -> None:
    """Every field must be a scalar (or None) so it renders as one cell."""
    for key, value in data.items():
        assert not isinstance(value, (dict, list, tuple, set)), (
            f"field {key!r} is nested ({type(value).__name__}); it would render "
            "as a JSON blob in the merged sweep table"
        )


@pytest.mark.parametrize(
    "parser,raw",
    [
        (AIPerfParser(), AIPERF_RAW),
        (GenAIPerfParser(), GENAI_RAW),
        (VLLMBenchParser(), VLLM_RAW),
    ],
)
def test_record_is_flat_with_envelope(parser, raw):
    block = parser.parse(raw, device="N150")
    assert block.kind == parser.kind
    # envelope fields live in targets, never duplicated into the row data
    assert block.targets.get("model") == MODEL
    assert block.targets.get("device") == "N150"
    assert block.targets.get("timestamp")
    for envelope in ("kind", "model", "device", "timestamp"):
        assert envelope not in block.data
    _assert_flat(block.data)


def test_aiperf_maps_metrics_and_keeps_isl_integer():
    data = AIPerfParser().parse(AIPERF_RAW, device="N150").data
    assert data["concurrency"] == 32
    assert data["num_requests"] == 256
    # ISL kept as int so 16384 never renders as "1.638e+04"
    assert data["input_sequence_length"] == 16384
    assert isinstance(data["input_sequence_length"], int)
    # OSL stays float (mean can be fractional)
    assert data["output_sequence_length"] == pytest.approx(126.5)
    assert data["mean_ttft_ms"] == pytest.approx(1240.09)
    assert data["p50_ttft"] == pytest.approx(786.0)
    assert data["p99_ttft"] == pytest.approx(2093.6)
    assert data["mean_tpot_ms"] == pytest.approx(129.8)
    assert data["mean_e2el_ms"] == pytest.approx(17525.9)
    assert data["tput_user"] == pytest.approx(7.71)
    assert data["tps_decode_throughput"] == pytest.approx(228.47)
    assert data["request_throughput"] == pytest.approx(1.81)


def test_aiperf_errors_none_when_clean():
    data = AIPerfParser().parse(AIPERF_RAW, device="N150").data
    # None -> the Errors column drops out of an all-successful sweep
    assert data["error_request_count"] is None


@pytest.mark.parametrize("err_value", [3, {"avg": 3}])
def test_aiperf_errors_surface_when_present(err_value):
    raw = {**AIPERF_RAW, "error_request_count": err_value}
    data = AIPerfParser().parse(raw, device="N150").data
    assert data["error_request_count"] == 3


def test_genai_unwraps_concurrency_list_and_resolves_model():
    block = GenAIPerfParser().parse(GENAI_RAW, device="N150")
    data = block.data
    assert block.targets.get("model") == MODEL
    # genai-perf reports concurrency as a single-element list
    assert data["concurrency"] == 16
    assert data["num_requests"] == 64
    assert data["input_sequence_length"] == 2048
    assert data["p50_ttft"] == pytest.approx(250.0)
    assert data["request_throughput"] == pytest.approx(2.5)
    # genai-perf has no error counter, so the field is absent (not None)
    assert "error_request_count" not in data


def test_genai_reads_nested_perf_analyzer_concurrency():
    # Newer genai-perf nests concurrency under perf_analyzer.stimulus.
    raw = {
        "input_config": {
            "model": MODEL,
            "perf_analyzer": {"stimulus": {"concurrency": 32}},
        },
        "time_to_first_token": {"avg": 100.0},
    }
    data = GenAIPerfParser().parse(raw, device="N150").data
    assert data["concurrency"] == 32


def test_vllm_derives_isl_osl_and_maps_percentiles():
    data = VLLMBenchParser().parse(VLLM_RAW, device="N150").data
    assert data["concurrency"] == 32
    assert data["num_requests"] == 128
    # ISL/OSL are derived from token totals / completed requests
    assert data["input_sequence_length"] == 2048
    assert isinstance(data["input_sequence_length"], int)
    assert data["output_sequence_length"] == pytest.approx(128.0)
    # vLLM's "median" maps to p50; p99 carried through
    assert data["p50_ttft"] == pytest.approx(300.1)
    assert data["p99_ttft"] == pytest.approx(1200.9)
    assert data["tps_decode_throughput"] == pytest.approx(480.0)
    assert data["error_request_count"] is None


def test_vllm_errors_surface_from_failed_count():
    data = VLLMBenchParser().parse({**VLLM_RAW, "failed": 4}, device="N150").data
    assert data["error_request_count"] == 4


def test_vllm_handles_zero_completed_without_dividing():
    raw = {**VLLM_RAW, "completed": 0}
    data = VLLMBenchParser().parse(raw, device="N150").data
    assert data["num_requests"] == 0
    # no completed requests -> ISL/OSL cannot be derived, stay None
    assert data["input_sequence_length"] is None
    assert data["output_sequence_length"] is None


def test_missing_metrics_yield_none_not_keyerror():
    # An empty raw must not raise; numeric fields fall back to None.
    for parser in (AIPerfParser(), GenAIPerfParser(), VLLMBenchParser()):
        data = parser.parse({}, device="N150").data
        _assert_flat(data)
        assert data["mean_ttft_ms"] is None


def test_canonical_keys_have_display_names():
    # The flat keys must resolve to the shared, human display headers.
    assert display_name("mean_ttft_ms") == "TTFT (ms)"
    assert display_name("mean_tpot_ms") == "TPOT (ms)"
    assert display_name("tput_user") == "Tput User (TPS)"
    assert display_name("error_request_count") == "Errors"
    assert display_name("input_sequence_length") == "ISL"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

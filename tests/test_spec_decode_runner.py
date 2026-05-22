# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for benchmarking.run_spec_decode_benchmarks helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarking import run_spec_decode_benchmarks as runner
from benchmarking.run_spec_decode_benchmarks import (
    build_pair_filename,
    build_result_filename,
    build_spec_serve_cmd,
    extract_slug_from_filename,
    pair_phase,
    parse_endpoint_url,
    parse_workflow_args,
    select_profile,
    warmup_endpoint,
)
from benchmarking.spec_decode_common import SpecDecodeRunSpec


def _spec_bench_run() -> SpecDecodeRunSpec:
    return SpecDecodeRunSpec(
        dataset_kind="spec_bench",
        category="writing",
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
    )


def _speed_bench_run() -> SpecDecodeRunSpec:
    return SpecDecodeRunSpec(
        dataset_kind="speed_bench",
        category=None,
        output_len=512,
        max_concurrency=4,
        num_prompts=16,
        speed_bench_subset="throughput_1k",
    )


def test_parse_workflow_args_empty():
    assert parse_workflow_args(None) == {}
    assert parse_workflow_args("") == {}


def test_parse_workflow_args_key_value_pairs():
    parsed = parse_workflow_args(
        "phase=baseline url=http://localhost:8000 warmup-requests=8"
    )
    assert parsed["phase"] == "baseline"
    assert parsed["url"] == "http://localhost:8000"
    assert parsed["warmup-requests"] == "8"


def test_parse_endpoint_url_with_port():
    assert parse_endpoint_url("http://example.com:9000") == ("example.com", 9000)


def test_parse_endpoint_url_uses_default_port_when_missing():
    assert parse_endpoint_url("http://example.com") == ("example.com", 8000)


def test_parse_endpoint_url_handles_missing_scheme():
    assert parse_endpoint_url("127.0.0.1:1234") == ("127.0.0.1", 1234)


def test_build_result_filename_uses_prefix_and_slug():
    name = build_result_filename(
        "tt-vllm-plugin_Llama-3.1-8B-Instruct_n300",
        "n300",
        _spec_bench_run(),
        run_timestamp="2026-05-20_10-00-00",
    )
    assert name.startswith("benchmark_spec_decode_spec_")
    assert "n300" in name
    assert "spec_bench_writing" in name


def test_build_result_filename_baseline_role():
    name = build_result_filename(
        "modelid", "n300", _spec_bench_run(),
        role="baseline", run_timestamp="2026-05-20_10-00-00",
    )
    assert name.startswith("benchmark_spec_decode_baseline_")


def test_build_pair_filename_uses_pair_role():
    name = build_pair_filename(
        "modelid", "n300", _spec_bench_run(), run_timestamp="2026-05-20_10-00-00"
    )
    assert name.startswith("benchmark_spec_decode_pair_")


def test_extract_slug_from_filename_round_trip():
    name = build_result_filename(
        "model123", "gpu", _spec_bench_run(), run_timestamp="2026-05-20_10-00-00"
    )
    parts = extract_slug_from_filename(name)
    assert parts is not None
    assert parts["role"] == "spec"
    assert parts["model_id"] == "model123"
    assert parts["device"] == "gpu"
    assert parts["slug"] == "spec_bench_writing_osl-128_maxcon-4_n-16"


def test_extract_slug_returns_none_for_other_filenames():
    assert extract_slug_from_filename("benchmark_id_x_2026-04-01_y.json") is None
    assert extract_slug_from_filename("random.json") is None


def test_build_spec_serve_cmd_spec_bench():
    cmd = build_spec_serve_cmd(
        benchmark_script=Path("/venv/bin/vllm"),
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        host="127.0.0.1", port=8000,
        run_spec=_spec_bench_run(),
        result_path=Path("/tmp/out.json"),
    )
    assert cmd[0:3] == ["/venv/bin/vllm", "bench", "serve"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--dataset-name") + 1] == "spec_bench"
    assert cmd[cmd.index("--spec-bench-category") + 1] == "writing"
    assert "--speed-bench-category" not in cmd


def test_build_spec_serve_cmd_speed_bench_includes_subset():
    cmd = build_spec_serve_cmd(
        benchmark_script=Path("/venv/bin/vllm"),
        hf_model_repo="x/y",
        host="10.0.0.5", port=8001,
        run_spec=_speed_bench_run(),
        result_path=Path("/tmp/out.json"),
    )
    assert cmd[cmd.index("--dataset-name") + 1] == "speed_bench"
    assert cmd[cmd.index("--speed-bench-dataset-subset") + 1] == "throughput_1k"
    # category=None must omit the --speed-bench-category flag entirely;
    # passing a sentinel like "default" exact-matches against the JSONL's
    # category column and loads 0 prompts (the original bug).
    assert "--speed-bench-category" not in cmd


def test_build_spec_serve_cmd_includes_auth_header_when_token_set():
    cmd = build_spec_serve_cmd(
        benchmark_script=Path("/venv/bin/vllm"),
        hf_model_repo="x/y",
        host="127.0.0.1", port=8000,
        run_spec=_spec_bench_run(),
        result_path=Path("/tmp/out.json"),
        jwt_token="encoded.jwt",
    )
    assert "--header" in cmd
    assert cmd[cmd.index("--header") + 1] == "Authorization: Bearer encoded.jwt"


def test_build_spec_serve_cmd_omits_header_when_no_token():
    cmd = build_spec_serve_cmd(
        benchmark_script=Path("/venv/bin/vllm"),
        hf_model_repo="x/y",
        host="127.0.0.1", port=8000,
        run_spec=_spec_bench_run(),
        result_path=Path("/tmp/out.json"),
    )
    assert "--header" not in cmd


def test_select_profile_smoke_for_smoke_test_mode():
    profile = select_profile(SimpleNamespace(limit_samples_mode="smoke-test"))
    # Smoke profile exercises both dataset code paths with real category /
    # subset values so a future regression to the historical 0-prompt bugs
    # ("mt_bench" / "default" / "throughput") would fail in CI.
    assert {p.dataset_kind for p in profile} == {"spec_bench", "speed_bench"}
    assert all(
        p.category is None or p.category in {
            "writing", "roleplay", "reasoning", "math", "coding",
            "extraction", "stem", "humanities", "translation",
            "summarization", "qa", "math_reasoning", "rag",
        }
        for p in profile
        if p.dataset_kind == "spec_bench"
    )


def test_select_profile_full_when_no_limit_mode():
    profile = select_profile(SimpleNamespace(limit_samples_mode=None))
    assert len(profile) > 1
    assert {p.dataset_kind for p in profile} == {"spec_bench", "speed_bench"}


def test_warmup_endpoint_zero_requests_is_noop(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(runner.requests, "post", fake_post)
    successes = warmup_endpoint("http://x", "model", num_requests=0)
    assert successes == 0
    assert calls == []


def test_warmup_endpoint_sends_n_requests(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(runner.requests, "post", fake_post)
    successes = warmup_endpoint(
        "http://x:8000", "meta/llama", jwt_token="tok", num_requests=3
    )
    assert successes == 3
    assert len(calls) == 3
    url, headers, payload = calls[0]
    assert url == "http://x:8000/v1/chat/completions"
    assert headers["Authorization"] == "Bearer tok"
    assert payload["model"] == "meta/llama"
    assert payload["temperature"] == 0.0  # apples-to-apples requires deterministic output


def test_warmup_endpoint_counts_only_successes(monkeypatch):
    import requests as _requests

    call_count = {"n": 0}

    def fake_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise _requests.exceptions.ConnectionError("boom")
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(runner.requests, "post", fake_post)
    successes = warmup_endpoint("http://x", "m", num_requests=3)
    assert successes == 2  # first and third succeed; second raised


def _write_result(path: Path, **fields) -> None:
    with open(path, "w") as f:
        json.dump(fields, f)


def test_pair_phase_writes_pair_when_baseline_and_spec_match(tmp_path):
    spec = _spec_bench_run()
    baseline_path = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="baseline", run_timestamp="2026-05-20_10-00-00"
    )
    spec_path = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="spec", run_timestamp="2026-05-20_10-05-00"
    )
    _write_result(
        baseline_path,
        mean_e2el_ms=200.0, p50_e2el_ms=180.0, p95_e2el_ms=250.0,
        output_throughput=100.0,
    )
    _write_result(
        spec_path,
        mean_e2el_ms=100.0, p50_e2el_ms=90.0, p95_e2el_ms=125.0,
        output_throughput=200.0,
        spec_decode_metrics={"acceptance_rate": 0.8, "dataset_kind": "spec_bench",
                             "category": "writing"},
    )
    written = pair_phase(tmp_path)
    assert len(written) == 1
    pair_data = json.loads(written[0].read_text())
    assert pair_data["speedup_p50_e2el"] == pytest.approx(2.0)
    assert pair_data["output_tput_ratio"] == pytest.approx(2.0)
    assert pair_data["slug"] == spec.slug
    assert pair_data["dataset_kind"] == "spec_bench"
    assert pair_data["category"] == "writing"


def test_pair_phase_skips_when_only_one_role_present(tmp_path):
    spec = _spec_bench_run()
    spec_path = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="spec", run_timestamp="2026-05-20_10-00-00"
    )
    _write_result(spec_path, mean_e2el_ms=100.0)
    written = pair_phase(tmp_path)
    assert written == []


def test_pair_phase_uses_latest_timestamp_per_role(tmp_path):
    spec = _spec_bench_run()
    older_baseline = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="baseline", run_timestamp="2026-05-20_09-00-00"
    )
    newer_baseline = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="baseline", run_timestamp="2026-05-20_10-00-00"
    )
    spec_path = tmp_path / build_result_filename(
        "mid", "gpu", spec, role="spec", run_timestamp="2026-05-20_10-05-00"
    )
    # Older baseline has 4x latency, newer has 2x — newer should be chosen.
    _write_result(older_baseline, mean_e2el_ms=400.0)
    _write_result(newer_baseline, mean_e2el_ms=200.0)
    _write_result(spec_path, mean_e2el_ms=100.0)
    written = pair_phase(tmp_path)
    assert len(written) == 1
    pair_data = json.loads(written[0].read_text())
    assert pair_data["speedup_mean_e2el"] == pytest.approx(2.0)

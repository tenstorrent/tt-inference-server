# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for benchmarking.run_spec_decode_benchmarks helpers."""

from pathlib import Path
from types import SimpleNamespace

from benchmarking import run_spec_decode_benchmarks as runner
from benchmarking.run_spec_decode_benchmarks import (
    build_aiperf_cmd,
    parse_workflow_args,
    select_profile,
    warmup_endpoint,
)
from benchmarking.spec_decode_common import SpecDecodeRunSpec


def _spec_bench_run() -> SpecDecodeRunSpec:
    return SpecDecodeRunSpec(
        public_dataset="spec_bench",
        output_len=128,
        max_concurrency=4,
        num_prompts=16,
    )


def _speed_bench_throughput_run() -> SpecDecodeRunSpec:
    return SpecDecodeRunSpec(
        public_dataset="speed_bench_throughput_1k",
        output_len=512,
        max_concurrency=4,
        num_prompts=16,
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


def test_build_aiperf_cmd_spec_bench():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(),
        artifact_dir=Path("/tmp/artifacts"),
    )
    assert cmd[0:4] == ["/venv/bin/python", "-m", "aiperf", "profile"]
    assert cmd[cmd.index("--model") + 1] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cmd[cmd.index("--public-dataset") + 1] == "spec_bench"
    assert cmd[cmd.index("--concurrency") + 1] == "4"
    assert cmd[cmd.index("--request-count") + 1] == "16"
    assert cmd[cmd.index("--output-tokens-mean") + 1] == "128"
    # Deterministic, EOS-ignored: required for apples-to-apples spec-decode.
    assert "ignore_eos:true" in cmd
    assert "temperature:0" in cmd


def test_build_aiperf_cmd_speed_bench_throughput_slug():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="x/y",
        url="http://10.0.0.5:8001",
        run_spec=_speed_bench_throughput_run(),
        artifact_dir=Path("/tmp/artifacts"),
    )
    assert cmd[cmd.index("--public-dataset") + 1] == "speed_bench_throughput_1k"


def test_build_aiperf_cmd_includes_api_key_when_token_set():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="x/y",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(),
        artifact_dir=Path("/tmp/artifacts"),
        jwt_token="encoded.jwt",
    )
    assert "--api-key" in cmd
    assert cmd[cmd.index("--api-key") + 1] == "encoded.jwt"


def test_build_aiperf_cmd_omits_api_key_when_no_token():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="x/y",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(),
        artifact_dir=Path("/tmp/artifacts"),
    )
    assert "--api-key" not in cmd


def test_select_profile_smoke_for_smoke_test_mode():
    profile = select_profile(SimpleNamespace(limit_samples_mode="smoke-test"))
    # Smoke profile must touch both Spec-Bench and SPEED-Bench code paths so a
    # mistyped aiperf slug fails loudly instead of silently loading 0 prompts.
    slugs = {p.public_dataset for p in profile}
    assert "spec_bench" in slugs
    assert any(s.startswith("speed_bench_") for s in slugs)


def test_select_profile_full_when_no_limit_mode():
    profile = select_profile(SimpleNamespace(limit_samples_mode=None))
    assert len(profile) > 1
    slugs = {p.public_dataset for p in profile}
    assert "spec_bench" in slugs
    assert any(s.startswith("speed_bench_throughput_") for s in slugs)


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
    assert (
        payload["temperature"] == 0.0
    )  # apples-to-apples requires deterministic output


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



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


def _spec_bench_run(*, output_len=None, num_prompts=16) -> SpecDecodeRunSpec:
    return SpecDecodeRunSpec(
        public_dataset="spec_bench",
        max_concurrency=4,
        num_prompts=num_prompts,
        output_len=output_len,
    )


def test_parse_workflow_args_key_value_pairs():
    parsed = parse_workflow_args(
        "phase=baseline url=http://localhost:8000 warmup-requests=8"
    )
    assert parsed["phase"] == "baseline"
    assert parsed["url"] == "http://localhost:8000"
    assert parsed["warmup-requests"] == "8"
    assert parse_workflow_args(None) == {}


def test_build_aiperf_cmd_carries_run_spec_and_determinism_flags():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(output_len=128),
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
    assert "--api-key" not in cmd


def test_build_aiperf_cmd_omits_output_length_flags_when_unset():
    # Natural-EOS mode: the model should decode until its own stop, so the
    # runner must NOT pin max_tokens or pass ignore_eos. temperature:0 stays
    # because draft/target determinism is independent of output length.
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="m/x",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(),
        artifact_dir=Path("/tmp/artifacts"),
    )
    assert "--output-tokens-mean" not in cmd
    assert "--output-tokens-stddev" not in cmd
    assert "ignore_eos:true" not in cmd
    assert "temperature:0" in cmd


def test_build_aiperf_cmd_omits_request_count_when_num_prompts_unset():
    # Full-dataset mode: leaving num_prompts unset must drop --request-count
    # so aiperf falls back to its default (every prompt in the public
    # dataset). Pinning a number here would silently cap longer datasets.
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="m/x",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(num_prompts=None),
        artifact_dir=Path("/tmp/artifacts"),
    )
    assert "--request-count" not in cmd
    assert "--concurrency" in cmd


def test_build_aiperf_cmd_includes_api_key_when_token_set():
    cmd = build_aiperf_cmd(
        venv_python=Path("/venv/bin/python"),
        hf_model_repo="x/y",
        url="http://127.0.0.1:8000",
        run_spec=_spec_bench_run(),
        artifact_dir=Path("/tmp/artifacts"),
        jwt_token="encoded.jwt",
    )
    assert cmd[cmd.index("--api-key") + 1] == "encoded.jwt"


def test_select_profile_smoke_mode_picks_smoke_profile():
    profile = select_profile(SimpleNamespace(limit_samples_mode="smoke-test"))
    # Smoke profile must be non-empty and every entry must use a recognised
    # public-dataset family slug — a mistyped slug would otherwise load 0
    # prompts at runtime without an obvious error.
    assert profile, "smoke profile must contain at least one run spec"
    for run_spec in profile:
        assert run_spec.public_dataset.startswith(("spec_bench", "speed_bench")), (
            f"unexpected public_dataset slug: {run_spec.public_dataset}"
        )


def test_warmup_endpoint_sends_n_requests_with_correct_payload(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(runner.requests, "post", fake_post)
    successes = warmup_endpoint(
        "http://x:8000", "meta/llama", jwt_token="tok", num_requests=3
    )
    assert successes == 3
    url, headers, payload = calls[0]
    assert url == "http://x:8000/v1/chat/completions"
    assert headers["Authorization"] == "Bearer tok"
    assert payload["model"] == "meta/llama"
    # apples-to-apples requires deterministic output
    assert payload["temperature"] == 0.0


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

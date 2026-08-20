# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the LLM performance orchestrator."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_module.config import LLMRunConfig
from llm_module.runner import RunnerResult
from test_module.context import MediaContext
from test_module.llm_tests import llm_performance_tests as lpt


class _NamedDriver:
    def __init__(self, name):
        self.name = name


def _incoming_config():
    return LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        custom_dataset_path=Path("speed_bench_prompts_isl-128_n-8.jsonl"),
    )


def _run_with_driver(monkeypatch, tmp_path, driver, incoming, fake_ensure):
    seen = {}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, configs, server, context):
            seen["runner_configs"] = list(configs)
            return RunnerResult()

    monkeypatch.setattr(lpt, "ensure_custom_dataset", fake_ensure)
    monkeypatch.setattr(lpt, "LLMPerformanceRunner", FakeRunner)
    monkeypatch.setattr(lpt, "accept_blocks", lambda *args, **kwargs: None)

    ctx = MediaContext(
        all_params=None,
        model_spec=SimpleNamespace(
            hf_model_repo="google/diffusiongemma-26B-A4B-it",
            metadata={"tokenizer_trust_remote_code": True},
        ),
        device=SimpleNamespace(name="P300X2"),
        output_path=str(tmp_path),
        service_port=8000,
    )
    lpt.run_llm_performance(
        ctx,
        driver=driver,
        configs=[incoming],
        server_controller=object(),
    )
    return seen


def test_run_llm_performance_prepares_custom_datasets_before_the_vllm_runner(
    monkeypatch, tmp_path
):
    incoming = _incoming_config()
    prepared = LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        custom_dataset_path=tmp_path / "llm" / incoming.custom_dataset_path.name,
    )
    calls = []

    def fake_ensure(config, server, output_dir):
        calls.append((config, server.model, output_dir))
        return prepared

    seen = _run_with_driver(
        monkeypatch, tmp_path, _NamedDriver("vllm"), incoming, fake_ensure
    )

    assert calls == [(incoming, "google/diffusiongemma-26B-A4B-it", tmp_path / "llm")]
    assert seen["runner_configs"] == [prepared]


@pytest.mark.parametrize("driver_name", ["aiperf", "genai_perf", "guidellm"])
def test_run_llm_performance_skips_custom_dataset_prep_for_non_vllm_drivers(
    monkeypatch, tmp_path, driver_name
):
    incoming = _incoming_config()

    def fail_ensure(*args, **kwargs):
        raise AssertionError("non-vLLM drivers must not materialize SPEED-Bench")

    seen = _run_with_driver(
        monkeypatch, tmp_path, _NamedDriver(driver_name), incoming, fail_ensure
    )

    assert seen["runner_configs"] == [incoming]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

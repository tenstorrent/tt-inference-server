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


def test_run_llm_performance_prepares_custom_datasets_before_the_runner(
    monkeypatch, tmp_path
):
    relative = Path("speed_bench_prompts_isl-128_n-8.jsonl")
    incoming = LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        custom_dataset_path=relative,
    )
    prepared = LLMRunConfig(
        isl=128,
        osl=128,
        max_concurrency=1,
        num_prompts=8,
        custom_dataset_path=tmp_path / "llm" / relative.name,
    )
    seen = {}

    def fake_ensure(config, server, output_dir):
        seen["config"] = config
        seen["model"] = server.model
        seen["output_dir"] = output_dir
        return prepared

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
        driver=object(),
        configs=[incoming],
        server_controller=object(),
    )

    assert seen["config"] is incoming
    assert seen["model"] == "google/diffusiongemma-26B-A4B-it"
    assert seen["output_dir"] == tmp_path / "llm"
    assert seen["runner_configs"] == [prepared]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

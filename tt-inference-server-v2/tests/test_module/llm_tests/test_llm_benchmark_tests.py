# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the LLM benchmark caller: driver selection + sweep wiring."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import DeviceTypes

from test_module.llm_tests import llm_benchmark_tests as lbt


def _find_model_spec(*, model_name: str, device: DeviceTypes, impl_name: str):
    for spec in MODEL_SPECS.values():
        if (
            spec.model_name == model_name
            and spec.device_type == device
            and spec.impl.impl_name == impl_name
        ):
            return spec
    raise AssertionError(
        f"Could not find model spec for model={model_name!r}, "
        f"device={device.name}, impl={impl_name!r}"
    )


@pytest.mark.parametrize(
    "tools,expected_kind",
    [
        ("vllm", "vllm"),
        ("aiperf", "aiperf"),
        ("guidellm", "guidellm"),
        ("genai", "genai_perf"),
        ("genai_perf", "genai_perf"),
    ],
)
def test_make_driver_maps_tools_to_driver(tools, expected_kind):
    driver = lbt._make_driver(tools, Path("/tmp/venv/bin/python"))
    assert driver.name == expected_kind


def test_make_driver_rejects_unknown_tool():
    with pytest.raises(ValueError):
        lbt._make_driver("nope", None)


def test_run_llm_bench_short_circuits_on_empty_sweep(monkeypatch):
    """No configs -> no Blocks, and the perf runner is never invoked."""
    monkeypatch.setattr(
        "llm_module.benchmark_configs.get_llm_configs", lambda *a, **k: []
    )

    called = {"run": False}

    def _boom(*a, **k):
        called["run"] = True
        raise AssertionError("run_llm_performance must not run with no configs")

    monkeypatch.setattr(lbt, "run_llm_performance", _boom)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Llama-3.1-8B-Instruct"),
        device=SimpleNamespace(name="T3K"),
        runtime_config=None,
    )
    result = lbt.run_llm_bench(ctx, tools="aiperf")
    assert result.blocks == []
    assert result.ok is False
    assert called["run"] is False


def test_run_llm_bench_builds_real_runtime_spec_sweep(monkeypatch, tmp_path):
    """Exercise the real v2 LLM config path without running a benchmark."""
    source_spec = _find_model_spec(
        model_name="Qwen3-8B",
        device=DeviceTypes.N150,
        impl_name="tt-transformers",
    )
    runtime_spec = replace(
        source_spec,
        device_model_spec=replace(
            source_spec.device_model_spec,
            max_context=256,
            max_concurrency=32,
        ),
    )

    seen = {}

    def _capture(ctx, *, driver, configs, auth_token=""):
        seen["driver"] = driver.name
        seen["configs"] = configs
        from llm_module.runner import RunnerResult

        return RunnerResult(return_codes=[0])

    monkeypatch.setattr(lbt, "run_llm_performance", _capture)

    ctx = SimpleNamespace(
        model_spec=runtime_spec,
        device=runtime_spec.device_type,
        runtime_config=SimpleNamespace(limit_samples_mode=None),
        output_path=str(tmp_path),
    )
    result = lbt.run_llm_bench(
        ctx,
        tools="vllm",
        venv_python=Path("/tmp/venv/bin/python"),
    )

    assert result.ok is True
    assert seen["driver"] == "vllm"
    assert any(
        cfg.isl == 128 and cfg.osl == 128 and cfg.max_concurrency == 1
        for cfg in seen["configs"]
    )


def test_guidellm_uses_scenarios_not_sweep(monkeypatch, tmp_path):
    """--tools guidellm builds the scenario set; the ISL/OSL sweep is bypassed."""
    monkeypatch.setattr(
        "llm_module.benchmark_configs.get_llm_configs",
        lambda *a, **k: pytest.fail("sweep configs must not be built for guidellm"),
    )
    sentinel = [object()]
    monkeypatch.setattr("llm_module.build_guidellm_scenarios", lambda *a, **k: sentinel)

    seen = {}

    def _capture(ctx, *, driver, configs, auth_token=""):
        seen["configs"] = configs
        from llm_module.runner import RunnerResult

        return RunnerResult()

    monkeypatch.setattr(lbt, "run_llm_performance", _capture)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(
            model_name="Llama-3.1-8B-Instruct", hf_model_repo="org/m"
        ),
        device=SimpleNamespace(name="T3K"),
        runtime_config=SimpleNamespace(workflow_args=None),
        output_path=str(tmp_path),
    )
    lbt.run_llm_bench(ctx, tools="guidellm", venv_python=Path("/tmp/venv/bin/python"))
    assert seen["configs"] is sentinel


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

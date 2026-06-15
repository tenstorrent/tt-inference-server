# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the LLM benchmark caller: driver selection + sweep wiring."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module.llm_tests import llm_benchmark_tests as lbt


@pytest.mark.parametrize(
    "tools,expected_kind",
    [
        ("vllm", "vllm"),
        ("inferencemax", "inferencex"),
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
    blocks = lbt.run_llm_bench(ctx, tools="aiperf")
    assert blocks == []
    assert called["run"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

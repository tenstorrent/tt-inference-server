# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Routing tests for the v2 LLM-benchmark bridge.

LLM models + ``--workflow benchmarks`` are fully ported to v2 (replacing
v1's run_benchmarks.py + run_reports.py). These check the predicate, the
``can_route_to_v2`` decision, and the launcher command the bridge builds.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import v2_bridge
from workflows.workflow_types import ModelType, WorkflowType


def _spec(model_type, name="Llama-3.1-8B-Instruct"):
    return SimpleNamespace(model_type=model_type, model_name=name)


def _rc(workflow="benchmarks", **kw):
    base = dict(
        workflow=workflow,
        prefix_cache=False,
        tools="aiperf",
        jwt_secret=None,
        device="t3k",
        service_port="8000",
        docker_server=False,
        server_url=None,
        sdxl_num_prompts="100",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_llm_benchmarks_routes_to_v2():
    spec, rc = _spec(ModelType.LLM), _rc()
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc) is True
    assert v2_bridge.can_route_to_v2(spec, rc) is True


def test_prefix_cache_is_not_llm_bench_but_still_routes():
    spec, rc = _spec(ModelType.LLM), _rc(prefix_cache=True)
    # prefix-cache has its own dispatch; llm-bench must defer to it
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc) is False
    assert v2_bridge.can_route_to_v2(spec, rc) is True


def test_llm_evals_stays_on_v1():
    spec, rc = _spec(ModelType.LLM), _rc(workflow="evals")
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.EVALS, spec, rc) is False
    assert v2_bridge.can_route_to_v2(spec, rc) is False


def test_release_stays_on_v1_except_benchmarks_step():
    """Release orchestration stays on v1; only the benchmarks sub-step uses v2."""
    spec, rc = _spec(ModelType.LLM), _rc(workflow="release")
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.RELEASE, spec, rc) is False
    assert v2_bridge.can_route_to_v2(spec, rc) is False
    bench_rc = _rc(workflow="benchmarks")
    assert (
        v2_bridge._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, bench_rc) is True
    )


def test_non_routed_media_benchmarks_stays_on_v1():
    spec, rc = _spec(ModelType.IMAGE, name="not-a-routed-model"), _rc()
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc) is False
    assert v2_bridge.can_route_to_v2(spec, rc) is False


def test_build_llm_bench_cmd_forwards_tools_and_jwt():
    v2_dir = Path(__file__).resolve().parents[2] / "tt-inference-server-v2"
    cmd = v2_bridge._build_llm_bench_cmd(
        v2_dir,
        _spec(ModelType.LLM),
        _rc(tools="guidellm"),
        "/tmp/spec.json",
        Path("/tmp/out"),
    )
    assert str(v2_dir / "run_llm_bench.py") in cmd
    assert cmd[cmd.index("--workflow") + 1] == "benchmarks"
    assert cmd[cmd.index("--tools") + 1] == "guidellm"
    assert "--jwt-secret" not in cmd  # None is not forwarded

    cmd_jwt = v2_bridge._build_llm_bench_cmd(
        v2_dir,
        _spec(ModelType.LLM),
        _rc(jwt_secret="sek"),
        "/tmp/spec.json",
        Path("/tmp/out"),
    )
    assert cmd_jwt[cmd_jwt.index("--jwt-secret") + 1] == "sek"


def test_run_v2_llm_benchmark_workflow_invokes_launcher(monkeypatch, tmp_path):
    spec, rc = _spec(ModelType.LLM), _rc(server_url="https://console.example.com")
    calls = []

    def fake_run_command(cmd, logger=None, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(v2_bridge, "run_command", fake_run_command)
    monkeypatch.setattr(
        v2_bridge, "get_default_workflow_root_log_dir", lambda: tmp_path
    )

    result = v2_bridge.run_v2_llm_benchmark_workflow(spec, rc, "/tmp/spec.json")

    assert result.workflow_name == "benchmarks"
    assert result.return_code == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert "run_llm_bench.py" in cmd[1]
    assert cmd[cmd.index("--server-url") + 1] == "https://console.example.com"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

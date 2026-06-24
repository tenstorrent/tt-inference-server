# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Routing tests for the v2 LLM-benchmark bridge."""

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


def test_llm_evals_routes_to_v2():
    spec, rc = _spec(ModelType.LLM), _rc(workflow="evals")
    assert v2_bridge._is_llm_benchmark_run(WorkflowType.EVALS, spec, rc) is False
    assert v2_bridge._is_llm_eval_run(WorkflowType.EVALS, spec) is True
    assert v2_bridge.can_route_to_v2(spec, rc) is True


def test_llm_release_routes_to_v2():
    spec, rc = _spec(ModelType.LLM), _rc(workflow="release")
    assert v2_bridge._is_llm_eval_run(WorkflowType.RELEASE, spec) is True
    assert v2_bridge.can_route_to_v2(spec, rc) is True


def test_media_eval_run_is_not_llm_eval():
    spec = _spec(ModelType.IMAGE, name="not-a-routed-model")
    assert v2_bridge._is_llm_eval_run(WorkflowType.EVALS, spec) is False
    assert v2_bridge.can_route_to_v2(spec, _rc(workflow="evals")) is False


def test_release_provisions_eval_and_bench_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.LLM)
    monkeypatch.setattr(
        v2_bridge,
        "_llm_eval_venv_types",
        lambda ms: [WorkflowVenvType.EVALS_COMMON, WorkflowVenvType.EVALS_META],
    )
    venvs = v2_bridge._v2_dependency_venv_types(spec, WorkflowType.RELEASE)
    assert WorkflowVenvType.EVALS_COMMON in venvs
    assert WorkflowVenvType.EVALS_META in venvs
    assert WorkflowVenvType.V2_LLM_VLLM in venvs


def test_evals_provisions_only_eval_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.LLM)
    monkeypatch.setattr(
        v2_bridge,
        "_llm_eval_venv_types",
        lambda ms: [WorkflowVenvType.EVALS_COMMON],
    )
    venvs = v2_bridge._v2_dependency_venv_types(spec, WorkflowType.EVALS)
    assert venvs == [WorkflowVenvType.EVALS_COMMON]
    assert WorkflowVenvType.V2_LLM_VLLM not in venvs


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
    assert "--jwt-secret" not in cmd  

    cmd_jwt = v2_bridge._build_llm_bench_cmd(
        v2_dir,
        _spec(ModelType.LLM),
        _rc(jwt_secret="sek"),
        "/tmp/spec.json",
        Path("/tmp/out"),
    )
    assert cmd_jwt[cmd_jwt.index("--jwt-secret") + 1] == "sek"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

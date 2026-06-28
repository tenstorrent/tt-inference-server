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
        lambda ms, rc=None: [
            WorkflowVenvType.EVALS_COMMON,
            WorkflowVenvType.EVALS_META,
        ],
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
        lambda ms, rc=None: [WorkflowVenvType.EVALS_COMMON],
    )
    venvs = v2_bridge._v2_dependency_venv_types(spec, WorkflowType.EVALS)
    assert venvs == [WorkflowVenvType.EVALS_COMMON]
    assert WorkflowVenvType.V2_LLM_VLLM not in venvs


def _ev_task(name, venv):
    return SimpleNamespace(task_name=name, workflow_venv_type=venv)


def test_selected_eval_tasks_eval_samples_filters_to_requested():
    from workflows.workflow_types import WorkflowVenvType as V

    tasks = [
        _ev_task("meta_gpqa", V.EVALS_META),
        _ev_task("longbench_single_e", V.EVALS_COMMON),
    ]
    rc = SimpleNamespace(
        eval_samples='{"longbench_single_e": [0, 1, 2]}', limit_samples_mode=None
    )
    sel = v2_bridge._selected_eval_tasks(tasks, rc)
    assert [t.task_name for t in sel] == ["longbench_single_e"]


def test_selected_eval_tasks_smoke_keeps_first_only():
    from workflows.workflow_types import WorkflowVenvType as V

    tasks = [
        _ev_task("meta_gpqa", V.EVALS_META),
        _ev_task("longbench_single_e", V.EVALS_COMMON),
    ]
    rc = SimpleNamespace(eval_samples=None, limit_samples_mode="smoke-test")
    sel = v2_bridge._selected_eval_tasks(tasks, rc)
    assert [t.task_name for t in sel] == ["meta_gpqa"]


def test_selected_eval_tasks_no_narrowing_returns_all():
    from workflows.workflow_types import WorkflowVenvType as V

    tasks = [_ev_task("a", V.EVALS_COMMON), _ev_task("b", V.EVALS_META)]
    rc = SimpleNamespace(eval_samples=None, limit_samples_mode=None)
    assert v2_bridge._selected_eval_tasks(tasks, rc) == tasks


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


def _patch_eval_configs(monkeypatch, *, agentic):
    from workflows.workflow_types import WorkflowVenvType
    import evals.eval_config as eval_config

    venv = (
        WorkflowVenvType.EVALS_AGENTIC if agentic else WorkflowVenvType.EVALS_COMMON
    )
    cfg = SimpleNamespace(tasks=[SimpleNamespace(workflow_venv_type=venv)])
    monkeypatch.setattr(
        eval_config, "EVAL_CONFIGS", {"Llama-3.1-8B-Instruct": cfg}, raising=False
    )


def test_llm_release_includes_agentic_true_when_agentic_task(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=True)
    assert v2_bridge._llm_release_includes_agentic(_spec(ModelType.LLM)) is True


def test_llm_release_includes_agentic_false_without_agentic_task(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=False)
    assert v2_bridge._llm_release_includes_agentic(_spec(ModelType.LLM)) is False


def test_llm_release_includes_agentic_false_for_non_llm(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=True)
    assert v2_bridge._llm_release_includes_agentic(_spec(ModelType.IMAGE)) is False


def test_release_dispatches_only_engine(monkeypatch, tmp_path):
    spec, rc = _spec(ModelType.LLM), _rc(workflow="release")
    calls = []

    monkeypatch.setattr(
        v2_bridge, "run_command", lambda cmd, logger=None, env=None: calls.append(cmd) or 0
    )
    monkeypatch.setattr(v2_bridge, "_ensure_v2_venv", lambda ms: Path("/fake/python"))
    monkeypatch.setattr(
        v2_bridge, "_ensure_v2_dependency_venvs", lambda *a, **k: None
    )
    monkeypatch.setattr(
        v2_bridge, "get_default_workflow_root_log_dir", lambda: tmp_path
    )
    monkeypatch.setattr(v2_bridge, "ensure_readwriteable_dir", lambda p: None)

    results = v2_bridge.run_v2_workflows(spec, rc, str(tmp_path / "spec.json"))

    # Agentic now runs in-process inside the release engine; no extra
    # subprocess for agentic, tests, or report merging.
    assert [r.workflow_name for r in results] == ["release"]
    assert all(r.return_code == 0 for r in results)
    assert any("run.py" in c[1] for c in calls)
    assert not any("run_agentic.py" in c[1] for c in calls)
    assert not any("run_release_merge.py" in c[1] for c in calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Routing tests for the v2 LLM-benchmark bridge."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import workflow_dispatch
from workflows.workflow_types import ModelType, WorkflowType


def _spec(model_type, name="Llama-3.1-8B-Instruct"):
    return SimpleNamespace(model_type=model_type, model_name=name)


def _rc(workflow="benchmarks", **kw):
    base = dict(
        workflow=workflow,
        prefix_cache=False,
        prefix_cache_preset="full",
        prefix_cache_scenarios=None,
        prefix_cache_arrival=None,
        prefix_cache_request_rate=None,
        prefix_cache_scenarios_json=None,
        prefix_cache_trace=None,
        prefix_cache_metrics_url=None,
        spec_decode=False,
        spec_decode_preset="full",
        spec_decode_warmup_requests=None,
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
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is True
    )
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


def test_prefix_cache_is_not_llm_bench_but_still_routes():
    spec, rc = _spec(ModelType.LLM), _rc(prefix_cache=True)
    # prefix-cache has its own dispatch; llm-bench must defer to it
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is False
    )
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


def test_llm_evals_routes_to_v2():
    spec, rc = _spec(ModelType.LLM), _rc(workflow="evals")
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.EVALS, spec, rc) is False
    )
    assert workflow_dispatch._is_llm_eval_run(WorkflowType.EVALS, spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


def test_llm_release_routes_to_v2():
    spec, rc = _spec(ModelType.LLM), _rc(workflow="release")
    assert workflow_dispatch._is_llm_eval_run(WorkflowType.RELEASE, spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


def test_media_eval_run_is_not_llm_eval():
    # An IMAGE model is not an LLM eval, but it still routes to v2 by model_type.
    spec = _spec(ModelType.IMAGE, name="not-a-routed-model")
    assert workflow_dispatch._is_llm_eval_run(WorkflowType.EVALS, spec) is False
    assert workflow_dispatch.can_dispatch_to_engine(spec, _rc(workflow="evals")) is True


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "release"])
def test_vlm_routes_to_v2_like_llm(workflow):
    spec = _spec(ModelType.VLM, name="Qwen2.5-VL-7B-Instruct")
    assert (
        workflow_dispatch.can_dispatch_to_engine(spec, _rc(workflow=workflow)) is True
    )


def test_vlm_benchmarks_is_llm_benchmark_run():
    spec, rc = _spec(ModelType.VLM, name="Qwen2.5-VL-7B-Instruct"), _rc()
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is True
    )


def test_vlm_evals_is_llm_eval_run():
    spec = _spec(ModelType.VLM, name="Qwen2.5-VL-7B-Instruct")
    assert workflow_dispatch._is_llm_eval_run(WorkflowType.EVALS, spec) is True
    assert workflow_dispatch._is_llm_eval_run(WorkflowType.RELEASE, spec) is True


def test_vlm_release_provisions_llm_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.VLM, name="Qwen2.5-VL-7B-Instruct")
    monkeypatch.setattr(
        workflow_dispatch,
        "_llm_eval_venv_types",
        lambda ms, rc=None: [WorkflowVenvType.EVALS_VISION],
    )
    venvs = workflow_dispatch._engine_dependency_venv_types(spec, WorkflowType.RELEASE)
    assert WorkflowVenvType.EVALS_VISION in venvs
    assert WorkflowVenvType.LLM_VLLM in venvs


def test_release_provisions_eval_and_bench_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.LLM)
    monkeypatch.setattr(
        workflow_dispatch,
        "_llm_eval_venv_types",
        lambda ms, rc=None: [
            WorkflowVenvType.EVALS_COMMON,
            WorkflowVenvType.EVALS_META,
        ],
    )
    venvs = workflow_dispatch._engine_dependency_venv_types(spec, WorkflowType.RELEASE)
    assert WorkflowVenvType.EVALS_COMMON in venvs
    assert WorkflowVenvType.EVALS_META in venvs
    assert WorkflowVenvType.LLM_VLLM in venvs


def test_release_provisions_prefix_cache_and_spec_decode_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.LLM)
    rc = _rc(workflow="release", prefix_cache=True, spec_decode=True)
    monkeypatch.setattr(
        workflow_dispatch, "_llm_eval_venv_types", lambda ms, rc=None: []
    )
    monkeypatch.setattr(
        workflow_dispatch, "_llm_release_includes_agentic", lambda ms: False
    )

    venvs = workflow_dispatch._engine_dependency_venv_types(
        spec, WorkflowType.RELEASE, rc
    )

    assert WorkflowVenvType.LLM_VLLM in venvs
    assert WorkflowVenvType.PREFIX_CACHE in venvs
    assert WorkflowVenvType.SPEC_DECODE in venvs


def test_evals_provisions_only_eval_venvs(monkeypatch):
    from workflows.workflow_types import WorkflowVenvType

    spec = _spec(ModelType.LLM)
    monkeypatch.setattr(
        workflow_dispatch,
        "_llm_eval_venv_types",
        lambda ms, rc=None: [WorkflowVenvType.EVALS_COMMON],
    )
    venvs = workflow_dispatch._engine_dependency_venv_types(spec, WorkflowType.EVALS)
    assert venvs == [WorkflowVenvType.EVALS_COMMON]
    assert WorkflowVenvType.LLM_VLLM not in venvs


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
    sel = workflow_dispatch._selected_eval_tasks(tasks, rc)
    assert [t.task_name for t in sel] == ["longbench_single_e"]


def test_selected_eval_tasks_smoke_keeps_first_only():
    from workflows.workflow_types import WorkflowVenvType as V

    tasks = [
        _ev_task("meta_gpqa", V.EVALS_META),
        _ev_task("longbench_single_e", V.EVALS_COMMON),
    ]
    rc = SimpleNamespace(eval_samples=None, limit_samples_mode="smoke-test")
    sel = workflow_dispatch._selected_eval_tasks(tasks, rc)
    assert [t.task_name for t in sel] == ["meta_gpqa"]


def test_selected_eval_tasks_no_narrowing_returns_all():
    from workflows.workflow_types import WorkflowVenvType as V

    tasks = [_ev_task("a", V.EVALS_COMMON), _ev_task("b", V.EVALS_META)]
    rc = SimpleNamespace(eval_samples=None, limit_samples_mode=None)
    assert workflow_dispatch._selected_eval_tasks(tasks, rc) == tasks


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
@pytest.mark.parametrize(
    "model_name",
    [
        "stable-diffusion-3.5-large",
        "FLUX.1-dev",
        "Qwen-Image",
        "Qwen-Image-2512",
        "some-unlisted-image-model",
    ],
)
def test_any_image_model_routes_to_v2(model_name, workflow):
    """All IMAGE models route to v2 purely by model_type — no per-name
    allowlist — so even an unlisted image model routes."""
    spec, rc = _spec(ModelType.IMAGE, name=model_name), _rc(workflow=workflow)
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is False
    )
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


@pytest.mark.parametrize(
    "model_name",
    [
        "Wan2.2-T2V-A14B-Diffusers",
        "Wan2.2-I2V-A14B-Diffusers",
        "Wan2.2-I2V-A14B-Prodia",
        "Wan2.2-I2V-AniSora-V3.2",
        "Wan2.2-I2V-Distill-LightX2V",
        "Wan2.2-I2V-LoRA",
    ],
)
@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
def test_wan_video_routes_to_v2(model_name, workflow):
    spec, rc = _spec(ModelType.VIDEO, name=model_name), _rc(workflow=workflow)
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is False
    )
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
def test_mochi_video_routes_to_v2(workflow):
    spec, rc = _spec(ModelType.VIDEO, name="mochi-1-preview"), _rc(workflow=workflow)
    assert (
        workflow_dispatch._is_llm_benchmark_run(WorkflowType.BENCHMARKS, spec, rc)
        is False
    )
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
def test_any_video_model_routes_to_v2(workflow):
    """All VIDEO models route to v2 by model_type, not by a per-name allowlist."""
    spec, rc = (
        _spec(ModelType.VIDEO, name="some-unlisted-video-model"),
        _rc(workflow=workflow),
    )
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
@pytest.mark.parametrize(
    "model_type,model_name",
    [
        (ModelType.AUDIO, "whisper-large-v3"),
        (ModelType.AUDIO, "distil-large-v3"),
        (ModelType.AUDIO, "some-unlisted-audio-model"),
        (ModelType.TEXT_TO_SPEECH, "speecht5_tts"),
        (ModelType.TEXT_TO_SPEECH, "some-unlisted-tts-model"),
    ],
)
def test_any_audio_tts_model_routes_to_v2(model_type, model_name, workflow):
    """AUDIO / TEXT_TO_SPEECH models route to v2 by model_type — no per-name
    allowlist — matching the IMAGE/VIDEO behavior."""
    spec, rc = _spec(model_type, name=model_name), _rc(workflow=workflow)
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


@pytest.mark.parametrize("workflow", ["benchmarks", "evals", "spec_tests", "release"])
@pytest.mark.parametrize("model_type", [ModelType.CNN, ModelType.EMBEDDING])
def test_cnn_embedding_routes_to_v2(model_type, workflow):
    """CNN / EMBEDDING models are fully onboarded to v2 and route by model_type
    — no per-name allowlist — matching the IMAGE/VIDEO/AUDIO behavior."""
    spec, rc = _spec(model_type, name="some-model"), _rc(workflow=workflow)
    assert workflow_dispatch.is_engine_routed_model(spec) is True
    assert workflow_dispatch.can_dispatch_to_engine(spec, rc) is True


def test_build_llm_bench_cmd_forwards_tools_and_jwt():
    repo_root = Path(__file__).resolve().parents[2]
    cmd = workflow_dispatch._build_llm_bench_cmd(
        repo_root,
        _spec(ModelType.LLM),
        _rc(tools="guidellm"),
        "/tmp/spec.json",
        Path("/tmp/out"),
    )
    assert str(repo_root / "launchers" / "run_llm_bench.py") in cmd
    assert cmd[cmd.index("--workflow") + 1] == "benchmarks"
    assert cmd[cmd.index("--tools") + 1] == "guidellm"
    assert "--jwt-secret" not in cmd

    cmd_jwt = workflow_dispatch._build_llm_bench_cmd(
        repo_root,
        _spec(ModelType.LLM),
        _rc(jwt_secret="sek"),
        "/tmp/spec.json",
        Path("/tmp/out"),
    )
    assert cmd_jwt[cmd_jwt.index("--jwt-secret") + 1] == "sek"


def test_llm_benchmark_builds_launcher_command(monkeypatch, tmp_path):
    spec, rc = _spec(ModelType.LLM), _rc(server_url="https://console.example.com")
    monkeypatch.setattr(
        workflow_dispatch, "get_default_workflow_root_log_dir", lambda: tmp_path
    )

    commands = workflow_dispatch.build_engine_commands(spec, rc, "/tmp/spec.json")

    # LLM benchmarks run run_llm_bench.py in the current interpreter
    # (venv_type=None); the launcher re-execs into its tool venv. argv[0] is the
    # launcher script (VenvCommand prepends python).
    assert len(commands) == 1
    command = commands[0]
    assert command.venv_type is None
    assert "run_llm_bench.py" in command.argv[0]
    assert command.argv[command.argv.index("--server-url") + 1] == (
        "https://console.example.com"
    )


def _patch_eval_configs(monkeypatch, *, agentic):
    from workflows.workflow_types import WorkflowVenvType
    import reference_config.evals.eval_config as eval_config

    venv = WorkflowVenvType.EVALS_AGENTIC if agentic else WorkflowVenvType.EVALS_COMMON
    cfg = SimpleNamespace(tasks=[SimpleNamespace(workflow_venv_type=venv)])
    monkeypatch.setattr(
        eval_config, "EVAL_CONFIGS", {"Llama-3.1-8B-Instruct": cfg}, raising=False
    )


def test_llm_release_includes_agentic_true_when_agentic_task(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=True)
    assert workflow_dispatch._llm_release_includes_agentic(_spec(ModelType.LLM)) is True


def test_llm_release_includes_agentic_false_without_agentic_task(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=False)
    assert (
        workflow_dispatch._llm_release_includes_agentic(_spec(ModelType.LLM)) is False
    )


def test_llm_release_includes_agentic_false_for_non_llm(monkeypatch):
    _patch_eval_configs(monkeypatch, agentic=True)
    assert (
        workflow_dispatch._llm_release_includes_agentic(_spec(ModelType.IMAGE)) is False
    )


class _FakeRunner:
    """Captures the command list a WorkflowRunner would execute, returns rc=0."""

    captured: list = []

    def __init__(self, commands):
        type(self).captured = list(commands)

    def run(self):
        return 0


def _patch_engine_dispatch(monkeypatch, tmp_path):
    """Stub the dispatch side effects and capture the built command(s).

    VenvCommand provisions venvs only on execute(), which the fake WorkflowRunner
    never calls, so no venv setup needs stubbing.
    """
    monkeypatch.setattr(workflow_dispatch, "WorkflowRunner", _FakeRunner)
    monkeypatch.setattr(
        workflow_dispatch, "get_default_workflow_root_log_dir", lambda: tmp_path
    )
    monkeypatch.setattr(workflow_dispatch, "ensure_readwriteable_dir", lambda p: None)


def test_release_dispatches_only_engine(monkeypatch, tmp_path):
    from workflows.workflow_types import WorkflowVenvType

    spec, rc = _spec(ModelType.LLM), _rc(workflow="release")
    _patch_engine_dispatch(monkeypatch, tmp_path)

    results = workflow_dispatch.dispatch_workflows(
        spec, rc, str(tmp_path / "spec.json")
    )

    # Agentic now runs in-process inside the release engine; no extra
    # subprocess for agentic, tests, or report merging.
    assert [r.workflow_name for r in results] == ["release"]
    assert all(r.return_code == 0 for r in results)
    # The generic path builds a single VenvCommand for run_workflows.py in the
    # WORKFLOW_RUN_SCRIPT venv, driven by the WorkflowRunner.
    assert len(_FakeRunner.captured) == 1
    engine_cmd = _FakeRunner.captured[0]
    assert engine_cmd.venv_type == WorkflowVenvType.WORKFLOW_RUN_SCRIPT
    assert any("run_workflows.py" in a for a in engine_cmd.argv)
    assert not any("run_agentic.py" in a for a in engine_cmd.argv)
    assert not any("run_release_merge.py" in a for a in engine_cmd.argv)


def test_release_forwards_prefix_cache_and_spec_decode_flags(monkeypatch, tmp_path):
    spec = _spec(ModelType.LLM)
    rc = _rc(
        workflow="release",
        prefix_cache=True,
        prefix_cache_preset="ci",
        prefix_cache_scenarios="multi_turn",
        prefix_cache_arrival="poisson",
        prefix_cache_request_rate=2.0,
        prefix_cache_metrics_url=[
            "blaze-a29-server-ngrok.n.cloud.tenstorrent.com/metrics"
        ],
        spec_decode=True,
        spec_decode_preset="ci",
        spec_decode_warmup_requests=2,
    )
    _patch_engine_dispatch(monkeypatch, tmp_path)

    workflow_dispatch.dispatch_workflows(spec, rc, str(tmp_path / "spec.json"))

    assert len(_FakeRunner.captured) == 1
    argv = _FakeRunner.captured[0].argv
    assert "--prefix-cache" in argv
    assert argv[argv.index("--prefix-cache-preset") + 1] == "ci"
    assert argv[argv.index("--prefix-cache-scenarios") + 1] == "multi_turn"
    assert argv[argv.index("--prefix-cache-arrival") + 1] == "poisson"
    assert argv[argv.index("--prefix-cache-request-rate") + 1] == "2.0"
    assert (
        argv[argv.index("--prefix-cache-metrics-url") + 1]
        == "blaze-a29-server-ngrok.n.cloud.tenstorrent.com/metrics"
    )
    assert "--spec-decode" in argv
    assert argv[argv.index("--spec-decode-preset") + 1] == "ci"
    assert argv[argv.index("--spec-decode-warmup-requests") + 1] == "2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

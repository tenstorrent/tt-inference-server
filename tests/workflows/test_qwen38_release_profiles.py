"""Host-only release builder checks: no server, requests or device access."""
from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from llm_module.benchmark_configs import get_llm_configs
from tests.workflows.test_qwen38_minimal_release import _qwen38_spec
from workflow_module import command_factory
from workflows.model_spec import resolve_model_spec


def _spec():
    return resolve_model_spec([_qwen38_spec()], model="Qwen/Qwen3.8-27B",
                              device="p300x2", impl="qwen38-autoport", catalog_name="Shield")


def test_release_profiles_and_ordinary_sweep_are_separate():
    spec = _spec()
    ordinary = get_llm_configs(spec, spec.device_type)
    selected = get_llm_configs(spec, spec.device_type, release_scope=True)
    assert len(ordinary) == 23
    assert max(c.isl for c in ordinary) == 131072
    assert [(c.isl, c.osl, c.max_concurrency, c.num_prompts) for c in selected] == [
        (4096, 128, 1, 8), (4096, 128, 8, 24),
        (4096, 128, 16, 48), (4096, 128, 32, 96),
    ]
    assert all(not c.targets and not c.token_timing for c in selected)
    assert get_llm_configs(spec, spec.device_type) == ordinary


def test_release_scope_is_not_applied_to_another_implementation():
    spec = _spec()
    spec = replace(spec, impl=replace(spec.impl, impl_name="another-implementation"))
    assert get_llm_configs(spec, spec.device_type, release_scope=True) == get_llm_configs(spec, spec.device_type)


def test_factory_and_runner_forward_release_scope(monkeypatch):
    from test_module.llm_tests import llm_benchmark_tests as runner
    spec = _spec()
    ctx = SimpleNamespace(model_spec=spec, device=spec.device_type, runtime_config=None)
    captured = Mock(return_value=object())
    monkeypatch.setattr(runner, "run_llm_performance", captured)
    monkeypatch.setattr(command_factory, "_release_bench_venv_python", lambda args: None)
    for workflow, count in (("release", 4), ("benchmarks", 23)):
        options = command_factory._build_llm_bench_options(Namespace(
            workflow=workflow, tools="vllm", jwt_secret=None,
            prefix_cache=False, spec_decode=False))
        assert options.release_scope is (workflow == "release")
        runner.run_llm_bench(ctx, release_scope=options.release_scope)
        assert len(captured.call_args.kwargs["configs"]) == count

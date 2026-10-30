# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Goal-12 route and release-floor contracts; no device work."""
import copy
import json

import pytest

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode
from reference_config.evals.eval_config import _eval_config_map


def test_default_route_is_stage11_autoport():
    templates = load_templates_from_yaml(get_repo_root_path() / 'workflows/model_specs/dev/llm.yaml')
    specs = [s for t in templates for s in t.expand_to_specs() if s.hf_model_repo == 'Qwen/Qwen3.8-27B']
    spec = resolve_model_spec(specs, model='Qwen/Qwen3.8-27B', device='p300x2', impl=None, catalog_name='Shield')
    assert spec.impl.code_path == 'models/autoports/qwen_qwen3_8_27b'
    d = spec.device_model_spec
    assert d.max_concurrency == 32
    assert int(d.vllm_args['max_num_seqs']) == 32
    assert int(d.vllm_args['block_size']) == 2048
    assert int(d.vllm_args['mamba_block_size']) == 2048
    assert int(d.vllm_args['num_gpu_blocks_override']) == 385
    assert d.vllm_args['enable_prefix_caching'] is True
    assert int(d.vllm_args['shutdown_timeout']) == 30
    assert d.env_vars['TT_MODEL_CLASS_OVERRIDES'].endswith('tt.generator_vllm:Qwen38ForCausalLM')


@pytest.fixture(params=[
    ("terminal_bench_2_1", "terminal-bench/terminal-bench-2-1", "terminus-2",
     ["terminal-bench/fix-git", "terminal-bench/break-filter-js-from-html"], 16384, None),
    ("swe_bench_verified", "swebench-verified", "mini-swe-agent",
     ["django__django-11299", "astropy__astropy-14096"], 32768, None),
    ("swe_bench_verified", "swebench-verified", "mini-swe-agent",
     ["django__django-11299", "astropy__astropy-14096"], 32768, 7200),
], ids=["terminal-bench", "swe-bench-default", "swe-bench-7200"])
def smoke_arm(request, tmp_path, monkeypatch):
    from reference_config.evals import qwen38_goal12

    task_name, dataset, agent, names, cap, timeout = request.param
    controls = qwen38_goal12.controls()
    controls.pop("request_timeout_sec", None)
    if timeout is not None:
        controls["request_timeout_sec"] = timeout
    controls.update(evals=[task_name], dataset=dataset, dataset_ref=None,
                    task_names=names, sample_count=2, max_concurrent=32,
                    intent="isolated_smoke", scored=False)
    # Exercise the actual JSON controls loader, independent of which arm is
    # currently published, rather than replacing selection with a test stub.
    control_module = tmp_path / "qwen38_goal12.py"
    control_module.with_suffix(".json").write_text(json.dumps(controls))
    monkeypatch.setattr(qwen38_goal12, "__file__", str(control_module))
    assert qwen38_goal12.controls() == controls
    return task_name, dataset, agent, names, cap, timeout


def test_smoke_retains_release_values_and_dataset_identity(smoke_arm):
    from reference_config.evals.qwen38_goal12 import configure

    name, dataset, agent, names, cap, timeout = smoke_arm
    original = _eval_config_map['Qwen/Qwen3.8-27B']
    snapshot = copy.deepcopy(original)
    release = next(t for t in original.tasks if t.task_name == name)
    tasks = configure(original, EvalLimitMode).tasks
    assert [t.task_name for t in tasks] == [name]
    cfg = tasks[0].agentic_eval_config
    base = release.agentic_eval_config
    assert cfg.dataset == base.dataset == dataset
    assert cfg.agent == base.agent == agent
    assert cfg.task_names_map[EvalLimitMode.CI_NIGHTLY] == names
    assert cfg.task_names_map[EvalLimitMode.SMOKE_TEST] == names
    assert cfg.n_tasks == 2
    assert cfg.agent_timeout_sec == base.agent_timeout_sec == 21600
    expected_kwargs = copy.deepcopy(base.agent_kwargs)
    if timeout is not None:
        expected_kwargs.setdefault("config", {}).setdefault("model", {}).setdefault("model_kwargs", {})["timeout"] = timeout
    assert cfg.agent_kwargs == expected_kwargs
    if name == "terminal_bench_2_1":
        assert cfg.agent_kwargs['llm_kwargs']['timeout'] == 3600
        assert cfg.agent_kwargs['llm_kwargs']['max_tokens'] == cap
    else:
        assert base.llm_timeout_sec == 3600
        assert cfg.llm_timeout_sec == (3600 if timeout is None else timeout)
        assert cfg.llm_timeout_sec >= base.llm_timeout_sec
        assert cfg.agent_kwargs['max_tokens'] == cap
    assert cfg.n_concurrent_trials == 32
    assert cfg.harbor_timeout_sec is None
    assert tasks[0].score == release.score
    assert tasks[0].priority == release.priority == "must"
    assert original == snapshot


def test_smoke_selection_is_model_scoped_and_keeps_release_config(smoke_arm):
    from types import SimpleNamespace
    from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks

    name, _, _, _, _, _ = smoke_arm
    original = _eval_config_map['Qwen/Qwen3.8-27B']
    snapshot = copy.deepcopy(original)
    ctx = SimpleNamespace(all_params=SimpleNamespace(tasks=original.tasks),
                          runtime_config=None, model_spec=SimpleNamespace(impl=SimpleNamespace(impl_id='qwen38_autoport')))
    selected = _select_agentic_tasks(ctx)
    assert [task.task_name for task in selected] == [name]
    assert len(original.tasks) == 3
    assert original.tasks[1].agentic_eval_config.n_tasks == 89
    ctx.model_spec.impl.impl_id = 'qwen38_27b_qb2'
    assert [task.task_name for task in _select_agentic_tasks(ctx)] == [
        "terminal_bench_2_1", "swe_bench_verified"]
    assert original == snapshot

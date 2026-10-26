# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import json
from types import SimpleNamespace

import pytest

from llm_module.agentic.harbor import _write_harbor_config
from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers.agentic import build_harbor_config
from reference_config.evals.eval_config import (
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode

MODEL = "ibm-granite/granite-4.2-30b"


def test_standard_plugin_resolves_granite_without_a_custom_implementation():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    specs = [
        spec
        for template in templates
        if template.weights == [MODEL]
        for spec in template.expand_to_specs()
    ]
    spec = resolve_model_spec(
        specs,
        model=MODEL,
        device="p300x2",
        impl="tt-vllm-plugin",
        catalog_name="Shield",
    )
    assert spec.impl.impl_name == "tt-vllm-plugin"
    device = spec.device_model_spec
    assert device.max_context == 131072
    assert device.max_tokens_all_users == 655360
    assert device.vllm_args["enable-prefix-caching"] is True
    assert "no-enable-prefix-caching" not in device.vllm_args
    assert device.vllm_args["enable-prompt-tokens-details"] is True
    assert json.loads(device.vllm_args["default-chat-template-kwargs"]) == {
        "enable_thinking": True,
        "low_effort": False,
    }
    assert spec.metadata.get("spec_tests_skip_reason") is None


@pytest.mark.parametrize("index", [1])
def test_standard_harbor_config_preserves_cohort_and_enables_thinking(tmp_path, index):
    task = _eval_config_map[MODEL].tasks[index]
    config = build_harbor_config(
        task,
        ServerConnection(base_url="http://localhost", service_port=8000, model=MODEL),
        DriverContext(output_dir=tmp_path),
        runtime_config=SimpleNamespace(limit_samples_mode="ci-nightly"),
        n_tasks=5,
    )
    generated = json.loads(_write_harbor_config(config).read_text())
    assert generated["datasets"][0]["task_names"] == [
        "django__django-11299",
        "astropy__astropy-14096",
        "matplotlib__matplotlib-25332",
        "sympy__sympy-13551",
        "scikit-learn__scikit-learn-14629",
    ]
    assert generated["datasets"][0]["n_tasks"] == 5
    assert generated["n_concurrent_trials"] == 5
    assert generated["agents"][0]["override_timeout_sec"] == 6 * 60 * 60
    agent = generated["agents"][0]
    kwargs = agent["kwargs"]
    assert kwargs["max_tokens"] == 32768
    holder = kwargs["config"]["model"]["model_kwargs"]
    assert holder["timeout"] == 3600
    assert holder["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "low_effort": False,
    }
    assert "UV_OVERRIDE" in agent["env"]
    assert kwargs["config"]["environment"]["env"]["PATH"].startswith(
        "/opt/miniconda3/envs/testbed/bin:"
    )


@pytest.mark.parametrize("index,minimum,total", [(0, 6, 10), (1, 2, 5)])
def test_ci_gates_without_claiming_gpu_controls(index, minimum, total):
    task = _eval_config_map[MODEL].tasks[index]
    assert task.score.gpu_reference_score is None
    reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
    assert accept_eval_score(reference, minimum * 100 / total, n_total=total)
    assert not accept_eval_score(reference, (minimum - 1) * 100 / total, n_total=total)


def test_granite_gpqa_matches_other_dense_reasoning_models():
    gpqa = _eval_config_map[MODEL].tasks[0]
    assert gpqa.task_name == "r1_gpqa_diamond"
    assert gpqa.max_concurrent == 10
    assert gpqa.model_kwargs["timeout"] == 7200
    assert gpqa.use_chat_api and gpqa.thinking
    assert gpqa.score.score_func_kwargs["result_keys"] == ["exact_match,none"]
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 10
    for model in ("google/gemma-4-31B-it", "Qwen/Qwen3.8-27B"):
        peer = _eval_config_map[model].tasks[0]
        assert peer.task_name == gpqa.task_name
        assert (
            peer.score.score_func_kwargs["result_keys"]
            == gpqa.score.score_func_kwargs["result_keys"]
        )


def test_release_runs_only_gpqa_and_swe():
    assert [task.task_name for task in _eval_config_map[MODEL].tasks] == [
        "r1_gpqa_diamond",
        "swe_bench_verified",
    ]
    swe = _eval_config_map[MODEL].tasks[1]
    assert swe.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 5
    assert swe.limit_samples_map[EvalLimitMode.SMOKE_TEST] == 5

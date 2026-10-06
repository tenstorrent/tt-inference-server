# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.benchmark_configs import get_llm_configs
from reference_config.evals.eval_config import (
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode, ModelStatusTypes


MODEL = "ibm-granite/granite-4.2-30b"


def test_granite_release_resolves_demo_and_preserves_capacity_and_benchmarks():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    template = next(t for t in templates if t.weights == [MODEL])
    spec = resolve_model_spec(
        template.expand_to_specs(),
        model=MODEL,
        device="p300x2",
        impl="granite42-30b-qb2",
        catalog_name="Shield",
    )
    device = spec.device_model_spec
    assert spec.status == ModelStatusTypes.FUNCTIONAL
    assert spec.impl.code_path == "models/demos/granite42_30b_qb2"
    assert device.max_concurrency == 16
    assert device.max_context == device.max_tokens_all_users == 131072
    assert device.env_vars["EXTRA_MODELS_DIR"] == "../../tt-metal/models/demos"
    assert "TT_MODEL_CLASS_OVERRIDES" not in device.env_vars
    assert "GRANITE_VLLM_HOST_COMPAT" not in device.env_vars
    assert device.vllm_args["revision"] == "9e668ce1c538387ef24d3644e9b0606647762636"
    assert device.vllm_args["tokenizer_revision"] == device.vllm_args["revision"]
    configs = get_llm_configs(spec, spec.device_type)
    assert {(4096, 128, c) for c in (1, 8, 16)} <= {
        (c.isl, c.osl, c.max_concurrency) for c in configs
    }
    assert max(c.isl for c in configs) >= 65536
    assert [(c.isl, c.osl, c.max_concurrency) for c in configs if c.targets] == [
        (128, 128, 1)
    ]


def test_granite_ten_case_evals_preserve_generation_and_published_targets():
    gpqa, terminal, swe = _eval_config_map[MODEL].tasks
    assert gpqa.max_concurrent == 10
    assert gpqa.seed == 42
    assert gpqa.gen_kwargs["max_gen_toks"] == 32768
    assert gpqa.gen_kwargs["temperature"] == 1.0
    assert gpqa.gen_kwargs["top_p"] == 0.95
    assert gpqa.gen_kwargs["chat_template_kwargs"] == {
        "enable_thinking": True,
        "low_effort": False,
    }
    assert gpqa.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 10
    assert gpqa.score.published_score == 66.41
    reference = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert accept_eval_score(reference, 0.0, n_total=10) is None
    for task, passing, failing in ((terminal, 30.0, 20.0), (swe, 60.0, 50.0)):
        config = task.agentic_eval_config
        names = config.task_names_map[EvalLimitMode.CI_NIGHTLY]
        assert len(names) == len(set(names)) == 10
        assert config.n_concurrent_trials == 10
        assert config.agent_timeout_sec == 7200
        assert config.llm_timeout_sec == 3600
        reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert reference["tolerance"] == 0.0
        assert reference["is_subset_reference"] is False
        assert "not a matched subset GPU control" in reference["reference_ref"]
        assert accept_eval_score(reference, passing, n_total=10) is True
        assert accept_eval_score(reference, failing, n_total=10) is False

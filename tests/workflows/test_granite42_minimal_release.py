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
    assert "not qualified" in spec.metadata["spec_tests_skip_reason"]
    assert spec.status == ModelStatusTypes.FUNCTIONAL
    assert spec.impl.code_path == "models/demos/granite42_30b_qb2"
    assert device.max_concurrency == 16
    assert device.max_context == 131072
    assert device.max_tokens_all_users == 327680
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


def test_granite_fixed_subset_evals_preserve_generation_and_published_targets():
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
    for task, passing, failing in (
        (gpqa, 60.0, 50.0),
        (terminal, 20.0, 10.0),
        (swe, 50.0, 40.0),
    ):
        assert task.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 10
        reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert reference["tolerance"] == 0.0
        assert reference["is_subset_reference"] is True
        assert "not measured subset GPU control" in reference["reference_ref"]
        assert accept_eval_score(reference, passing, n_total=10) is True
        assert accept_eval_score(reference, failing, n_total=10) is False
        full_reference = resolve_eval_reference(task.score, None)
        assert full_reference["reference_score"] == task.score.published_score
        assert full_reference["is_subset_reference"] is False
    for task in (terminal, swe):
        config = task.agentic_eval_config
        names = config.task_names_map[EvalLimitMode.CI_NIGHTLY]
        expected_cases = 10
        assert len(names) == len(set(names)) == expected_cases
        assert config.n_concurrent_trials == 10
        assert config.agent_timeout_sec == 7200
        assert config.llm_timeout_sec == 3600
    assert terminal.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/break-filter-js-from-html",
        "terminal-bench/cobol-modernization",
        "terminal-bench/compile-compcert",
        "terminal-bench/feal-differential-cryptanalysis",
        "terminal-bench/qemu-startup",
        "terminal-bench/caffe-cifar-10",
        "terminal-bench/password-recovery",
        "terminal-bench/portfolio-optimization",
        "terminal-bench/hf-model-inference",
        "terminal-bench/financial-document-processor",
    ]
    assert swe.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "django__django-11299",
        "astropy__astropy-14096",
        "matplotlib__matplotlib-25332",
        "sympy__sympy-13551",
        "scikit-learn__scikit-learn-14629",
        "django__django-15098",
        "sphinx-doc__sphinx-8593",
        "sympy__sympy-13852",
        "pydata__xarray-3095",
        "django__django-15695",
    ]


def test_granite_ci_commands_select_fixed_subsets():
    from types import SimpleNamespace

    from llm_module.drivers.agentic import resolve_n_tasks, resolve_task_names
    from llm_module.eval_command import build_eval_command

    runtime = SimpleNamespace(limit_samples_mode="ci_nightly")
    gpqa, terminal, swe = _eval_config_map[MODEL].tasks
    spec = SimpleNamespace(
        model_id="granite-4.2-30b",
        model_name="granite-4.2-30b",
        hf_model_repo=MODEL,
        device_model_spec=SimpleNamespace(
            max_context=131072,
            max_concurrency=16,
            eval_max_retries=0,
        ),
    )
    command = build_eval_command(
        gpqa,
        spec,
        "p300x2",
        "/tmp/granite-evals",
        8000,
        runtime_config=runtime,
    )
    assert command[command.index("--limit") + 1] == "10"
    assert "num_concurrent=10" in command[command.index("--model_args") + 1]
    for task in (terminal, swe):
        expected_cases = 10
        assert resolve_n_tasks(task, runtime) == expected_cases
        assert len(resolve_task_names(task, runtime)) == expected_cases


def test_agentic_report_keeps_effective_ci_reference_after_consolidation():
    from llm_module.parsers.agentic import AgenticEvalParser
    from report_module.generator import _consolidate_eval_blocks
    from report_module.renderers import render_evals
    from workflow_module.engine_types import ReportCheckTypes

    blocks = []
    for task, correct, n_trials, reference in (
        (_eval_config_map[MODEL].tasks[1], 2, 10, 20.0),
        (_eval_config_map[MODEL].tasks[2], 5, 10, 50.0),
    ):
        raw = {
            "stats": {
                "evals": {
                    task.task_name: {
                        "metrics": [{"mean": correct / n_trials}],
                        "n_trials": n_trials,
                    }
                }
            }
        }
        block = AgenticEvalParser(
            task_name=task.task_name,
            score=task.score,
            limit_mode=EvalLimitMode.CI_NIGHTLY,
        ).parse(raw)
        assert block.data["accuracy_check"] == ReportCheckTypes.PASS
        assert block.data["gpu_reference_score"] == reference
        assert (
            block.data["ratio_to_reference"] == (correct * 100 / n_trials) / reference
        )
        assert block.data["n_samples"] == n_trials
        blocks.append(block)
    (merged,) = _consolidate_eval_blocks(blocks)
    markdown = render_evals(merged, {})
    assert "not measured subset GPU control" in markdown
    assert "Reference Source" in markdown
    assert "GPU Reference Score" not in markdown
    assert "round the required correct count down" in markdown
    assert "equivalent to the GPU" not in markdown


def test_agentic_failure_keeps_ci_reference_and_remains_a_failure():
    from llm_module.parsers.agentic import AgenticEvalParser
    from workflow_module.engine_types import ReportCheckTypes

    task = _eval_config_map[MODEL].tasks[2]
    block = AgenticEvalParser(
        task_name=task.task_name,
        score=task.score,
        limit_mode=EvalLimitMode.CI_NIGHTLY,
    ).failure_block(return_code=7)
    assert block.data["gpu_reference_score"] == 50.0
    assert "not measured subset GPU control" in block.data["gpu_reference_score_ref"]
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL
    assert block.data["score"] is None
    assert block.data["success"] is False
    assert block.data["subprocess_rc"] == 7


def test_granite_direct_agentic_candidate_keeps_gpqa_reasoning():
    gpqa, terminal, swe = _eval_config_map[MODEL].tasks
    assert gpqa.gen_kwargs["chat_template_kwargs"] == {
        "enable_thinking": True,
        "low_effort": False,
    }
    assert terminal.agentic_eval_config.agent_kwargs["llm_kwargs"] == {
        "top_p": 0.95,
        "timeout": 3600,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False, "low_effort": False}
        },
    }
    kwargs = swe.agentic_eval_config.agent_kwargs["config"]["model"]["model_kwargs"]
    assert (
        swe.agentic_eval_config.agent_kwargs["config"]["environment"]["env"][
            "PATH"
        ].split(":")[0]
        == "/opt/miniconda3/envs/testbed/bin"
    )
    assert kwargs == {
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False, "low_effort": False}
        },
    }

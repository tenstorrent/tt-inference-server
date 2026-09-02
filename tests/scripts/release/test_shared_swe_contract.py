# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""One SWE gate contract for every model.

The generic gate loads every model-specific fact from the model's catalogue
entry (``reference_config.evals.eval_config``); the gate script itself carries
zero model branches. These tests are the contract:

* no per-model prompt-template overrides (``swebench.py`` rejects them),
* token budgets derive from the declared ``max_context`` via
  ``resolve_token_budget``,
* gate runs are local-behavioral and stay score-NA until CS supplies
  thresholds,
* the gate config has one shape for every model.
"""

from dataclasses import fields
from pathlib import Path

import pytest

from llm_module.parsers.agentic import compute_accuracy_check
from reference_config.evals.eval_config import EVAL_CONFIGS
from scripts.release.run_local_swe_gate import (
    build_gate_config,
    resolve_token_budget,
)
from workflows.workflow_types import EvalLimitMode, ReportCheckTypes

MODELS = ["gpt-oss-120b", "Qwen3.6-27B", "gemma-4-31B-it"]

GATE_SOURCE = (
    Path(__file__).resolve().parents[3] / "scripts/release/run_local_swe_gate.py"
)


def _gate(model):
    return build_gate_config(
        model, max_context=32768, max_output_tokens=2048, step_limit=16
    )


def _swe_task(model):
    config = EVAL_CONFIGS[model]
    return next(
        task for task in config.tasks if task.task_name == "swe_bench_verified"
    )


@pytest.mark.parametrize("model", MODELS)
def test_no_prompt_template_overrides(model):
    cfg = _gate(model)
    assert "instance_template" not in cfg.mini_agent_kwargs
    assert "system_template" not in cfg.mini_agent_kwargs


@pytest.mark.parametrize("model", MODELS)
def test_budget_derived_from_context(model):
    ctx, inp, out = resolve_token_budget(32768, 2048)
    assert (ctx, inp, out) == (32768, 30720, 2048)
    cfg = _gate(model)
    assert cfg.max_context == 32768
    assert cfg.max_input_tokens == 30720
    assert cfg.max_output_tokens == 2048
    assert cfg.max_input_tokens + cfg.max_output_tokens == cfg.max_context


@pytest.mark.parametrize(
    "context,output", [(0, 1), (-1, 1), (4096, 0), (4096, -1), (4096, 4096)]
)
def test_resolve_token_budget_rejects_invalid_envelopes(context, output):
    with pytest.raises(ValueError):
        resolve_token_budget(context, output)


@pytest.mark.parametrize("model", MODELS)
def test_score_integrity_stays_na(model):
    cfg = _gate(model)
    assert cfg.published_score is None
    assert cfg.gpu_reference_score is None
    assert cfg.qualification_claim == "local_behavioral_only"


@pytest.mark.parametrize("model", ["gpt-oss-120b", "Qwen3.6-27B"])
def test_bounded_catalogue_rows_stay_report_only_until_cs_sets_reference(model):
    # Folded from the retired per-model contract tests: the bounded catalogue
    # rows carry no invented reference, and any accuracy therefore reports NA
    # rather than a pass/fail verdict.
    score = _swe_task(model).score
    assert score.published_score is None
    assert score.gpu_reference_score is None
    for accuracy in (100.0, 0.0):
        assert (
            compute_accuracy_check(
                {"accuracy": accuracy}, score, EvalLimitMode.CI_NIGHTLY
            )
            == ReportCheckTypes.NA
        )


def test_gate_config_identical_shape_across_models():
    shapes = {
        m: sorted(field.name for field in fields(_gate(m))) for m in MODELS
    }
    assert len({tuple(v) for v in shapes.values()}) == 1


@pytest.mark.parametrize("model", MODELS)
def test_model_owned_config_comes_from_the_catalogue(model):
    cfg = _gate(model)
    catalogue = _swe_task(model).swebench_eval_config
    assert cfg.completion_kwargs == catalogue.completion_kwargs
    assert cfg.temperature == catalogue.temperature
    assert cfg.top_p == catalogue.top_p
    assert cfg.hf_model_repo == EVAL_CONFIGS[model].hf_model_repo


@pytest.mark.parametrize("model", MODELS)
def test_step_limit_is_caller_owned(model):
    assert _gate(model).mini_agent_kwargs == {"step_limit": 16}
    other = build_gate_config(
        model, max_context=32768, max_output_tokens=2048, step_limit=12
    )
    assert other.mini_agent_kwargs == {"step_limit": 12}


def test_unknown_model_fails_closed():
    with pytest.raises(ValueError):
        build_gate_config(
            "not-a-model", max_context=32768, max_output_tokens=2048, step_limit=16
        )


def test_gate_script_has_no_model_branches():
    source = GATE_SOURCE.read_text()
    assert "if model ==" not in source
    assert "if args.model ==" not in source
    assert "instance_template" not in source
    assert "system_template" not in source


# --- planner/catalogue consistency: a SWE row must be satisfiable by the ---
# --- artifact its catalogue template declares. -----------------------------

REPO = Path(__file__).resolve().parents[3]
DEV_CATALOGUE = REPO / "workflows/model_specs/dev/llm.yaml"


def _quetzal_swe_pairs():
    """Yield (model_repo, spec, swe_task) for every dev quetzal template
    whose model declares a swe_bench_verified row."""
    from reference_config.evals.eval_config import _eval_config_map
    from workflows.model_spec import load_templates_from_yaml

    for template in load_templates_from_yaml(DEV_CATALOGUE):
        if template.impl.impl_id != "quetzal":
            continue
        model_repo = template.weights[0]
        eval_config = _eval_config_map.get(model_repo)
        if eval_config is None:
            continue
        swe_tasks = [
            task
            for task in eval_config.tasks
            if task.task_name == "swe_bench_verified"
        ]
        if not swe_tasks:
            continue
        yield model_repo, template.expand_to_specs()[0], swe_tasks[0]


def test_swe_rows_fit_their_declared_prefill_buckets():
    checked = 0
    for model_repo, spec, task in _quetzal_swe_pairs():
        cfg = task.swebench_eval_config
        max_context = spec.device_model_spec.max_context
        if task.min_context_required and task.min_context_required > max_context:
            # The row refuses to launch on this artifact (min-context gate);
            # it cannot violate the bucket contract it never reaches.
            continue
        # An admissible row's complete envelope must fit the serving contract.
        assert cfg.max_input_tokens + cfg.max_output_tokens <= max_context, (
            f"{model_repo}: SWE envelope {cfg.max_input_tokens}+"
            f"{cfg.max_output_tokens} exceeds the artifact's {max_context} context"
        )
        declared = spec.env_vars.get("QUETZAL_REQUIRED_PREFILL_BUCKETS")
        if declared is None:
            # No one-shot bucket contract is pinned in the template; the
            # launch planner still refuses at run time from the endpoint's
            # capabilities receipt (plan_agentic_external_run).
            continue
        buckets = sorted(int(value) for value in declared.split(","))
        checked += 1
        assert cfg.max_input_tokens <= buckets[-1], (
            f"{model_repo}: SWE max_input_tokens {cfg.max_input_tokens} does "
            f"not fit largest declared one-shot prefill bucket {buckets[-1]}"
        )
    assert checked >= 1, "no declared-bucket quetzal SWE row was checked"


def test_gpt_f0b_planner_and_eval_row_agree():
    from reference_config.evals.eval_config import _eval_config_map
    from scripts.release import plan_gpt120_f0b_quetzal_enrollment as planner

    task = next(
        task
        for task in _eval_config_map["openai/gpt-oss-120b"].tasks
        if task.task_name == "swe_bench_verified"
    )
    cfg = task.swebench_eval_config

    # One authoritative bucket set: the planner names it once and both its
    # publication validator and rendered env derive from it.
    buckets = planner.GPT_PREFILL_BUCKETS
    assert buckets == (8192,)
    assert list(planner.EXPECTED_ARTIFACTS_PREFILL_BUCKETS) == sorted(buckets)

    # The eval row's bounded input derives from that bucket set: the complete
    # rendered input must fit the largest one-shot bucket, and the envelope
    # must fit the declared minimum context.
    assert cfg.max_input_tokens == planner.GPT_SWEBENCH_INPUT_TOKENS
    assert cfg.max_output_tokens == planner.GPT_SWEBENCH_OUTPUT_TOKENS
    assert task.min_context_required == planner.GPT_SWEBENCH_MIN_CONTEXT
    assert cfg.max_input_tokens <= max(buckets)
    assert (
        cfg.max_input_tokens + cfg.max_output_tokens <= task.min_context_required
    )

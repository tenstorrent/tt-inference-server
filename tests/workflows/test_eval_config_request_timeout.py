# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Capacity-derived lm-eval per-request timeout for Quetzal long-generation
tasks (reference_config.evals.eval_config.resolve_capacity_request_timeout)."""

from types import SimpleNamespace

import pytest

from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import (
    DEFAULT_DECODE_FLOOR_TOK_S_PER_USER,
    LM_EVAL_DEFAULT_REQUEST_TIMEOUT_S,
    QWQ_R1_REQUEST_TIMEOUT_S,
    R1_TASK_DEFAULT_MAX_GEN_TOKS,
    EvalTask,
    _eval_config_map,
    capacity_request_timeout_s,
    resolve_capacity_request_timeout,
)
from workflows.utils_report import PerformanceTarget
from workflows.workflow_types import WorkflowVenvType


def _spec(
    *,
    impl_id="quetzal",
    device="P300X2",
    max_context=65536,
    max_concurrency=32,
    kv_tokens=80000,
    perf_reference=(),
):
    return SimpleNamespace(
        model_id="m",
        model_name="m",
        hf_model_repo="org/m",
        impl=SimpleNamespace(impl_id=impl_id),
        device_model_spec=SimpleNamespace(
            device=SimpleNamespace(name=device),
            max_context=max_context,
            max_concurrency=max_concurrency,
            max_tokens_all_users=kv_tokens,
            override_tt_config={"kv_cache_usable_tokens": kv_tokens},
            perf_reference=list(perf_reference),
            eval_max_retries=None,
        ),
    )


def _r1(task_name="r1_gpqa_diamond", **kwargs):
    kwargs.setdefault("gen_kwargs", {"stream": "false", "max_gen_toks": "32768"})
    kwargs.setdefault("model_kwargs", {"max_length": 65536})
    return EvalTask(task_name=task_name, **kwargs)


def _model_args(command):
    raw = command[command.index("--model_args") + 1]
    return dict(item.split("=", 1) for item in raw.split(","))


def _gen_kwargs(command):
    raw = command[command.index("--gen_kwargs") + 1]
    return dict(item.split("=", 1) for item in raw.split(","))


# --- the formula --------------------------------------------------------------

# (max_context, max_concurrency, KV pool tokens) from each row's P300X2 device
# model spec. All four are oversubscribed (fewer full-context KV slots than
# concurrent requests), so each gets one full-length 32768-token generation at
# the 4.5 tok/s floor (7282 s), doubled for queue wait, rounded up to 30 min.
ROWS = {
    # Hand-set 4 h on nkapre/qb2-wave7 (ee9413b1); that commit's own
    # derivation was 14564 s, above the 14400 s it rounded to.
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (65536, 32, 80000),
    # Hand-set QWQ_R1_REQUEST_TIMEOUT_S = 3 h.
    "Qwen/QwQ-32B": (40960, 32, 149952),
    "llm-jp/llm-jp-4-33b-thinking": (65536, 32, 140544),
    # Hand-set 4 h in its EvalConfig.
    "IFM/K2-Horizon-MoVA-36B-A4B": (131072, 32, 131072),
}


@pytest.mark.parametrize("repo", sorted(ROWS))
def test_reasoning_rows_get_a_queue_covering_full_length_timeout(repo):
    max_context, concurrency, kv = ROWS[repo]
    timeout = capacity_request_timeout_s(
        max_gen_toks=R1_TASK_DEFAULT_MAX_GEN_TOKS,
        max_context=max_context,
        max_concurrency=concurrency,
        kv_tokens=kv,
        decode_tok_s=DEFAULT_DECODE_FLOOR_TOK_S_PER_USER,
    )
    assert timeout == 16200
    assert timeout >= 2 * 32768 / 4.5
    # Never below the hand-derived per-model values it generalises.
    assert timeout >= max(QWQ_R1_REQUEST_TIMEOUT_S, 4 * 60 * 60)


def test_no_queue_factor_when_the_kv_pool_holds_every_request():
    # 4 concurrent requests x 32768 context fit a 131072-token pool.
    timeout = capacity_request_timeout_s(
        max_gen_toks=32768,
        max_context=32768,
        max_concurrency=4,
        kv_tokens=131072,
        decode_tok_s=4.5,
    )
    assert timeout == 9000  # 7282 s rounded up to a 1800 s multiple


def test_queue_factor_is_capped():
    common = dict(max_gen_toks=9000, max_context=65536, decode_tok_s=4.5)
    two_waves = capacity_request_timeout_s(max_concurrency=2, kv_tokens=65536, **common)
    many_waves = capacity_request_timeout_s(
        max_concurrency=32, kv_tokens=65536, **common
    )
    # 9000 / 4.5 = 2000 s, x2 (capped) = 4000 s -> 5400 s
    assert two_waves == many_waves == 5400


def test_never_below_the_harness_default():
    assert (
        capacity_request_timeout_s(
            max_gen_toks=16,
            max_context=4096,
            max_concurrency=1,
            kv_tokens=4096,
            decode_tok_s=100.0,
        )
        == LM_EVAL_DEFAULT_REQUEST_TIMEOUT_S
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_gen_toks", 0),
        ("max_context", -1),
        ("max_concurrency", True),
        ("kv_tokens", 1.5),
        ("decode_tok_s", 0),
    ],
)
def test_rejects_non_positive_inputs(field, value):
    kwargs = dict(
        max_gen_toks=32768,
        max_context=65536,
        max_concurrency=32,
        kv_tokens=80000,
        decode_tok_s=4.5,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        capacity_request_timeout_s(**kwargs)


# --- when it applies ----------------------------------------------------------


def test_applies_to_a_quetzal_r1_task_without_an_explicit_timeout():
    assert resolve_capacity_request_timeout(_r1(), _spec()) == 16200


def test_r1_task_without_max_gen_toks_uses_the_task_yaml_default():
    task = _r1(gen_kwargs={"stream": "false"})
    assert resolve_capacity_request_timeout(task, _spec()) == 16200


def test_explicit_timeout_always_wins():
    for explicit in ("600", "10800", "99999"):
        task = _r1(model_kwargs={"max_length": 65536, "timeout": explicit})
        assert resolve_capacity_request_timeout(task, _spec()) is None


@pytest.mark.parametrize("impl_id", ["tt_transformers", None])
def test_only_quetzal_rows(impl_id):
    assert resolve_capacity_request_timeout(_r1(), _spec(impl_id=impl_id)) is None


def test_short_generation_tasks_keep_the_harness_default():
    task = EvalTask(task_name="ifeval", gen_kwargs={"max_gen_toks": "1280"})
    assert resolve_capacity_request_timeout(task, _spec()) is None


def test_non_r1_task_with_unknown_output_budget_is_left_alone():
    assert resolve_capacity_request_timeout(EvalTask(task_name="mmlu"), _spec()) is None


def test_long_non_r1_generation_task_applies():
    task = EvalTask(task_name="aime25", gen_kwargs={"max_gen_toks": "65536"})
    spec = _spec(max_context=131072, kv_tokens=131072)
    # 65536 / 4.5 = 14564 s, x2 = 29127 s -> 30600 s
    assert resolve_capacity_request_timeout(task, spec) == 30600


def test_lmms_eval_tasks_are_left_alone():
    task = _r1(workflow_venv_type=WorkflowVenvType.EVALS_VISION)
    assert resolve_capacity_request_timeout(task, _spec()) is None


def test_missing_kv_capacity_is_left_alone():
    spec = _spec()
    spec.device_model_spec.max_tokens_all_users = None
    spec.device_model_spec.override_tt_config = {}
    assert resolve_capacity_request_timeout(_r1(), spec) is None


def test_kv_capacity_is_the_smaller_declared_budget():
    # Pool holds 2 full requests per the override but only 1 per the TT config.
    spec = _spec(max_context=32768, max_concurrency=2, kv_tokens=65536)
    spec.device_model_spec.override_tt_config = {"kv_cache_usable_tokens": 40000}
    task = _r1(gen_kwargs={"max_gen_toks": "8100"})
    # 8100 / 4.5 = 1800 s; one KV slot for 2 requests -> x2 = 3600 s.
    assert resolve_capacity_request_timeout(task, spec) == 3600


def test_effective_concurrency_is_clamped_to_the_device():
    spec = _spec(max_context=32768, max_concurrency=2, kv_tokens=65536)
    task = _r1(gen_kwargs={"max_gen_toks": "8100"})
    # 2 KV slots hold both requests the device admits: no queue factor.
    assert resolve_capacity_request_timeout(task, spec, num_concurrent=32) is None


def _perf(concurrency, tput_user):
    return SimpleNamespace(
        max_concurrency=concurrency,
        targets={"target": PerformanceTarget(tput_user=tput_user)},
    )


def test_slower_measured_rate_at_eval_concurrency_lengthens_the_timeout():
    spec = _spec(perf_reference=[_perf(32, 2.0)])
    # 32768 / 2.0 = 16384 s, x2 = 32768 s -> 34200 s
    assert resolve_capacity_request_timeout(_r1(), spec) == 34200


def test_faster_or_low_concurrency_rates_never_shorten_it():
    spec = _spec(perf_reference=[_perf(32, 40.0), _perf(1, 1.0)])
    assert resolve_capacity_request_timeout(_r1(), spec) == 16200


# --- catalog and eval command -------------------------------------------------


def test_qwq_explicit_timeout_is_kept_and_not_above_the_helper():
    spec = _spec(max_context=40960, kv_tokens=149952)
    for task in _eval_config_map["Qwen/QwQ-32B"].tasks:
        assert task.model_kwargs["timeout"] == str(QWQ_R1_REQUEST_TIMEOUT_S)
        assert resolve_capacity_request_timeout(task, spec) is None
        without = EvalTask(
            task_name=task.task_name,
            gen_kwargs=task.gen_kwargs,
            model_kwargs={"max_length": task.model_kwargs["max_length"]},
        )
        assert resolve_capacity_request_timeout(without, spec) >= (
            QWQ_R1_REQUEST_TIMEOUT_S
        )


def test_eval_command_sends_the_capacity_timeout():
    command = build_eval_command(_r1(), _spec(), "P300X2", "/tmp/evals", 8000)
    assert _model_args(command)["timeout"] == "16200"
    # Samples and output budget are untouched.
    assert _gen_kwargs(command)["max_gen_toks"] == "32768"
    assert "--limit" not in command


def test_eval_command_keeps_an_explicit_timeout():
    task = _r1(model_kwargs={"max_length": 65536, "timeout": "10800"})
    command = build_eval_command(task, _spec(), "P300X2", "/tmp/evals", 8000)
    assert _model_args(command)["timeout"] == "10800"


def test_eval_command_unchanged_for_other_impls():
    command = build_eval_command(
        _r1(), _spec(impl_id="tt_transformers"), "P300X2", "/tmp/evals", 8000
    )
    assert "timeout" not in _model_args(command)


def test_eval_command_sizes_from_the_clamped_output_budget():
    # A 4096-token deployment clamps max_gen_toks to 3008, which the default
    # covers at the floor rate (3008 / 4.5 x 2 = 1337 s): no timeout added.
    spec = _spec(max_context=4096, kv_tokens=4096)
    command = build_eval_command(_r1(), spec, "P300X2", "/tmp/evals", 8000)
    assert _gen_kwargs(command)["max_gen_toks"] == "3008"
    assert "timeout" not in _model_args(command)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Every dev catalog profile must resolve the request body of every eval task it
runs: a profile's ``eval_task_request_body_overrides`` and the task's own
``thinking`` pin may not disagree (``resolve_request_body`` raises when they do,
which otherwise surfaces only when the server boots). Two implementations of
one checkpoint share the model's eval tasks, so a pin that is right for one
profile has to live in that profile's override rather than on the task. The same
holds for the eval gate: a profile's ``eval_task_score_overrides`` and
``eval_task_limit_samples_overrides`` replace the shared task's reference and
sample cap for that profile only."""

from types import SimpleNamespace

import pytest

from llm_module.eval_command import build_eval_command
from llm_module.eval_configs import (
    _apply_profile_request_body_overrides,
    _apply_profile_score_overrides,
    get_llm_eval_tasks,
)
from llm_module.request_overrides import resolve_request_body
from reference_config.evals.eval_config import (
    ModeReferenceScore,
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import EvalLimitMode


def _dev_specs():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [spec for template in templates for spec in template.expand_to_specs()]


def _profile_tasks(spec):
    """The model's eval tasks as the profile runs them (request-body + gate overrides)."""
    tasks = _eval_config_map[spec.hf_model_repo].tasks
    tasks = _apply_profile_request_body_overrides(tasks, spec)
    tasks = _apply_profile_score_overrides(tasks, spec)
    return {task.task_name: task for task in tasks}


def _profile_bodies(spec):
    return {
        name: resolve_request_body(task) for name, task in _profile_tasks(spec).items()
    }


def _gemma4_26b_profile(impl, device):
    specs = [s for s in _dev_specs() if s.hf_model_repo == "google/gemma-4-26B-A4B-it"]
    return resolve_model_spec(
        specs,
        model="google/gemma-4-26B-A4B-it",
        device=device,
        impl=impl,
        catalog_name="Shield",
    )


def test_every_dev_profile_resolves_every_eval_request_body():
    specs = [spec for spec in _dev_specs() if spec.hf_model_repo in _eval_config_map]
    assert specs
    failures = []
    for spec in specs:
        try:
            _profile_bodies(spec)
        except ValueError as exc:
            failures.append(
                f"{spec.hf_model_repo} / {spec.impl.impl_name} / {spec.device_type}: {exc}"
            )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("impl", "device", "thinking"),
    [
        ("gemma4-26b-a4b-qb2", "p300x2", False),
        ("tt-transformers", "p300x2", True),
        ("tt-transformers", "blackhole_galaxy", True),
    ],
)
def test_gemma4_26b_gpqa_thinking_mode_is_pinned_per_profile(impl, device, thinking):
    """The dedicated QB2 profile runs GPQA Diamond thinking off (its validated path);
    the tt-transformers profiles run it thinking on (the mode of the published and
    GPU references). The shared task carries no pin of its own."""
    spec = _gemma4_26b_profile(impl, device)
    gpqa = next(
        t
        for t in _eval_config_map["google/gemma-4-26B-A4B-it"].tasks
        if t.task_name == "r1_gpqa_diamond"
    )
    assert gpqa.thinking is None
    body = _profile_bodies(spec)["r1_gpqa_diamond"]
    assert body["chat_template_kwargs"]["enable_thinking"] is thinking


def test_gemma4_26b_dedicated_profile_keeps_its_own_gpqa_gate():
    """The dedicated QB2 profile grades GPQA Diamond as it did on main (its thinking-off
    path was calibrated on the fixed 10-question ci-nightly cohort): no full-set GPU
    reference, and ci-nightly runs 10 questions against 80.0 at tolerance 0."""
    gpqa = _profile_tasks(_gemma4_26b_profile("gemma4-26b-a4b-qb2", "p300x2"))[
        "r1_gpqa_diamond"
    ]
    assert gpqa.score.published_score == 82.3
    assert gpqa.score.gpu_reference_score is None
    assert gpqa.score.gpu_reference_score_ref is None
    assert gpqa.score.mode_reference_scores == {
        EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
            score=80.0,
            ref="fixed 10-question CI cohort, thinking off (this profile's validated path)",
            tolerance=0.0,
        )
    }
    assert gpqa.limit_samples_map == {
        EvalLimitMode.CI_NIGHTLY: 10,
        EvalLimitMode.SMOKE_TEST: 0.01,
    }
    nightly = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert nightly["reference_score"] == 80.0
    assert nightly["tolerance"] == 0.0
    assert nightly["is_subset_reference"] is True
    assert accept_eval_score(nightly, 80.0, n_total=10) is True
    assert accept_eval_score(nightly, 70.0, n_total=10) is False
    full = resolve_eval_reference(gpqa.score, None)
    assert full["reference_score"] is None
    assert full["is_subset_reference"] is False
    # The request-body pin of the same profile is untouched by the gate override.
    assert gpqa.request_body == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("device", ["p300x2", "blackhole_galaxy"])
def test_gemma4_26b_tt_transformers_profiles_keep_the_full_set_gpu_reference(device):
    gpqa = _profile_tasks(_gemma4_26b_profile("tt-transformers", device))[
        "r1_gpqa_diamond"
    ]
    assert gpqa.score.gpu_reference_score == 80.81
    assert not gpqa.score.mode_reference_scores
    assert EvalLimitMode.CI_NIGHTLY not in gpqa.limit_samples_map
    nightly = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert nightly["reference_score"] == 80.81
    assert nightly["is_subset_reference"] is False


def _gpqa_task():
    return next(
        t
        for t in _eval_config_map["google/gemma-4-26B-A4B-it"].tasks
        if t.task_name == "r1_gpqa_diamond"
    )


def test_profile_gate_override_rejects_unknown_keys_and_modes():
    unknown_key = SimpleNamespace(
        model_name="x",
        metadata={
            "eval_task_score_overrides": {"r1_gpqa_diamond": {"gpu_reference": 1.0}}
        },
    )
    with pytest.raises(ValueError, match=r"gpu_reference'\]"):
        _apply_profile_score_overrides([_gpqa_task()], unknown_key)
    unknown_mode = SimpleNamespace(
        model_name="x",
        metadata={
            "eval_task_score_overrides": {
                "r1_gpqa_diamond": {
                    "mode_reference_scores": {"ci_weekly": {"score": 1.0}}
                }
            }
        },
    )
    with pytest.raises(ValueError, match="ci_weekly"):
        _apply_profile_score_overrides([_gpqa_task()], unknown_mode)
    unknown_mode_field = SimpleNamespace(
        model_name="x",
        metadata={
            "eval_task_score_overrides": {
                "r1_gpqa_diamond": {
                    "mode_reference_scores": {"ci_nightly": {"score": 1.0, "n": 10}}
                }
            }
        },
    )
    with pytest.raises(ValueError, match="'n'"):
        _apply_profile_score_overrides([_gpqa_task()], unknown_mode_field)
    bad_limit = SimpleNamespace(
        model_name="x",
        metadata={
            "eval_task_limit_samples_overrides": {"r1_gpqa_diamond": {"ci_nightly": 0}}
        },
    )
    with pytest.raises(ValueError, match="positive number"):
        _apply_profile_score_overrides([_gpqa_task()], bad_limit)
    # A task the profile does not override, or a profile without overrides, is returned as is.
    other = SimpleNamespace(
        model_name="x",
        metadata={
            "eval_task_limit_samples_overrides": {"livecodebench": {"ci_nightly": 5}}
        },
    )
    assert _apply_profile_score_overrides([_gpqa_task()], other) == [_gpqa_task()]
    assert _apply_profile_score_overrides(
        [_gpqa_task()], SimpleNamespace(metadata={})
    ) == [_gpqa_task()]


def _catalog_pack():
    """A target pack over the catalog's eval configs: the registered pack keys
    EVAL_CONFIGS on the active MODEL_SPECS catalog, which the dev entries are not in."""
    return SimpleNamespace(
        eval_config=lambda repo: _eval_config_map.get(repo),
        resolve_eval_task_for_device=lambda task, device: task,
        resolve_eval_reference=resolve_eval_reference,
    )


def test_gemma4_26b_dedicated_gate_reaches_the_grader(monkeypatch, tmp_path):
    """The runner grades the tasks ``get_llm_eval_tasks`` returns (reference through
    ``resolve_eval_reference``, sample cap through lm-eval's ``--limit``), so the
    profile's gate has to come through that path, not only the helper."""
    import workflow_module.target_pack as target_pack

    monkeypatch.setattr(target_pack, "get_target_pack", _catalog_pack)
    nightly = SimpleNamespace(
        eval_tasks=None,
        eval_samples=None,
        limit_samples_mode="ci-nightly",
        eval_shard=None,
        eval_shards=None,
    )
    dedicated = _gemma4_26b_profile("gemma4-26b-a4b-qb2", "p300x2")
    tasks = get_llm_eval_tasks(dedicated, nightly, device=dedicated.device_type)
    gpqa = next(t for t in tasks if t.task_name == "r1_gpqa_diamond")
    ref = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert (ref["reference_score"], ref["tolerance"], ref["is_subset_reference"]) == (
        80.0,
        0.0,
        True,
    )
    cmd = build_eval_command(
        gpqa, dedicated, dedicated.device_type, tmp_path, 8000, runtime_config=nightly
    )
    assert cmd[cmd.index("--limit") + 1] == "10"
    assert "--samples" not in cmd
    assert cmd[2:4] == [
        "--request-body",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
    ]

    shared = _gemma4_26b_profile("tt-transformers", "p300x2")
    tasks = get_llm_eval_tasks(shared, nightly, device=shared.device_type)
    gpqa = next(t for t in tasks if t.task_name == "r1_gpqa_diamond")
    ref = resolve_eval_reference(gpqa.score, EvalLimitMode.CI_NIGHTLY)
    assert (ref["reference_score"], ref["is_subset_reference"]) == (80.81, False)
    cmd = build_eval_command(
        gpqa, shared, shared.device_type, tmp_path, 8000, runtime_config=nightly
    )
    assert "--limit" not in cmd

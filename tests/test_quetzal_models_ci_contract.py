# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from reference_config.evals.eval_config import ModeReferenceScore
from workflows.model_spec import load_templates_from_yaml
from workflows.validate_setup import validate_quetzal_models_ci_contract
from workflows.workflow_types import EvalLimitMode


def _runtime(workflow="release"):
    return SimpleNamespace(
        workflow=workflow,
        agentic_benchmark=None,
        external_agentic_contract=None,
    )


def _valid_env():
    package_id = "sha256-" + "a" * 64 + "-" + "b" * 64
    root = f"/home/container_app_user/quetzal/packages/{package_id}"
    return {
        "QUETZAL_VLLM": "1",
        "VLLM_PLUGINS": "quetzal_model_registry,tt",
        "TT_VLLM_BUILTIN_MODELS": "0",
        "QUETZAL_PACKAGE_ID": package_id,
        "QUETZAL_PACKAGE_ROOT": root,
        "QZ_MODELS_ROOT": root,
        "QZ_QUALIFICATION_MANIFEST": f"{root}/qualification_manifest.yaml",
        "QUETZAL_BUNDLE_MANIFEST_SHA256": "c" * 64,
        "QUETZAL_REQUIRED_SOURCE_REVISION": "d" * 40,
        "QUETZAL_REQUIRED_TT_METAL_COMMIT": "1" * 40,
        "QUETZAL_TT_METAL_PATCHSET_STATUS": "applied",
        "QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256": "e" * 64,
        "QUETZAL_RUNTIME_ATTESTATION_SHA256": "f" * 64,
        "QUETZAL_PREFILL_GENERATED_PY": f"{root}/prefill/generated.py",
        "QUETZAL_PREFILL_METADATA_JSON": f"{root}/prefill/metadata.json",
        "QUETZAL_DECODE_GENERATED_PY": f"{root}/decode/generated.py",
        "QUETZAL_DECODE_METADATA_JSON": f"{root}/decode/metadata.json",
        "QUETZAL_WEIGHTS": f"{root}/compiled_weights/weights.pt",
    }


def _valid_task():
    nightly_ids = [
        "django__django-11299",
        "sympy__sympy-13551",
    ]
    cfg = SimpleNamespace(
        agent_backend="mini-swe-agent",
        max_input_tokens=5 * 1024,
        max_output_tokens=2 * 1024,
        mini_observation_chars=2048,
        mini_agent_kwargs={"step_limit": 8},
        instance_selection_provenance="reviewed fixed five-instance subset v1",
        qualification_claim="models_ci_graded",
        selection_policy="reviewed_fixed_subset",
        dataset_revision="2" * 40,
        ordered_instance_ids_sha256=hashlib.sha256(
            json.dumps(nightly_ids, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        n_tasks=None,
        instance_ids_map={
            EvalLimitMode.SMOKE_TEST: ["django__django-11299"],
            EvalLimitMode.CI_NIGHTLY: nightly_ids,
        },
    )
    score = SimpleNamespace(
        published_score=None,
        published_score_ref=None,
        gpu_reference_score=None,
        gpu_reference_score_ref=None,
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                score=50.0,
                ref="independent exact two-instance GPU baseline",
            )
        },
    )
    return SimpleNamespace(
        task_name="swe_bench_verified",
        agentic_eval_config=None,
        swebench_eval_config=cfg,
        min_context_required=8 * 1024,
        score=score,
    )


def _valid_spec():
    return SimpleNamespace(
        impl=SimpleNamespace(impl_id="quetzal"),
        env_vars=_valid_env(),
        device_model_spec=SimpleNamespace(max_context=8 * 1024, device="P300X2"),
        model_name="synthetic-quetzal-model",
    )


def test_non_quetzal_and_nonrelease_paths_are_unchanged():
    native = _valid_spec()
    native.impl.impl_id = "native"
    native.env_vars = {}
    validate_quetzal_models_ci_contract(native, _runtime())

    local = _valid_spec()
    local.env_vars = {}
    validate_quetzal_models_ci_contract(local, _runtime("agentic"))


def test_complete_graded_generated_contract_passes(monkeypatch):
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks",
        lambda *_args: [_valid_task()],
    )
    validate_quetzal_models_ci_contract(_valid_spec(), _runtime())


def test_unpatched_upstream_runtime_contract_passes(monkeypatch):
    spec = _valid_spec()
    spec.env_vars["QUETZAL_TT_METAL_PATCHSET_STATUS"] = "none"
    spec.env_vars.pop("QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256")
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks",
        lambda *_args: [_valid_task()],
    )
    validate_quetzal_models_ci_contract(spec, _runtime())


def test_full_dataset_contract_fails_until_complete_coverage_is_provable(
    monkeypatch,
):
    task = _valid_task()
    task.swebench_eval_config.selection_policy = "full_dataset"
    task.swebench_eval_config.dataset_revision = "2" * 40
    task.swebench_eval_config.instance_ids_map.pop(EvalLimitMode.CI_NIGHTLY)
    task.score.gpu_reference_score = 42.0
    task.score.gpu_reference_score_ref = (
        "independent full SWE-bench Verified run with the same harness"
    )
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks", lambda *_args: [task]
    )
    with pytest.raises(ValueError, match="authoritative dataset cardinality"):
        validate_quetzal_models_ci_contract(_valid_spec(), _runtime())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda spec, task: spec.env_vars.update(
                QUETZAL_RUNTIME_ATTESTATION_SHA256="bad"
            ),
            "RUNTIME_ATTESTATION",
        ),
        (
            lambda spec, task: spec.env_vars.pop("QUETZAL_REQUIRED_TT_METAL_COMMIT"),
            "missing",
        ),
        (
            lambda spec, task: spec.env_vars.update(
                QUETZAL_REQUIRED_TT_METAL_COMMIT="bad"
            ),
            "TT_METAL_COMMIT",
        ),
        (
            lambda spec, task: spec.env_vars.update(
                QUETZAL_TT_METAL_PATCHSET_STATUS="none"
            ),
            "cannot carry",
        ),
        (
            lambda spec, task: spec.env_vars.update(
                QUETZAL_TT_METAL_PATCHSET_STATUS="unknown"
            ),
            "must be 'applied' or 'none'",
        ),
        (lambda spec, task: spec.env_vars.update(VLLM_PLUGINS="tt"), "VLLM_PLUGINS"),
        (
            lambda spec, task: spec.env_vars.update(
                QUETZAL_DECODE_GENERATED_PY="/tmp/escaped/generated.py"
            ),
            "inside the exact",
        ),
        (
            lambda spec, task: setattr(spec.device_model_spec, "max_context", 4096),
            "exceeds",
        ),
        (
            lambda spec, task: setattr(
                task.swebench_eval_config, "mini_observation_chars", None
            ),
            "observation",
        ),
        (
            lambda spec, task: task.swebench_eval_config.mini_agent_kwargs.clear(),
            "step_limit",
        ),
        (
            lambda spec, task: setattr(
                task.swebench_eval_config, "instance_selection_provenance", ""
            ),
            "provenance",
        ),
        (
            lambda spec, task: setattr(
                task.swebench_eval_config,
                "qualification_claim",
                "local_behavioral_only",
            ),
            "local/report-only",
        ),
        (lambda spec, task: task.score.mode_reference_scores.clear(), "mode reference"),
        (
            lambda spec, task: task.score.mode_reference_scores.update(
                {EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(score=0, ref="")}
            ),
            "positive measured score",
        ),
        (
            lambda spec, task: setattr(
                task.swebench_eval_config, "dataset_revision", None
            ),
            "dataset revision",
        ),
        (
            lambda spec, task: setattr(
                task.swebench_eval_config,
                "ordered_instance_ids_sha256",
                "0" * 64,
            ),
            "canonical ordered",
        ),
    ],
)
def test_incomplete_release_contracts_fail_closed(monkeypatch, mutation, match):
    spec = _valid_spec()
    task = _valid_task()
    mutation(spec, task)
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks",
        lambda *_args: [task],
    )
    with pytest.raises(ValueError, match=match):
        validate_quetzal_models_ci_contract(spec, _runtime())


@pytest.mark.parametrize(
    "package_id",
    [
        "sha256-junk",
        "sha256-" + "a" * 63 + "-" + "b" * 64,
        "sha256-" + "A" * 64 + "-" + "b" * 64,
        "../sha256-" + "a" * 64 + "-" + "b" * 64,
    ],
)
def test_package_id_uses_exact_content_addressed_grammar(monkeypatch, package_id):
    spec = _valid_spec()
    spec.env_vars["QUETZAL_PACKAGE_ID"] = package_id
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks",
        lambda *_args: [_valid_task()],
    )
    with pytest.raises(ValueError, match="content-addressed"):
        validate_quetzal_models_ci_contract(spec, _runtime())


def test_missing_context_admitted_swe_is_not_a_release(monkeypatch):
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks", lambda *_args: []
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_quetzal_models_ci_contract(_valid_spec(), _runtime())


def test_missing_runtime_attestation_is_a_provenance_warning(monkeypatch, caplog):
    spec = _valid_spec()
    spec.env_vars.pop("QUETZAL_RUNTIME_ATTESTATION_SHA256")
    monkeypatch.setattr(
        "workflows.validate_setup._selected_agentic_tasks",
        lambda *_args: [_valid_task()],
    )

    validate_quetzal_models_ci_contract(spec, _runtime())

    assert "[unattested]" in caplog.text
    assert "qualification remains runnable" in caplog.text


@pytest.mark.parametrize(
    "model_repo,expected",
    [
        ("Qwen/Qwen3.6-27B", "local/report-only"),
        ("google/gemma-4-31B-it", "exactly one"),
        ("openai/gpt-oss-120b", "local/report-only"),
    ],
)
def test_every_current_quetzal_dev_row_traverses_same_release_gate(
    model_repo, expected
):
    templates = load_templates_from_yaml(Path("workflows/model_specs/dev/llm.yaml"))
    template = next(
        item
        for item in templates
        if item.impl.impl_id == "quetzal" and item.weights == [model_repo]
    )
    spec = template.expand_to_specs()[0]
    with pytest.raises(ValueError, match=expected):
        validate_quetzal_models_ci_contract(spec, _runtime())


def test_every_active_models_ci_quetzal_row_is_subject_to_release_gate():
    config = json.loads(Path(".github/workflows/models-ci-config.json").read_text())
    active = []
    for model_name, model_config in config["models"].items():
        for implementation in model_config.get("implementations", []):
            if implementation.get("impl") == "quetzal":
                active.append(model_name)
    assert active == ["Qwen3.6-27B"]

    templates = load_templates_from_yaml(Path("workflows/model_specs/dev/llm.yaml"))
    template = next(
        item
        for item in templates
        if item.impl.impl_id == "quetzal" and item.weights == ["Qwen/Qwen3.6-27B"]
    )
    with pytest.raises(ValueError, match="local/report-only"):
        validate_quetzal_models_ci_contract(template.expand_to_specs()[0], _runtime())

"""Per-profile eval config variants (catalog metadata ``eval_config_variant``)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_module.parsers.agentic import compute_accuracy_check
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    EVAL_CONFIG_VARIANTS,
    EvalLimitMode,
    resolve_eval_config,
)
from workflow_module import command_factory
from workflows import workflow_dispatch
from workflows.model_spec import load_templates_from_yaml
from workflows.target_pack_provider import TenstorrentTargetPack
from workflows.workflow_types import WorkflowVenvType

GEMMA = "google/gemma-4-12B-it"
DEV_LLM_SPECS_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "model_specs"
    / "dev"
    / "llm.yaml"
)


def _spec(metadata, repo=GEMMA):
    return SimpleNamespace(hf_model_repo=repo, metadata=metadata, model_name="test")


def _agentic(cfg):
    return [
        t.task_name
        for t in cfg.tasks
        if t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]


def test_gemma_qb2_variant_swaps_only_the_agentic_tasks():
    shared = ALL_EVAL_CONFIGS[GEMMA]
    variant = EVAL_CONFIG_VARIANTS["gemma4-12b-qb2"]
    assert _agentic(shared) == ["terminal_bench_2"]
    assert _agentic(variant) == ["terminal_bench_2_1", "swe_bench_verified"]
    lm_eval = lambda cfg: [  # noqa: E731
        t for t in cfg.tasks if t.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC
    ]
    # The lm-eval tasks are the shared objects, so the shared gates and
    # profile overrides apply to them unchanged.
    assert lm_eval(variant) == lm_eval(shared)


@pytest.mark.parametrize("task_name", ["terminal_bench_2_1", "swe_bench_verified"])
def test_gemma_qb2_agentic_gate_is_one_of_five(task_name):
    task = next(
        t
        for t in EVAL_CONFIG_VARIANTS["gemma4-12b-qb2"].tasks
        if t.task_name == task_name
    )
    cfg = task.agentic_eval_config
    assert cfg.task_names_map[EvalLimitMode.CI_NIGHTLY] == cfg.task_names
    assert len(cfg.task_names) == 5
    check = lambda k: compute_accuracy_check(  # noqa: E731
        {"accuracy": k / 5, "n_trials": 5}, task.score, EvalLimitMode.CI_NIGHTLY
    )
    assert check(0) == 3  # FAIL
    assert check(1) == 2  # PASS


def test_resolve_eval_config_uses_variant_only_when_named():
    assert resolve_eval_config(GEMMA, _spec({})) is ALL_EVAL_CONFIGS[GEMMA]
    assert resolve_eval_config(GEMMA, None) is ALL_EVAL_CONFIGS[GEMMA]
    assert (
        resolve_eval_config(GEMMA, _spec({"eval_config_variant": "gemma4-12b-qb2"}))
        is EVAL_CONFIG_VARIANTS["gemma4-12b-qb2"]
    )


def test_resolve_eval_config_rejects_unknown_or_mismatched_variant():
    with pytest.raises(ValueError, match="not in EVAL_CONFIG_VARIANTS"):
        resolve_eval_config(GEMMA, _spec({"eval_config_variant": "nope"}))
    with pytest.raises(ValueError, match="is for google/gemma-4-12B-it"):
        resolve_eval_config(
            "Qwen/Qwen3.6-27B",
            _spec({"eval_config_variant": "gemma4-12b-qb2"}, repo="Qwen/Qwen3.6-27B"),
        )


def test_dev_catalog_points_only_gemma4_12b_qb2_at_the_variant():
    by_impl = {}
    for template in load_templates_from_yaml(DEV_LLM_SPECS_PATH):
        if GEMMA not in template.weights:
            continue
        meta = (template.metadata or {}).get(GEMMA, {})
        by_impl[template.impl.impl_id] = meta.get("eval_config_variant")
    assert by_impl["gemma4_12b_qb2"] == "gemma4-12b-qb2"
    assert all(v is None for k, v in by_impl.items() if k != "gemma4_12b_qb2")


def test_run_paths_resolve_the_variant(monkeypatch):
    spec = _spec({"eval_config_variant": "gemma4-12b-qb2"})
    pack = TenstorrentTargetPack()
    monkeypatch.setattr("workflow_module.target_pack.get_target_pack", lambda: pack)
    assert (
        command_factory._resolve_eval_config(spec)
        is EVAL_CONFIG_VARIANTS["gemma4-12b-qb2"]
    )
    assert (
        workflow_dispatch._eval_config_for(spec)
        is EVAL_CONFIG_VARIANTS["gemma4-12b-qb2"]
    )


def test_packs_without_the_hook_resolve_by_repo(monkeypatch):
    seen = []
    pack = SimpleNamespace(eval_config=lambda repo: seen.append(repo) or "shared")
    monkeypatch.setattr("workflow_module.target_pack.get_target_pack", lambda: pack)
    spec = _spec({"eval_config_variant": "gemma4-12b-qb2"})
    assert command_factory._resolve_eval_config(spec) == "shared"
    assert workflow_dispatch._eval_config_for(spec) == "shared"
    assert seen == [GEMMA, GEMMA]

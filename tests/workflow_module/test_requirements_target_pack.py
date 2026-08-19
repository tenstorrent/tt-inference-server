# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the requirements-driven model-spec provider and target pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow_module.requirements_schema import load_requirements
from workflows.model_spec_provider import (
    TenstorrentModelSpecProvider,
    hardware_to_device_name,
)
from workflows.requirements_target_pack import (
    RequirementsModelSpecProvider,
    RequirementsTargetPack,
)
from workflows.target_pack_provider import TenstorrentTargetPack
from workflows.workflow_types import DeviceTypes

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "requirements"
    / "acme-llm-serving.json"
)


@pytest.fixture
def doc():
    return load_requirements(_FIXTURE)


@pytest.fixture
def pack(doc):
    return RequirementsTargetPack(doc, TenstorrentTargetPack())


# --- hardware mapping --------------------------------------------------------


@pytest.mark.parametrize("hardware", ["SC8", "SC12", "SC20", "sc8", "SC128"])
def test_hardware_super_cluster_family_maps_to_super_cluster(hardware):
    # The Super Cluster ships in several node counts; every SC<N> is the same
    # SUPER_CLUSTER device from the engine's perspective.
    assert hardware_to_device_name(hardware) == "SUPER_CLUSTER"


def test_hardware_accepts_device_name_directly():
    assert hardware_to_device_name("galaxy") == "GALAXY"


@pytest.mark.parametrize("hardware", ["mystery-box", "SC", "SC8X", "8SC"])
def test_hardware_unknown_raises(hardware):
    with pytest.raises(ValueError, match="Unknown deployment.hardware"):
        hardware_to_device_name(hardware)


# --- off-catalog synthesis ---------------------------------------------------


def test_synthesize_off_catalog_spec():
    provider = TenstorrentModelSpecProvider()
    spec = provider.synthesize(
        model_name="acme/tiny-llm",
        hf_model_repo="acme/tiny-llm",
        device="super_cluster",
        max_context=8192,
        max_concurrency=16,
    )
    assert spec.model_name == "acme/tiny-llm"
    assert spec.device_type == DeviceTypes.SUPER_CLUSTER
    assert spec.device_model_spec.max_context == 8192
    assert spec.device_model_spec.max_concurrency == 16


def test_synthesized_model_id_is_a_single_path_component():
    """An HF org prefix must not turn model_id into a nested directory.

    model_id names the runtime-spec JSON, the run log, and the per-eval output
    dirs; a raw "org/model" silently writes into an "org/" subdirectory that
    nothing has created, which fails the run before the workflow starts.
    """
    spec = TenstorrentModelSpecProvider().synthesize(
        model_name="acme/tiny-llm",
        hf_model_repo="acme/tiny-llm",
        device="super_cluster",
        max_context=8192,
        max_concurrency=16,
    )
    assert "/" not in spec.model_id
    assert spec.model_id.startswith("acme__tiny-llm")


def test_requirements_provider_synthesizes_when_off_catalog(doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    # The document's model name is the HF repo path and is not a catalog name,
    # so resolution falls back to synthesis from the doc's metadata.
    spec = provider.resolve(doc.model.name, "super_cluster")
    assert spec.model_name == doc.model.name
    assert spec.device_model_spec.max_context == doc.model.context_length
    assert (
        spec.device_model_spec.max_concurrency
        == doc.deployment.max_concurrency_per_instance
    )


# --- eval config synthesis ---------------------------------------------------


def test_eval_config_maps_names_and_regates(pack, doc):
    cfg = pack.eval_config(doc.model.name)
    by_task = {t.task_name: t for t in cfg.tasks}
    assert set(by_task) == {
        "gpqa_diamond_cot_zeroshot",
        "swe_bench_verified",
        "terminal_bench_2",
    }
    # Reference score + tolerance re-gated from the document.
    gpqa = by_task["gpqa_diamond_cot_zeroshot"]
    assert gpqa.score.gpu_reference_score == 79.2
    assert gpqa.score.tolerance == 0.05
    assert gpqa.priority == "must"
    # Terminal-Bench 2.0 is a "should" in the document.
    assert by_task["terminal_bench_2"].priority == "should"
    # Borrowed tasks keep their runnable harness config.
    assert by_task["swe_bench_verified"].swebench_eval_config is not None
    assert by_task["terminal_bench_2"].agentic_eval_config is not None


def test_eval_config_unknown_name_raises(doc):
    from dataclasses import replace

    from workflow_module.requirements_schema import AccuracyEval

    bad = replace(doc, accuracy_evals=[AccuracyEval(name="Made-Up-Bench")])
    pack = RequirementsTargetPack(bad, TenstorrentTargetPack())
    with pytest.raises(ValueError, match="no known catalog task mapping"):
        pack.eval_config(doc.model.name)


def test_eval_config_none_when_no_evals(doc):
    from dataclasses import replace

    empty = replace(doc, accuracy_evals=[])
    pack = RequirementsTargetPack(empty, TenstorrentTargetPack())
    assert pack.eval_config(doc.model.name) is None


def test_eval_task_synthesized_with_neutral_defaults(doc, monkeypatch):
    from dataclasses import replace

    # Fully off-catalog path: no catalog model defines the task, so the pack
    # synthesizes a neutral EvalTask — max_length from the document's
    # contextLength, streaming on, no sampling overrides, doc-gated score.
    # Narrow the doc to GPQA only: the harness-backed evals (SWE-bench,
    # Terminal-Bench) cannot be synthesized and are covered by the test below.
    gpqa_only = replace(doc, accuracy_evals=[doc.accuracy_evals[0]])
    pack = RequirementsTargetPack(gpqa_only, TenstorrentTargetPack())
    monkeypatch.setattr(pack, "_find_task_template", lambda task_name: None)
    cfg = pack.eval_config("acme/off-catalog-model")
    gpqa = next(t for t in cfg.tasks if t.task_name == "gpqa_diamond_cot_zeroshot")

    assert gpqa.model_kwargs == {"timeout": "3600", "max_length": 131072}
    assert gpqa.gen_kwargs == {"stream": "True"}  # synthesized default: stream on
    assert gpqa.score.gpu_reference_score == 79.2
    assert gpqa.score.published_score == 80.9
    assert gpqa.score.tolerance == 0.05
    assert gpqa.score.score_func_kwargs == {
        "result_keys": ["exact_match,flexible-extract"],
        "unit": "percent",
    }
    assert gpqa.priority == "must"


def test_harness_concurrency_comes_from_the_document(doc, pack):
    """The borrowed template's trial count must not decide this deployment's.

    Which catalog model a harness template is borrowed from is decided by
    EVAL_CONFIGS iteration order, so its n_concurrent_trials is arbitrary here
    (Terminal-Bench 2 in particular borrows a serial, n=1 template). The
    document states what the instance under test serves concurrently, so that
    is what the harnesses run at.
    """
    harness_tasks = {
        task.task_name: (task.agentic_eval_config or task.swebench_eval_config)
        for task in pack.eval_config(doc.model.name).tasks
        if (task.agentic_eval_config or task.swebench_eval_config) is not None
    }
    assert set(harness_tasks) == {"swe_bench_verified", "terminal_bench_2"}
    for cfg in harness_tasks.values():
        assert cfg.n_concurrent_trials == doc.deployment.max_concurrency_per_instance


def test_harness_concurrency_falls_back_when_document_is_silent(doc):
    """No maxConcurrencyPerInstance means nothing to override with."""
    from dataclasses import replace

    silent = replace(
        doc, deployment=replace(doc.deployment, max_concurrency_per_instance=None)
    )
    pack = RequirementsTargetPack(silent, TenstorrentTargetPack())
    task = next(
        t
        for t in pack.eval_config(silent.model.name).tasks
        if t.task_name == "terminal_bench_2"
    )
    borrowed = pack._find_task_template("terminal_bench_2")
    assert (
        task.agentic_eval_config.n_concurrent_trials
        == borrowed.agentic_eval_config.n_concurrent_trials
    )


def test_eval_task_synthesis_rejects_harness_backed_task(pack, monkeypatch):
    # SWE-bench needs its SWEbenchEvalConfig harness wiring, which cannot be
    # synthesized — with no catalog template it must fail loudly.
    monkeypatch.setattr(pack, "_find_task_template", lambda task_name: None)
    with pytest.raises(ValueError, match="No catalog template or built-in profile"):
        pack.eval_config("acme/off-catalog-model")


# --- benchmark config synthesis ----------------------------------------------


def test_benchmark_config_builds_sweep_from_document(pack, doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    cfg = pack.benchmark_config(spec)

    assert len(cfg.tasks) == 1
    points = cfg.tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
    # One point per sweep entry in the document.
    assert len(points) == len(doc.scenarios[0].sweep)

    # Every point carries the scenario SLOs; only the peak-concurrency point
    # carries the aggregate scalar targets (throughput/goodput).
    peak = max(p.max_concurrency for p in points)
    peak_point = next(p for p in points if p.max_concurrency == peak)
    peak_target = peak_point.targets["target"]
    assert peak_target.ttft_ms == 2000
    assert peak_target.tpot_ms == 20
    assert peak_target.e2el_ms == 20000
    assert peak_target.tput == 12000
    assert peak_target.goodput == 99

    low_point = next(p for p in points if p.max_concurrency == 1)
    low_target = low_point.targets["target"]
    assert low_target.ttft_ms == 2000  # SLO applies to every point
    assert low_target.tput is None  # aggregate target only at peak
    # SLOs are must, so every point is a must sweep point here.
    assert low_point.priority == "must"


def test_smoke_test_benchmark_config_narrows_to_one_point(pack, doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    cfg = pack.benchmark_config(spec)
    smoke = pack.smoke_test_benchmark_config(cfg, DeviceTypes.SUPER_CLUSTER)
    points = smoke.tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
    assert len(points) == 1


def test_target_pack_delegates_unspecified_content(pack):
    # Anything the document does not define falls through to the TT pack.
    assert pack.agentic_traces_min_profile_seconds() == (
        TenstorrentTargetPack().agentic_traces_min_profile_seconds()
    )
    assert pack.performance_targets_path() == (
        TenstorrentTargetPack().performance_targets_path()
    )
    assert pack.extra_spec_metadata_fields() == (
        TenstorrentTargetPack().extra_spec_metadata_fields()
    )

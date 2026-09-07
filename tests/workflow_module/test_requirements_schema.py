# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the engine-owned requirements-document loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow_module.requirements_schema import (
    PRIORITY_MUST,
    PRIORITY_SHOULD,
    RequirementsError,
    RequirementsDoc,
    load_requirements,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "requirements"
    / "acme-llm-serving.json"
)


def test_loads_acme_fixture():
    doc = load_requirements(_FIXTURE)
    assert doc.id == "acme-llm-serving"
    assert doc.schema_version == "2.1.0"
    assert doc.model.name == "openai/gpt-oss-120b"
    assert doc.model.context_length == 131072
    assert doc.deployment.hardware == "SC8"
    assert doc.deployment.max_concurrency_per_instance == 64


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(RequirementsError, match="not found"):
        load_requirements(tmp_path / "does-not-exist.json")


def test_load_rejects_directory(tmp_path):
    # A directory is not a requirements document; fail loudly rather than
    # letting open() raise a confusing IsADirectoryError downstream.
    with pytest.raises(RequirementsError, match="not a file"):
        load_requirements(tmp_path)


def test_load_resolves_traversal_to_real_file(tmp_path):
    # A path containing ".." still resolves to the real file (canonicalized),
    # so legitimate relative references keep working.
    sub = tmp_path / "sub"
    sub.mkdir()
    target = tmp_path / "req.json"
    target.write_text(
        json.dumps(
            {
                "schemaVersion": "2.1.0",
                "model": {"name": "acme/tiny"},
                "deployment": {},
            }
        )
    )
    doc = load_requirements(sub / ".." / "req.json")
    assert doc.model.name == "acme/tiny"


def test_accuracy_evals_parsed_with_priorities():
    doc = load_requirements(_FIXTURE)
    by_name = {e.name: e for e in doc.accuracy_evals}
    assert set(by_name) == {"GPQA-Diamond", "SWE-bench Verified", "Terminal-Bench 2.0"}

    gpqa = by_name["GPQA-Diamond"]
    assert gpqa.gpu_reference_score == 79.2
    assert gpqa.published_score == 80.9
    assert gpqa.tolerance == 0.05
    assert gpqa.priority == PRIORITY_MUST
    assert gpqa.task_category == "science"

    tb = by_name["Terminal-Bench 2.0"]
    assert tb.priority == PRIORITY_SHOULD
    assert tb.gpu_reference_score == 41.5


def test_scenario_sweep_and_targets_parsed():
    doc = load_requirements(_FIXTURE)
    assert len(doc.scenarios) == 1
    scenario = doc.scenarios[0]
    assert scenario.id == "interactive-chat"
    assert scenario.osl_values == [128, 1024]

    # scalar targets: throughput (must) and goodput (should).
    by_metric = {t.metric: t for t in scenario.scalar_targets}
    assert by_metric["system_throughput"].target == 12000
    assert by_metric["system_throughput"].comparator == "gte"
    assert by_metric["system_throughput"].priority == PRIORITY_MUST
    assert by_metric["request_goodput"].priority == PRIORITY_SHOULD

    # SLOs.
    assert scenario.slo is not None
    assert scenario.slo.ttft_ms == 2000
    assert scenario.slo.tpot_ms == 20
    assert scenario.slo.e2el_ms == 20000

    # sweep points carry isl/osl/concurrency plus verbatim reference data.
    assert scenario.sweep
    first = scenario.sweep[0]
    assert (first.isl, first.osl, first.concurrency) == (128, 128, 1)
    assert first.reference["ttftMeanMs"] == 128


def test_unknown_keys_are_ignored(tmp_path):
    doc_dict = {
        "schemaVersion": "2.9.0",
        "id": "x",
        "model": {"name": "foo/bar", "contextLength": 4096, "somethingNew": 1},
        "deployment": {"hardware": "SC8", "futureField": True},
        "accuracyEvals": [],
        "scenarios": [],
        "unrecognizedTopLevel": {"a": 1},
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    doc = load_requirements(path)
    assert isinstance(doc, RequirementsDoc)
    assert doc.model.name == "foo/bar"
    assert doc.model.context_length == 4096


def test_unsupported_schema_major_rejected(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"schemaVersion": "3.0.0", "model": {"name": "a/b"}}))
    with pytest.raises(RequirementsError, match="Unsupported schemaVersion"):
        load_requirements(path)


def test_missing_schema_version_rejected(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"model": {"name": "a/b"}}))
    with pytest.raises(RequirementsError, match="schemaVersion"):
        load_requirements(path)


def test_missing_model_rejected(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"schemaVersion": "2.0.0"}))
    with pytest.raises(RequirementsError, match="model"):
        load_requirements(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(RequirementsError, match="not found"):
        load_requirements(tmp_path / "does-not-exist.json")


def test_invalid_priority_rejected(tmp_path):
    doc_dict = {
        "schemaVersion": "2.0.0",
        "model": {"name": "a/b"},
        "accuracyEvals": [{"name": "X", "priority": "nice-to-have"}],
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    with pytest.raises(RequirementsError, match="priority"):
        load_requirements(path)


def test_invalid_comparator_rejected(tmp_path):
    doc_dict = {
        "schemaVersion": "2.0.0",
        "model": {"name": "a/b"},
        "scenarios": [
            {
                "id": "s",
                "scalarTargets": [
                    {"metric": "system_throughput", "target": 1, "comparator": "eq"}
                ],
            }
        ],
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    with pytest.raises(RequirementsError, match="comparator"):
        load_requirements(path)


def _agentic_doc(**overrides):
    """A schema 2.6 document: identity nested, agentic sweep in workloads."""
    workload = {
        "kind": "agentic",
        "id": "w1",
        "name": "New agentic scenario",
        "slo": {"ttftMs": 2000, "tpotMs": 20, "e2elMs": 20000},
        "agenticWorkload": {"traces": [{"name": "AgentX - Claude Code"}]},
        "agenticSweep": [
            {"concurrency": 1, "e2elP90Ms": 16475.15},
            {"concurrency": 8, "e2elP90Ms": 12242.25},
        ],
        "maxConcurrency": 64,
    }
    workload.update(overrides)
    return {
        "schemaVersion": "2.6.0",
        "document": {
            "id": "doc-1",
            "meta": {"customer": "Ant"},
            "model": {"name": "google/gemma-4-31B-it", "contextLength": 131072},
            "deployment": {"hardware": "SC24", "maxConcurrencyPerInstance": 1},
        },
        "workloads": [workload],
    }


def _write(tmp_path, doc_dict):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    return path


def test_reads_identity_from_the_document_envelope(tmp_path):
    """Schema 2.6 nests model/deployment/meta under 'document'."""
    doc = load_requirements(_write(tmp_path, _agentic_doc()))

    assert doc.id == "doc-1"
    assert doc.model.name == "google/gemma-4-31B-it"
    assert doc.deployment.hardware == "SC24"
    assert doc.meta["customer"] == "Ant"


def test_parses_the_agentic_sweep_and_its_slos(tmp_path):
    doc = load_requirements(_write(tmp_path, _agentic_doc()))

    (workload,) = doc.agentic_workloads
    assert [p.concurrency for p in workload.sweep] == [1, 8]
    assert workload.max_concurrency == 64
    assert (workload.slo.ttft_ms, workload.slo.tpot_ms, workload.slo.e2el_ms) == (
        2000,
        20,
        20000,
    )


def test_keeps_sweep_point_expectations_for_grading(tmp_path):
    """Values we do not drive the run with still have to survive for grading."""
    doc = load_requirements(_write(tmp_path, _agentic_doc()))

    assert doc.agentic_workloads[0].sweep[0].reference["e2elP90Ms"] == 16475.15


def test_ignores_non_agentic_workloads(tmp_path):
    """A text workload is a benchmark scenario, not a trace replay."""
    doc_dict = _agentic_doc()
    doc_dict["workloads"].append({"kind": "text", "id": "w2"})

    doc = load_requirements(_write(tmp_path, doc_dict))

    assert [w.id for w in doc.agentic_workloads] == ["w1"]


def test_sweep_point_without_concurrency_rejected(tmp_path):
    doc_dict = _agentic_doc(agenticSweep=[{"e2elP90Ms": 1.0}])

    with pytest.raises(RequirementsError, match="concurrency"):
        load_requirements(_write(tmp_path, doc_dict))


def test_flat_documents_still_load(tmp_path):
    """Earlier 2.x revisions put identity at the top level and have no sweep."""
    doc_dict = {
        "schemaVersion": "2.1.0",
        "model": {"name": "a/b"},
        "deployment": {"hardware": "SC8"},
    }

    doc = load_requirements(_write(tmp_path, doc_dict))

    assert doc.model.name == "a/b"
    assert doc.deployment.hardware == "SC8"
    assert doc.agentic_workloads == []

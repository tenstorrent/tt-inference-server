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

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
    Slo,
    effective_slo,
    input_throughput_tps,
    load_requirements,
    stated_target,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "requirements"
    / "acme-llm-serving.json"
)


def _stage(**fields):
    """One delivery stage; a 3.x document always has at least one."""
    return {
        "key": "stage-1",
        "name": "Stage 1",
        "progress": {"status": "not_started"},
        **fields,
    }


def _doc(*, scenarios=(), evals=(), **identity):
    """A minimal canonical 3.1 document whose single stage holds ``scenarios``."""
    return {
        "schemaVersion": "3.1.0",
        "id": "d",
        "model": {"name": "a/b"},
        **identity,
        "stages": [_stage(scenarios=list(scenarios), accuracyEvals=list(evals))],
    }


def test_loads_acme_fixture():
    doc = load_requirements(_FIXTURE)
    assert doc.id == "acme-llm-serving"
    assert doc.schema_version == "3.1.0"
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
    target.write_text(json.dumps(_doc(model={"name": "acme/tiny"}, deployment={})))
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


def test_eval_gen_kwargs_parsed_into_the_harness_vocabulary():
    """The document's camelCase parameters, transliterated, and only those set."""
    from workflow_module.requirements_schema import AccuracyEval

    ae = AccuracyEval.from_dict(
        {
            "name": "GPQA-Diamond",
            "gpuReferenceScore": 79.2,
            "genKwargs": {
                "temperature": 0.6,
                "topP": 0.95,
                "topK": 20,
                "maxGenToks": 32768,
                "reasoningEffort": "high",
            },
        }
    )
    assert ae.gen_kwargs.stated() == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_gen_toks": 32768,
        "reasoning_effort": "high",
    }
    # Only the stated ones: an unset parameter is not a claim, so it must not
    # reach a harness as a value the document never chose.
    partial = AccuracyEval.from_dict(
        {
            "name": "GPQA-Diamond",
            "gpuReferenceScore": 79.2,
            "genKwargs": {"temperature": 0.6},
        }
    )
    assert partial.gen_kwargs.stated() == {"temperature": 0.6}


def test_eval_without_gen_kwargs_states_nothing():
    """Absent, empty and unrecognised-only all mean the same: no claim."""
    from workflow_module.requirements_schema import AccuracyEval

    base = {"name": "GPQA-Diamond", "gpuReferenceScore": 79.2}
    assert AccuracyEval.from_dict(base).gen_kwargs is None
    assert AccuracyEval.from_dict({**base, "genKwargs": {}}).gen_kwargs is None
    # Tolerant parsing (this loader ignores unknown keys so a newer minor
    # revision still loads) must not turn an unknown key into an empty claim.
    assert (
        AccuracyEval.from_dict({**base, "genKwargs": {"seed": 42}}).gen_kwargs is None
    )


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
    doc_dict = _doc(
        model={"name": "foo/bar", "contextLength": 4096, "somethingNew": 1},
        deployment={"hardware": "SC8", "futureField": True},
        unrecognizedTopLevel={"a": 1},
    )
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    doc = load_requirements(path)
    assert isinstance(doc, RequirementsDoc)
    assert doc.model.name == "foo/bar"
    assert doc.model.context_length == 4096


@pytest.mark.parametrize("version", ["2.8.0", "3.0.0", "4.0.0"])
def test_versions_outside_3_1_and_later_3_x_rejected(tmp_path, version):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({**_doc(), "schemaVersion": version}))
    with pytest.raises(RequirementsError, match="Unsupported schemaVersion"):
        load_requirements(path)


def test_later_3_x_minors_load():
    assert RequirementsDoc.from_dict({**_doc(), "schemaVersion": "3.7.2"}).id == "d"


def test_missing_stages_rejected():
    doc_dict = _doc()
    del doc_dict["stages"]
    with pytest.raises(RequirementsError, match="stages"):
        RequirementsDoc.from_dict(doc_dict)


def test_missing_id_rejected():
    doc_dict = _doc()
    del doc_dict["id"]
    with pytest.raises(RequirementsError, match="'id'"):
        RequirementsDoc.from_dict(doc_dict)


def _poc_stage(**overrides):
    return _stage(
        scenarios=[
            {"id": "chat", "sweep": [{"isl": 128, "osl": 128, "concurrency": 1}]}
        ],
        accuracyEvals=[{"name": "GPQA Diamond"}],
        **overrides,
    )


def test_single_stage_document_loads_from_its_stage():
    doc = RequirementsDoc.from_dict(
        {
            **_doc(deployment={"hardware": "SC8", "maxConcurrencyPerInstance": 32}),
            "stages": [_poc_stage(deployment={"maxConcurrencyPerInstance": 64})],
        }
    )
    assert [s.id for s in doc.scenarios] == ["chat"]
    assert [e.name for e in doc.accuracy_evals] == ["GPQA Diamond"]
    # A stage deployment is a partial override: set keys win, unset inherit.
    assert doc.deployment.hardware == "SC8"
    assert doc.deployment.max_concurrency_per_instance == 64


def _two_stage_doc(**perf_overrides):
    """Accuracy stage holds the evals; performance stage holds the scenarios."""
    return {
        **_doc(deployment={"hardware": "SC8", "maxConcurrencyPerInstance": 32}),
        "stages": [
            _stage(
                key="accuracy",
                name="Accuracy",
                accuracyEvals=[{"name": "GPQA Diamond"}],
            ),
            _stage(
                key="performance",
                name="Performance",
                scenarios=[
                    {
                        "id": "chat",
                        "sweep": [{"isl": 128, "osl": 128, "concurrency": 1}],
                    },
                    {
                        "id": "agent",
                        "kind": "agentic",
                        "agenticSweep": [{"concurrency": 1}],
                    },
                ],
                **perf_overrides,
            ),
        ],
    }


def test_every_stage_is_read_and_stamps_its_requirements():
    doc = RequirementsDoc.from_dict(_two_stage_doc())

    assert [(s.key, s.name, s.position) for s in doc.stages] == [
        ("accuracy", "Accuracy", 1),
        ("performance", "Performance", 2),
    ]
    assert [(e.name, e.stage) for e in doc.accuracy_evals] == [
        ("GPQA Diamond", "accuracy")
    ]
    assert [(s.id, s.stage) for s in doc.scenarios] == [("chat", "performance")]
    assert [(w.id, w.stage) for w in doc.agentic_workloads] == [
        ("agent", "performance")
    ]


def test_stages_must_share_one_deployment():
    with pytest.raises(RequirementsError, match="accuracy, performance"):
        RequirementsDoc.from_dict(
            _two_stage_doc(deployment={"maxConcurrencyPerInstance": 64})
        )


def test_a_validation_plan_stamps_stages_from_its_items():
    def stage(key):
        return {
            "key": key,
            "name": key.title(),
            "deployment": {"hardware": "BH-Galaxy"},
        }

    doc = RequirementsDoc.from_dict(
        {
            "schemaVersion": "3.1.0",
            "document": {"id": "d", "model": {"name": "a/b"}, "deployment": {}},
            "stages": [stage("accuracy"), stage("performance")],
            "workloads": [
                {"id": "chat", "kind": "text"},
                {
                    "id": "agent",
                    "kind": "agentic",
                    "agenticSweep": [{"concurrency": 1}],
                },
            ],
            "items": [
                {
                    "type": "accuracy_eval",
                    "stageKey": "accuracy",
                    "spec": {"name": "GPQA Diamond"},
                },
                {
                    "type": "operating_point",
                    "stageKey": "performance",
                    "scenarioId": "chat",
                    "targets": {"isl": 128, "osl": 128, "concurrency": 1},
                },
                {
                    "type": "agentic_operating_point",
                    "stageKey": "performance",
                    "scenarioId": "agent",
                    "targets": {"concurrency": 1},
                },
            ],
        }
    )

    assert [e.stage for e in doc.accuracy_evals] == ["accuracy"]
    assert [s.stage for s in doc.scenarios] == ["performance"]
    assert [w.stage for w in doc.agentic_workloads] == ["performance"]
    assert doc.deployment.hardware == "BH-Galaxy"


def test_nice_to_have_is_soft():
    doc = RequirementsDoc.from_dict(
        _doc(evals=[{"name": "X", "priority": "nice_to_have"}])
    )
    assert doc.accuracy_evals[0].priority == PRIORITY_SHOULD


def test_soft_metrics_are_read_per_scenario():
    doc = RequirementsDoc.from_dict(
        _doc(
            scenarios=[
                {"id": "chat", "softMetrics": ["ttftMeanMs"], "sweep": []},
                {
                    "id": "agent",
                    "kind": "agentic",
                    "softMetrics": ["e2elP95Ms"],
                    "agenticSweep": [{"concurrency": 1}],
                },
            ]
        )
    )
    assert doc.scenarios[0].soft_metrics == frozenset({"ttftMeanMs"})
    assert doc.agentic_workloads[0].soft_metrics == frozenset({"e2elP95Ms"})


@pytest.mark.parametrize(
    "row, isl, expected",
    [
        ({"inputThroughputTps": 5500}, 128, 5500.0),  # stated wins
        ({"reqThroughputRps": 10}, 128, 1280.0),  # fixed-length: ISL x RPS
        ({"totalThroughputTps": 1000, "decodeThroughputTps": 100}, None, 900.0),
        ({"totalThroughputTps": 100, "decodeThroughputTps": 100}, None, None),
        ({"reqThroughputRps": 10}, None, None),  # agentic needs total and output
        ({}, 128, None),
    ],
)
def test_input_throughput_follows_the_gauntlet_reader_rule(row, isl, expected):
    assert input_throughput_tps(row, isl=isl) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(250, 250.0), (0.5, 0.5), (0, None), (None, None), (True, None), ("1", None)],
)
def test_stated_target_treats_zero_as_blank(value, expected):
    assert stated_target({"ttftMeanMs": value}, "ttftMeanMs") == expected


def test_missing_schema_version_rejected(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"model": {"name": "a/b"}}))
    with pytest.raises(RequirementsError, match="schemaVersion"):
        load_requirements(path)


def test_missing_model_rejected(tmp_path):
    path = tmp_path / "doc.json"
    doc_dict = _doc()
    del doc_dict["model"]
    path.write_text(json.dumps(doc_dict))
    with pytest.raises(RequirementsError, match="model"):
        load_requirements(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(RequirementsError, match="not found"):
        load_requirements(tmp_path / "does-not-exist.json")


def test_invalid_priority_rejected(tmp_path):
    doc_dict = _doc(evals=[{"name": "X", "priority": "nice-to-have"}])
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    with pytest.raises(RequirementsError, match="priority"):
        load_requirements(path)


def test_invalid_comparator_rejected(tmp_path):
    doc_dict = _doc(
        scenarios=[
            {
                "id": "s",
                "scalarTargets": [
                    {"metric": "system_throughput", "target": 1, "comparator": "eq"}
                ],
            }
        ]
    )
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    with pytest.raises(RequirementsError, match="comparator"):
        load_requirements(path)


def _agentic_doc(**overrides):
    """A validation plan: identity under 'document', agentic sweep inline on its workload."""
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
        "schemaVersion": "3.1.0",
        "document": {
            "id": "doc-1",
            "meta": {"customer": "Ant"},
            "model": {"name": "google/gemma-4-31B-it", "contextLength": 131072},
            "deployment": {"hardware": "SC24", "maxConcurrencyPerInstance": 1},
        },
        "stages": [_stage(deployment={"hardware": "SC24"})],
        "workloads": [workload],
        "items": [],
    }


def _write(tmp_path, doc_dict):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc_dict))
    return path


def test_reads_identity_from_the_document_envelope(tmp_path):
    """A validation plan nests model/deployment/meta under 'document'."""
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


def _canonical_doc(*extra_scenarios):
    """A canonical document: every workload is a scenarios[] entry with a kind.

    This is the shape llm-gauntlet itself stores, with the scenarios on the
    document's single delivery stage.
    """
    return {
        "schemaVersion": "3.1.0",
        "id": "canon-1",
        "model": {"name": "moonshotai/Kimi-K2.7-Code", "contextLength": 262144},
        "deployment": {"hardware": "SC20", "maxConcurrencyPerInstance": 32},
        "stages": [
            _stage(
                scenarios=[
                    {
                        "kind": "text",
                        "id": "s-text",
                        "oslValues": [128],
                        "slo": {"ttftMs": 4100, "tpotMs": 22.2, "e2elMs": 10000},
                        "sweep": [{"isl": 128, "osl": 128, "concurrency": 1}],
                    },
                    {
                        "kind": "agentic",
                        "id": "s-agentic",
                        "name": "Claude Code trace replay",
                        "oslValues": [],
                        "slo": {"ttftMs": 1000, "tpotMs": 10, "e2elMs": 20000},
                        "maxConcurrency": 64,
                        "agenticWorkload": {
                            "traces": [{"name": "AgentX - Claude Code"}]
                        },
                        "agenticSweep": [
                            {"concurrency": 1, "e2elP90Ms": 16475.15},
                            {"concurrency": 64},
                        ],
                    },
                    *extra_scenarios,
                ]
            )
        ],
    }


def _scenarios(doc_dict):
    return doc_dict["stages"][0]["scenarios"]


def test_canonical_agentic_scenario_drives_the_sweep(tmp_path):
    """An agentic scenario in scenarios[] is an agentic workload, not a no-op."""
    doc = load_requirements(_write(tmp_path, _canonical_doc()))

    (workload,) = doc.agentic_workloads
    assert workload.id == "s-agentic"
    assert [p.concurrency for p in workload.sweep] == [1, 64]
    assert workload.max_concurrency == 64
    assert (workload.slo.ttft_ms, workload.slo.tpot_ms, workload.slo.e2el_ms) == (
        1000,
        10,
        20000,
    )
    assert [t["name"] for t in workload.traces] == ["AgentX - Claude Code"]
    assert workload.sweep[0].reference["e2elP90Ms"] == 16475.15


def test_canonical_agentic_scenario_is_not_also_a_benchmark_scenario(tmp_path):
    """It has a different home in the dataclass, so it must not be double-read.

    Left in ``scenarios`` it would be a benchmark scenario with an empty sweep.
    """
    doc = load_requirements(_write(tmp_path, _canonical_doc()))

    assert [s.id for s in doc.scenarios] == ["s-text"]


@pytest.mark.parametrize("kind", ["multimodal", "some-future-kind"])
def test_non_agentic_kinds_stay_benchmark_scenarios(tmp_path, kind):
    """Only ``agentic`` is split out; the adapter decides what it can run.

    Media kinds carry the same (ISL, OSL, concurrency) sweep, and an unknown
    future kind has to keep loading — tolerant parsing is this loader's policy.
    """
    doc_dict = _canonical_doc({"kind": kind, "id": "s-other", "oslValues": [128]})

    doc = load_requirements(_write(tmp_path, doc_dict))

    assert [s.id for s in doc.scenarios] == ["s-text", "s-other"]
    assert [s.kind for s in doc.scenarios] == ["text", kind]
    assert [w.id for w in doc.agentic_workloads] == ["s-agentic"]


def test_agentic_workload_needs_an_id(tmp_path):
    doc_dict = _canonical_doc()
    del _scenarios(doc_dict)[1]["id"]

    with pytest.raises(RequirementsError, match="agentic workload"):
        load_requirements(_write(tmp_path, doc_dict))


# --- per-row SLO overrides ---------------------------------------------------


def test_sweep_point_slo_is_typed_and_empty_means_absent(tmp_path):
    """An empty {} is how a serialized document spells "no SLOs declared"."""
    doc_dict = _canonical_doc()
    _scenarios(doc_dict)[0]["sweep"] = [
        {"isl": 128, "osl": 128, "concurrency": 1, "slo": {"ttftMs": 900}},
        {"isl": 128, "osl": 128, "concurrency": 32, "slo": {}},
        {"isl": 256, "osl": 128, "concurrency": 1},
    ]

    (scenario,) = load_requirements(_write(tmp_path, doc_dict)).scenarios

    assert scenario.sweep[0].slo == Slo(ttft_ms=900)
    assert scenario.sweep[1].slo is None
    assert scenario.sweep[2].slo is None
    # The raw row survives for provenance either way.
    assert scenario.sweep[0].reference["slo"] == {"ttftMs": 900}


def test_agentic_sweep_point_slo_is_typed(tmp_path):
    doc_dict = _canonical_doc()
    _scenarios(doc_dict)[1]["agenticSweep"] = [
        {"concurrency": 1, "slo": {"tpotMs": 5}},
        {"concurrency": 64},
    ]

    (workload,) = load_requirements(_write(tmp_path, doc_dict)).agentic_workloads

    assert workload.sweep[0].slo == Slo(tpot_ms=5)
    assert workload.sweep[1].slo is None


@pytest.mark.parametrize(
    "row, default, expected",
    [
        (None, None, None),
        (Slo(), None, None),
        (Slo(), Slo(), None),
        (None, Slo(ttft_ms=1), Slo(ttft_ms=1)),
        (Slo(ttft_ms=2), None, Slo(ttft_ms=2)),
        # A row override wins field by field...
        (
            Slo(ttft_ms=2, tpot_ms=3),
            Slo(ttft_ms=1, tpot_ms=1),
            Slo(ttft_ms=2, tpot_ms=3),
        ),
        # ...and a field the row leaves unset inherits, rather than clearing it.
        (
            Slo(ttft_ms=9000),
            Slo(ttft_ms=4100, tpot_ms=22.2, e2el_ms=10000),
            Slo(ttft_ms=9000, tpot_ms=22.2, e2el_ms=10000),
        ),
    ],
)
def test_effective_slo_merges_field_wise_row_wins(row, default, expected):
    """The document schema defines a row slo as a per-row override whose unset
    fields inherit the scenario default, so this merge is the contract rather
    than a convenience.
    """
    assert effective_slo(row, default) == expected


def test_effective_slo_none_keeps_unmeasurable_distinguishable():
    """All-None must collapse to None so callers can still say "no bars"."""
    assert effective_slo(Slo(ttft_ms=1), Slo()) == Slo(ttft_ms=1)


def test_sweep_point_effective_slo_helper(tmp_path):
    doc_dict = _canonical_doc()
    _scenarios(doc_dict)[0]["sweep"] = [
        {"isl": 128, "osl": 128, "concurrency": 1, "slo": {"ttftMs": 900}}
    ]

    (scenario,) = load_requirements(_write(tmp_path, doc_dict)).scenarios

    assert scenario.sweep[0].effective_slo(scenario.slo) == Slo(
        ttft_ms=900, tpot_ms=22.2, e2el_ms=10000
    )


# --- validation-plan exports (see _fold_validation_plan in requirements_schema.py) ---


def _plan(*, items, workloads):
    return {
        "schemaVersion": "3.1.0",
        "stages": [_stage(deployment={"hardware": "SC20"})],
        "document": {
            "id": "plan-1",
            "meta": {"customer": "Acme"},
            "model": {"name": "moonshotai/Kimi-K2.7-Code", "contextLength": 262144},
            "deployment": {"hardware": "SC20", "maxConcurrencyPerInstance": 32},
        },
        "workloads": workloads,
        "items": items,
    }


def _text_plan(**overrides):
    plan = _plan(
        workloads=[
            {
                "kind": "text",
                "id": "s-text",
                "name": "Agentic Coding Service",
                "oslValues": [128],
                "slo": {"ttftMs": 4100, "tpotMs": 22.2, "e2elMs": 10000},
                "maxConcurrency": 32,
            }
        ],
        items=[
            {
                "type": "operating_point",
                "scenarioId": "s-text",
                "scenarioName": "Agentic Coding Service",
                "kind": "text",
                "concurrency": c,
                "isl": 128,
                "osl": 128,
                "targets": {
                    "concurrency": c,
                    "isl": 128,
                    "osl": 128,
                    "ttftMeanMs": 4100,
                    "goodputPct": 90,
                },
                "slo": {"ttftMs": 4100} if c == 32 else {},
                "methodology": "Run `vllm bench serve` ...",
                "tools": ["vllm bench serve"],
            }
            for c in (1, 32)
        ]
        + [
            {
                "type": "accuracy_eval",
                "spec": {"name": "GPQA-Diamond", "gpuReferenceScore": 88.4},
                "acceptance": "≥ 83.98 %",
                "methodology": "Run the eval's published harness ...",
            }
        ],
    )
    plan.update(overrides)
    return plan


def test_export_sweep_rows_are_folded_back_onto_their_scenario(tmp_path):
    """The 'items' list is the scenario's sweep, expanded out. Put it back."""
    doc = load_requirements(_write(tmp_path, _text_plan()))

    (scenario,) = doc.scenarios
    assert scenario.id == "s-text"
    assert scenario.kind == "text"
    assert scenario.osl_values == [128]
    assert scenario.slo == Slo(ttft_ms=4100, tpot_ms=22.2, e2el_ms=10000)
    assert [(p.isl, p.osl, p.concurrency) for p in scenario.sweep] == [
        (128, 128, 1),
        (128, 128, 32),
    ]
    # Each item's targets object *is* the row, measurements included.
    assert scenario.sweep[0].reference["ttftMeanMs"] == 4100


def test_export_item_slo_becomes_the_rows_own_override(tmp_path):
    """An item's sibling slo is the effective per-row SLO the exporter merged."""
    doc = load_requirements(_write(tmp_path, _text_plan()))

    (scenario,) = doc.scenarios
    assert scenario.sweep[0].slo is None  # exported as {} => nothing declared
    assert scenario.sweep[1].slo == Slo(ttft_ms=4100)


def test_export_eval_specs_are_unwrapped(tmp_path):
    """accuracyEvals live one level deeper in an export, under items[].spec."""
    doc = load_requirements(_write(tmp_path, _text_plan()))

    assert [e.name for e in doc.accuracy_evals] == ["GPQA-Diamond"]
    assert doc.accuracy_evals[0].gpu_reference_score == 88.4


def test_export_agentic_points_come_from_the_inline_sweep(tmp_path):
    """An agentic workload keeps its sweep inline; agentic items are not re-read."""
    plan = _plan(
        workloads=[
            {
                "kind": "agentic",
                "id": "s-agentic",
                "agenticSweep": [{"concurrency": 7}],
            }
        ],
        items=[
            {
                "type": "agentic_operating_point",
                "scenarioId": "s-agentic",
                "concurrency": 99,
                "targets": {"concurrency": 99},
            }
        ],
    )

    doc = load_requirements(_write(tmp_path, plan))

    assert [p.concurrency for p in doc.agentic_workloads[0].sweep] == [7]


def test_export_folds_each_scenario_separately(tmp_path):
    """Items are regrouped by scenarioId, not poured into one bucket."""
    plan = _plan(
        workloads=[
            {"kind": "text", "id": "s-a", "oslValues": [128]},
            {"kind": "text", "id": "s-b", "oslValues": [128]},
        ],
        items=[
            {
                "type": "operating_point",
                "scenarioId": sid,
                "targets": {"concurrency": c, "isl": 128, "osl": 128},
            }
            for sid, c in (("s-a", 1), ("s-b", 8), ("s-b", 16))
        ],
    )

    doc = load_requirements(_write(tmp_path, plan))

    by_id = {s.id: [p.concurrency for p in s.sweep] for s in doc.scenarios}
    assert by_id == {"s-a": [1], "s-b": [8, 16]}


def test_documents_without_items_take_no_detour(tmp_path):
    """A canonical document must be unaffected by the fold."""
    doc = load_requirements(_write(tmp_path, _canonical_doc()))

    assert [s.id for s in doc.scenarios] == ["s-text"]
    assert [w.id for w in doc.agentic_workloads] == ["s-agentic"]

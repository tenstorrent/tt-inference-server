# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Engine-owned loader for the LLM-serving requirements document.

A requirements document (``schemaVersion`` 3.1 or later within major 3, e.g.
``acme-llm-serving.json``) is the vendor-neutral input that drives a validation
run: which accuracy evals
to run and the reference scores that gate them, which benchmark sweep points to
execute and the scalar targets / SLOs to compare against, plus enough model and
deployment metadata to run a model that is not in the built-in catalog.

The format is llm-gauntlet business, not Tenstorrent business, so this
loader lives engine-side and produces plain, adapter-agnostic dataclasses. The
Tenstorrent adapter (``workflows/requirements_target_pack.py``) maps these onto
``reference_config`` types via the :class:`~workflow_module.target_pack.TargetPack`
seam.

Two shapes load: the canonical document (identity at the root, requirements on
its ``stages[]``) and its validation-plan export (identity under ``document``,
operating points in ``items[]``). Unknown keys are ignored, so newer 3.x minors
still load; anything older than 3.1 is a hard error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Oldest ``schemaVersion`` this loader reads, and the major it stays within.
# 3.1 is where input throughput became an always-hard requirement.
MIN_SCHEMA_VERSION = (3, 1)

# Scenario ``kind`` discriminator. A canonical document keeps every workload in
# ``scenarios[]`` and tells them apart by this field; the agentic one is split
# out into :attr:`RequirementsDoc.agentic_workloads` because it sweeps
# concurrency alone and drives a different workflow.
AGENTIC_KIND = "agentic"
DEFAULT_SCENARIO_KIND = "text"

# Accepted priority values. ``must`` failures block acceptance; ``should``
# failures are informational (see report_module/acceptance_criteria.py).
PRIORITY_MUST = "must"
PRIORITY_SHOULD = "should"
# llm-gauntlet's third priority: advisory, like ``should``.
PRIORITY_NICE_TO_HAVE = "nice_to_have"
_VALID_PRIORITIES = frozenset({PRIORITY_MUST, PRIORITY_SHOULD, PRIORITY_NICE_TO_HAVE})


class RequirementsError(ValueError):
    """Raised when a requirements document cannot be parsed or is unsupported."""


def _normalize_priority(value: Any, *, where: str) -> str:
    """Coerce a raw priority to ``must``/``should`` (defaults to ``must``)."""
    if value is None:
        return PRIORITY_MUST
    priority = str(value).strip().lower()
    if priority not in _VALID_PRIORITIES:
        raise RequirementsError(
            f"{where}: priority must be one of {sorted(_VALID_PRIORITIES)}, "
            f"got {value!r}"
        )
    return PRIORITY_SHOULD if priority == PRIORITY_NICE_TO_HAVE else priority


def stated_target(row: Mapping[str, Any], key: str) -> Optional[float]:
    """``row[key]`` as a target, or None when it states none.

    Only a positive number is a target: llm-gauntlet writes a blank (soft) column
    as 0, and no latency, rate or percentage target is ever 0.
    """
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def input_throughput_tps(
    row: Mapping[str, Any], *, isl: Optional[int] = None
) -> Optional[float]:
    """A row's input-throughput target, read the way llm-gauntlet reads it.

    The stated ``inputThroughputTps``, else derived: ``ISL x RPS`` for a
    fixed-length row (pass ``isl``), ``total - output`` for an agentic row.
    """
    if row.get("inputThroughputTps") is not None:
        return stated_target(row, "inputThroughputTps")
    if isl is not None:
        rps = stated_target(row, "reqThroughputRps")
        return isl * rps if rps is not None else None
    total = stated_target(row, "totalThroughputTps")
    output = stated_target(row, "decodeThroughputTps")
    if total is None or output is None or total <= output:
        return None
    return total - output


def _soft_metrics(data: Mapping[str, Any]) -> frozenset:
    """A scenario's ``softMetrics``: columns reported but never asserted."""
    raw = data.get("softMetrics")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return frozenset()
    return frozenset(str(key) for key in raw)


@dataclass(frozen=True)
class EvalGenKwargs:
    """Generation parameters one accuracy eval must be measured under.

    A closed set, mirroring the document's own ``genKwargs``: a reference score
    is only a bar if the graded run samples the way the reference run did, and
    a misspelled parameter is a score measured under settings nobody chose.
    ``None`` is the document saying nothing about that parameter, which leaves
    whatever the harness already had for it.

    The field names are the document's, transliterated to snake_case. They
    coincide with what lm-eval calls these parameters because the document
    names them from that vocabulary; mapping them onto a harness is still the
    adapter's job, not this loader's.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_gen_toks: Optional[int] = None
    reasoning_effort: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalGenKwargs":
        return cls(
            temperature=_as_optional_float(data.get("temperature")),
            top_p=_as_optional_float(data.get("topP")),
            top_k=_as_optional_int(data.get("topK")),
            max_gen_toks=_as_optional_int(data.get("maxGenToks")),
            reasoning_effort=(
                str(data["reasoningEffort"])
                if data.get("reasoningEffort") is not None
                else None
            ),
        )

    def stated(self) -> Dict[str, Any]:
        """Only the parameters the document actually set, in field order."""
        return {
            name: value
            for name, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("top_k", self.top_k),
                ("max_gen_toks", self.max_gen_toks),
                ("reasoning_effort", self.reasoning_effort),
            )
            if value is not None
        }


@dataclass(frozen=True)
class AccuracyEval:
    """One accuracy benchmark to run, with the score reference that gates it."""

    name: str
    task_category: Optional[str] = None
    gpu_reference_score: Optional[float] = None
    published_score: Optional[float] = None
    published_score_url: Optional[str] = None
    tolerance: float = 0.05
    priority: str = PRIORITY_MUST
    unit: str = "%"
    gen_kwargs: Optional[EvalGenKwargs] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccuracyEval":
        name = data.get("name")
        if not name:
            raise RequirementsError("accuracyEvals[]: missing required 'name'")
        raw_gen_kwargs = data.get("genKwargs")
        gen_kwargs = (
            EvalGenKwargs.from_dict(raw_gen_kwargs)
            if isinstance(raw_gen_kwargs, Mapping)
            else None
        )
        return cls(
            name=str(name),
            task_category=data.get("taskCategory"),
            gpu_reference_score=_as_optional_float(data.get("gpuReferenceScore")),
            published_score=_as_optional_float(data.get("publishedScore")),
            published_score_url=data.get("publishedScoreUrl"),
            tolerance=_as_float(data.get("tolerance"), default=0.05),
            priority=_normalize_priority(
                data.get("priority"), where=f"accuracyEvals[{name!r}]"
            ),
            unit=str(data.get("unit", "%")),
            gen_kwargs=gen_kwargs if gen_kwargs and gen_kwargs.stated() else None,
        )


@dataclass(frozen=True)
class ScalarTarget:
    """A single scalar acceptance target for a benchmark scenario."""

    metric: str
    target: float
    comparator: str = "gte"
    statistic: str = "mean"
    unit: Optional[str] = None
    priority: str = PRIORITY_MUST

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScalarTarget":
        metric = data.get("metric")
        if not metric:
            raise RequirementsError("scalarTargets[]: missing required 'metric'")
        target = _as_optional_float(data.get("target"))
        if target is None:
            raise RequirementsError(
                f"scalarTargets[{metric!r}]: missing/invalid required 'target'"
            )
        comparator = str(data.get("comparator", "gte")).lower()
        if comparator not in ("gte", "lte"):
            raise RequirementsError(
                f"scalarTargets[{metric!r}]: comparator must be 'gte' or 'lte', "
                f"got {data.get('comparator')!r}"
            )
        return cls(
            metric=str(metric),
            target=target,
            comparator=comparator,
            statistic=str(data.get("statistic", "mean")),
            unit=data.get("unit"),
            priority=_normalize_priority(
                data.get("priority"), where=f"scalarTargets[{metric!r}]"
            ),
        )


@dataclass(frozen=True)
class Slo:
    """Per-request service-level objectives (all in ms).

    Declared by a scenario/workload as its default, and optionally overridden
    per sweep row. An unset field means "no bar for this metric", which is why
    :meth:`merged_over` inherits rather than treating ``None`` as a value.
    """

    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2el_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["Slo"]:
        # An empty ``{}`` is indistinguishable from an absent key, and must
        # stay that way: the upstream schema defaults a scenario's slo to
        # ``{}``, so serialized documents routinely carry one, and it means
        # "no SLOs declared" rather than "all bars present but unset".
        if not data:
            return None
        return cls(
            ttft_ms=_as_optional_float(data.get("ttftMs")),
            tpot_ms=_as_optional_float(data.get("tpotMs")),
            e2el_ms=_as_optional_float(data.get("e2elMs")),
        )

    def merged_over(self, default: Optional["Slo"]) -> "Slo":
        """This SLO layered over ``default``: a set field wins, unset inherits."""
        if default is None:
            return self
        return Slo(
            ttft_ms=self.ttft_ms if self.ttft_ms is not None else default.ttft_ms,
            tpot_ms=self.tpot_ms if self.tpot_ms is not None else default.tpot_ms,
            e2el_ms=self.e2el_ms if self.e2el_ms is not None else default.e2el_ms,
        )


def effective_slo(row: Optional[Slo], default: Optional[Slo]) -> Optional[Slo]:
    """The SLOs in force for one sweep row: its own layered over the default.

    Field-wise merge, row wins, unset inherits from default (mirrors
    ``effectiveSlo`` in llm-gauntlet's schema package). Returns ``None`` when
    nothing is declared either side.
    """
    merged = row.merged_over(default) if row is not None else default
    if merged is None or (
        merged.ttft_ms is None and merged.tpot_ms is None and merged.e2el_ms is None
    ):
        return None
    return merged


@dataclass(frozen=True)
class SweepPoint:
    """One (ISL, OSL, concurrency) point in a benchmark sweep.

    Only the fields the engine consumes to *drive* a run (isl/osl/concurrency)
    and the optional per-row SLO override are typed; the remaining reference
    measurements from the document are kept verbatim in :attr:`reference` for
    display/provenance.
    """

    isl: int
    osl: int
    concurrency: int
    reference: Mapping[str, Any] = field(default_factory=dict)
    slo: Optional[Slo] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepPoint":
        for key in ("isl", "osl", "concurrency"):
            if data.get(key) is None:
                raise RequirementsError(f"sweep[]: missing required {key!r}")
        return cls(
            isl=int(data["isl"]),
            osl=int(data["osl"]),
            concurrency=int(data["concurrency"]),
            reference=dict(data),
            slo=Slo.from_dict(data.get("slo")),
        )

    def effective_slo(self, default: Optional[Slo]) -> Optional[Slo]:
        """SLOs in force for this point, its own override beating ``default``."""
        return effective_slo(self.slo, default)


@dataclass(frozen=True)
class Scenario:
    """A benchmark scenario: a sweep plus its scalar targets and SLOs."""

    id: str
    name: Optional[str] = None
    kind: str = "text"
    description: Optional[str] = None
    osl_values: List[int] = field(default_factory=list)
    scalar_targets: List[ScalarTarget] = field(default_factory=list)
    slo: Optional[Slo] = None
    sweep: List[SweepPoint] = field(default_factory=list)
    soft_metrics: frozenset = frozenset()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        scenario_id = data.get("id")
        if not scenario_id:
            raise RequirementsError("scenarios[]: missing required 'id'")
        return cls(
            id=str(scenario_id),
            name=data.get("name"),
            kind=str(data.get("kind", "text")),
            description=data.get("description"),
            osl_values=[int(v) for v in data.get("oslValues", [])],
            scalar_targets=[
                ScalarTarget.from_dict(t) for t in data.get("scalarTargets", [])
            ],
            slo=Slo.from_dict(data.get("slo")),
            sweep=[SweepPoint.from_dict(p) for p in data.get("sweep", [])],
            soft_metrics=_soft_metrics(data),
        )


@dataclass(frozen=True)
class AgenticSweepPoint:
    """One concurrency point in an agentic trace-replay sweep.

    Only :attr:`concurrency` drives the run. The document's expected
    measurements for the point are kept verbatim in :attr:`reference` so what
    we measure can be graded against them, the same way :class:`SweepPoint`
    carries a benchmark point's references.
    """

    concurrency: int
    reference: Mapping[str, Any] = field(default_factory=dict)
    slo: Optional[Slo] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgenticSweepPoint":
        if data.get("concurrency") is None:
            raise RequirementsError("agenticSweep[]: missing required 'concurrency'")
        return cls(
            concurrency=int(data["concurrency"]),
            reference=dict(data),
            slo=Slo.from_dict(data.get("slo")),
        )

    def effective_slo(self, default: Optional[Slo]) -> Optional[Slo]:
        """SLOs in force for this point, its own override beating ``default``."""
        return effective_slo(self.slo, default)


@dataclass(frozen=True)
class AgenticWorkload:
    """An agentic trace-replay workload: a concurrency sweep plus its SLOs.

    The agentic counterpart to :class:`Scenario`. It sweeps concurrency alone
    rather than (ISL, OSL, concurrency), because the prompt sizes come from the
    replayed traces instead of the document.
    """

    id: str
    name: Optional[str] = None
    slo: Optional[Slo] = None
    sweep: List[AgenticSweepPoint] = field(default_factory=list)
    max_concurrency: Optional[int] = None
    traces: List[Mapping[str, Any]] = field(default_factory=list)
    soft_metrics: frozenset = frozenset()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgenticWorkload":
        workload_id = data.get("id")
        if not workload_id:
            raise RequirementsError("agentic workload: missing required 'id'")
        agentic = data.get("agenticWorkload")
        traces = agentic.get("traces", []) if isinstance(agentic, Mapping) else []
        return cls(
            id=str(workload_id),
            name=data.get("name"),
            slo=Slo.from_dict(data.get("slo")),
            sweep=[
                AgenticSweepPoint.from_dict(p) for p in data.get("agenticSweep", [])
            ],
            max_concurrency=_as_optional_int(data.get("maxConcurrency")),
            traces=[dict(t) for t in traces if isinstance(t, Mapping)],
            soft_metrics=_soft_metrics(data),
        )


@dataclass(frozen=True)
class ModelInfo:
    """Model identity from the requirements document."""

    name: str
    context_length: Optional[int] = None
    repo_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInfo":
        name = data.get("name")
        if not name:
            raise RequirementsError("model: missing required 'name'")
        return cls(
            name=str(name),
            context_length=_as_optional_int(data.get("contextLength")),
            repo_url=data.get("repoUrl"),
        )


@dataclass(frozen=True)
class Deployment:
    """Target deployment shape (hardware + concurrency)."""

    hardware: Optional[str] = None
    environment: Optional[str] = None
    max_concurrency_per_instance: Optional[int] = None
    max_instances: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "Deployment":
        data = data or {}
        return cls(
            hardware=data.get("hardware"),
            environment=data.get("environment"),
            max_concurrency_per_instance=_as_optional_int(
                data.get("maxConcurrencyPerInstance")
            ),
            max_instances=_as_optional_int(data.get("maxInstances")),
        )


def _scenario_kind(entry: Mapping[str, Any]) -> str:
    """The ``kind`` of a scenario/workload entry, defaulted like the schema."""
    return str(entry.get("kind", DEFAULT_SCENARIO_KIND))


def _lift_single_stage(
    data: Mapping[str, Any],
) -> "tuple[Mapping[str, Any], Mapping[str, Any]]":
    """The document's one delivery stage, with its requirements lifted to the root.

    Returns the document and the stage's deployment override (``{}`` if none).
    A canonical stage carries its own ``scenarios``/``accuracyEvals``; a
    validation-plan stage carries none (its points are in ``items``) and states
    the effective deployment. One run validates one stage, so several are
    rejected rather than merged.
    """
    stages = data.get("stages")
    if (
        not isinstance(stages, Sequence)
        or isinstance(stages, (str, bytes))
        or not stages
    ):
        raise RequirementsError("requirements: missing required 'stages'")
    if len(stages) > 1:
        keys = ", ".join(
            str(s.get("key") or s.get("name")) for s in stages if isinstance(s, Mapping)
        )
        raise RequirementsError(
            f"requirements: {len(stages)} delivery stages ({keys}); a run "
            "validates a single-stage document"
        )
    stage = stages[0] if isinstance(stages[0], Mapping) else {}
    lifted = dict(data)
    for key in ("scenarios", "accuracyEvals"):
        if key in stage:
            lifted[key] = stage[key]
    deployment = stage.get("deployment")
    return lifted, deployment if isinstance(deployment, Mapping) else {}


def _fold_validation_plan(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fold a validation-plan export back into the canonical document shape.

    Each ``workloads[]`` entry becomes a scenario. A fixed-length workload has
    its sweep stripped; its rows are the ``targets`` of its ``operating_point``
    items, with the item's resolved ``slo``. An agentic workload keeps its
    ``agenticSweep`` inline. Eval specs come from the ``accuracy_eval`` items.
    A document with no ``items`` is the canonical shape, returned untouched.
    """
    items = data.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return data

    sweeps: Dict[str, List[Mapping[str, Any]]] = {}
    evals: List[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        targets = item.get("targets")
        if kind == "accuracy_eval":
            spec = item.get("spec")
            if isinstance(spec, Mapping):
                evals.append(spec)
        elif kind == "operating_point" and isinstance(targets, Mapping):
            row = dict(targets)
            if item.get("slo"):
                row["slo"] = item["slo"]
            sweeps.setdefault(str(item.get("scenarioId", "")), []).append(row)

    scenarios: List[Mapping[str, Any]] = []
    for workload in data.get("workloads", []):
        if not isinstance(workload, Mapping):
            continue
        entry = dict(workload)
        if _scenario_kind(entry) != AGENTIC_KIND:
            entry["sweep"] = sweeps.get(str(entry.get("id", "")), [])
        scenarios.append(entry)
    return {**data, "scenarios": scenarios, "accuracyEvals": evals}


@dataclass(frozen=True)
class RequirementsDoc:
    """Parsed LLM-serving requirements document."""

    id: str
    schema_version: str
    model: ModelInfo
    deployment: Deployment
    accuracy_evals: List[AccuracyEval] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    agentic_workloads: List[AgenticWorkload] = field(default_factory=list)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequirementsDoc":
        schema_version = str(data.get("schemaVersion", ""))
        _check_schema_version(schema_version)
        data, stage_deployment = _lift_single_stage(data)
        # A validation-plan export carries its sweeps and eval specs in a flat
        # top-level "items" list rather than on the workloads; fold them back
        # so both it and the canonical document load identically.
        data = _fold_validation_plan(data)
        # The validation plan wraps identity (model, deployment, meta) in a
        # "document" envelope; the canonical document keeps it at the root.
        envelope = data.get("document")
        identity = envelope if isinstance(envelope, Mapping) else data
        model_data = identity.get("model")
        if not isinstance(model_data, Mapping):
            raise RequirementsError("requirements: missing required 'model' object")
        if not identity.get("id"):
            raise RequirementsError("requirements: missing required 'id'")
        raw_scenarios = [s for s in data.get("scenarios", []) if isinstance(s, Mapping)]
        # A stage's deployment is a partial override: set keys win, unset inherit.
        deployment = {
            **(identity.get("deployment") or {}),
            **{k: v for k, v in stage_deployment.items() if v is not None},
        }
        return cls(
            id=str(identity["id"]),
            schema_version=schema_version,
            model=ModelInfo.from_dict(model_data),
            deployment=Deployment.from_dict(deployment),
            accuracy_evals=[
                AccuracyEval.from_dict(e) for e in data.get("accuracyEvals", [])
            ],
            # Only the agentic kind is split out; other kinds stay here and are
            # skipped downstream by the adapter, not here.
            scenarios=[
                Scenario.from_dict(s)
                for s in raw_scenarios
                if _scenario_kind(s) != AGENTIC_KIND
            ],
            agentic_workloads=[
                AgenticWorkload.from_dict(s)
                for s in raw_scenarios
                if _scenario_kind(s) == AGENTIC_KIND
            ],
            meta=dict(identity.get("meta", {})),
        )


def _resolve_requirements_path(path: Union[str, Path]) -> Path:
    """Resolve ``path`` to a canonical absolute file, rejecting traversal.

    The path comes from the ``--requirements-json`` CLI flag (operator input).
    Canonicalizing with :meth:`Path.resolve` collapses ``..`` segments and
    follows symlinks, and requiring the result to be a regular file under an
    existing directory guards against accidental (or malicious) references to
    unexpected locations. This is a local dev CLI, so no fixed base directory
    is imposed — operators legitimately keep requirements docs anywhere — but
    the resolved path must exist as a real file.
    """
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise RequirementsError(f"Invalid requirements path {path!r}: {e}") from e
    if not resolved.exists():
        raise RequirementsError(f"Requirements file not found: {resolved}")
    if not resolved.is_file():
        raise RequirementsError(f"Requirements path is not a file: {resolved}")
    return resolved


def load_requirements(path: Union[str, Path]) -> RequirementsDoc:
    """Load and parse a requirements document from ``path``.

    Raises :class:`RequirementsError` on a missing file, invalid JSON, or an
    unsupported ``schemaVersion`` major.
    """
    json_path = _resolve_requirements_path(path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RequirementsError(f"Invalid JSON in {json_path}: {e}") from e
    if not isinstance(data, Mapping):
        raise RequirementsError(
            f"Requirements document must be a JSON object, got {type(data).__name__}"
        )
    doc = RequirementsDoc.from_dict(data)
    logger.info(
        "Loaded requirements id=%s revision=%s model=%s hardware=%s "
        "(%d evals, %d benchmark scenarios, %d agentic workloads)",
        doc.id,
        doc.meta.get("revision"),
        doc.model.name,
        doc.deployment.hardware,
        len(doc.accuracy_evals),
        len(doc.scenarios),
        len(doc.agentic_workloads),
    )
    return doc


def _check_schema_version(schema_version: str) -> None:
    if not schema_version:
        raise RequirementsError("requirements: missing required 'schemaVersion'")
    try:
        major, minor = (int(part) for part in schema_version.split(".")[:2])
    except ValueError as e:
        raise RequirementsError(
            f"requirements: unparseable schemaVersion {schema_version!r}"
        ) from e
    min_major, min_minor = MIN_SCHEMA_VERSION
    if major != min_major or minor < min_minor:
        raise RequirementsError(
            f"Unsupported schemaVersion {schema_version!r}: this loader supports "
            f"{min_major}.{min_minor} and later {min_major}.x; re-export the "
            "document from llm-gauntlet"
        )


def _as_float(value: Any, *, default: float) -> float:
    result = _as_optional_float(value)
    return default if result is None else result


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    result = _as_optional_float(value)
    return None if result is None else int(result)


__all__ = [
    "MIN_SCHEMA_VERSION",
    "AGENTIC_KIND",
    "DEFAULT_SCENARIO_KIND",
    "PRIORITY_MUST",
    "PRIORITY_SHOULD",
    "RequirementsError",
    "input_throughput_tps",
    "stated_target",
    "AccuracyEval",
    "EvalGenKwargs",
    "ScalarTarget",
    "Slo",
    "effective_slo",
    "SweepPoint",
    "Scenario",
    "AgenticSweepPoint",
    "AgenticWorkload",
    "ModelInfo",
    "Deployment",
    "RequirementsDoc",
    "load_requirements",
]

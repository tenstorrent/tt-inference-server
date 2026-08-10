# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Engine-owned loader for the Blaze customer-requirements document.

A requirements document (``schemaVersion`` 2.x, e.g. ``acme-llm-serving.json``)
is the vendor-neutral input that drives a validation run: which accuracy evals
to run and the reference scores that gate them, which benchmark sweep points to
execute and the scalar targets / SLOs to compare against, plus enough model and
deployment metadata to run a model that is not in the built-in catalog.

The format is Blaze/llm-gauntlet business, not Tenstorrent business, so this
loader lives engine-side and produces plain, adapter-agnostic dataclasses. The
Tenstorrent adapter (``workflows/requirements_target_pack.py``) maps these onto
``reference_config`` types via the :class:`~workflow_module.target_pack.TargetPack`
seam.

Parsing is deliberately tolerant: unknown keys are ignored (so newer minor
revisions of the schema still load), but an unsupported ``schemaVersion`` major
is a hard error since the shapes below are version-specific.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

logger = logging.getLogger(__name__)

# Major version of ``schemaVersion`` this loader understands. A document whose
# major differs is rejected rather than silently mis-parsed.
SUPPORTED_SCHEMA_MAJOR = 2

# Accepted priority values. ``must`` failures block acceptance; ``should``
# failures are informational (see report_module/acceptance_criteria.py).
PRIORITY_MUST = "must"
PRIORITY_SHOULD = "should"
_VALID_PRIORITIES = frozenset({PRIORITY_MUST, PRIORITY_SHOULD})


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
    return priority


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AccuracyEval":
        name = data.get("name")
        if not name:
            raise RequirementsError("accuracyEvals[]: missing required 'name'")
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
    """Per-request service-level objectives for a scenario (all in ms)."""

    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2el_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> Optional["Slo"]:
        if not data:
            return None
        return cls(
            ttft_ms=_as_optional_float(data.get("ttftMs")),
            tpot_ms=_as_optional_float(data.get("tpotMs")),
            e2el_ms=_as_optional_float(data.get("e2elMs")),
        )


@dataclass(frozen=True)
class SweepPoint:
    """One (ISL, OSL, concurrency) point in a benchmark sweep.

    Only the fields the engine consumes to *drive* a run (isl/osl/concurrency)
    are typed; the remaining reference measurements from the document are kept
    verbatim in :attr:`reference` for display/provenance.
    """

    isl: int
    osl: int
    concurrency: int
    reference: Mapping[str, Any] = field(default_factory=dict)

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
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        scenario_id = data.get("id") or data.get("name")
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


@dataclass(frozen=True)
class RequirementsDoc:
    """Parsed Blaze requirements document."""

    id: str
    schema_version: str
    model: ModelInfo
    deployment: Deployment
    accuracy_evals: List[AccuracyEval] = field(default_factory=list)
    scenarios: List[Scenario] = field(default_factory=list)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequirementsDoc":
        schema_version = str(data.get("schemaVersion", ""))
        _check_schema_version(schema_version)
        model_data = data.get("model")
        if not isinstance(model_data, Mapping):
            raise RequirementsError("requirements: missing required 'model' object")
        return cls(
            id=str(
                data.get("id") or data.get("model", {}).get("name") or "requirements"
            ),
            schema_version=schema_version,
            model=ModelInfo.from_dict(model_data),
            deployment=Deployment.from_dict(data.get("deployment")),
            accuracy_evals=[
                AccuracyEval.from_dict(e) for e in data.get("accuracyEvals", [])
            ],
            scenarios=[Scenario.from_dict(s) for s in data.get("scenarios", [])],
            meta=dict(data.get("meta", {})),
        )


def load_requirements(path: Union[str, Path]) -> RequirementsDoc:
    """Load and parse a requirements document from ``path``.

    Raises :class:`RequirementsError` on a missing file, invalid JSON, or an
    unsupported ``schemaVersion`` major.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise RequirementsError(f"Requirements file not found: {json_path}")
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
        "Loaded requirements id=%s model=%s hardware=%s (%d evals, %d scenarios)",
        doc.id,
        doc.model.name,
        doc.deployment.hardware,
        len(doc.accuracy_evals),
        len(doc.scenarios),
    )
    return doc


def _check_schema_version(schema_version: str) -> None:
    if not schema_version:
        raise RequirementsError("requirements: missing required 'schemaVersion'")
    major_str = schema_version.split(".", 1)[0]
    try:
        major = int(major_str)
    except ValueError as e:
        raise RequirementsError(
            f"requirements: unparseable schemaVersion {schema_version!r}"
        ) from e
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise RequirementsError(
            f"Unsupported schemaVersion {schema_version!r}: this loader supports "
            f"major version {SUPPORTED_SCHEMA_MAJOR}.x"
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
    "SUPPORTED_SCHEMA_MAJOR",
    "PRIORITY_MUST",
    "PRIORITY_SHOULD",
    "RequirementsError",
    "AccuracyEval",
    "ScalarTarget",
    "Slo",
    "SweepPoint",
    "Scenario",
    "ModelInfo",
    "Deployment",
    "RequirementsDoc",
    "load_requirements",
]

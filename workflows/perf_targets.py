#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes

if TYPE_CHECKING:
    from workflows.model_spec import AcceptanceCriteria

logger = logging.getLogger(__name__)
RELEASE_PERFORMANCE_SCHEMA_VERSION = "0.1.0"

DEFAULT_PERF_TARGETS_MAP: Dict[str, float] = {
    "functional": 0.10,
    "complete": 0.50,
    "target": 1.0,
}
REGRESSION_TARGET_NAME = "regression"
DEFAULT_REGRESSION_TOLERANCE = 0.05
_MISSING_IDENTITY_VALUES = {"", "N/A", "n/a"}
_INT_IDENTITY_FIELDS = {
    "isl",
    "osl",
    "max_concurrency",
    "num_eval_runs",
    "num_inference_steps",
    "image_height",
    "image_width",
    "images_per_prompt",
}
_STRICT_TEXT_IDENTITY_FIELDS = ("isl", "osl", "max_concurrency")
_STRICT_VLM_IDENTITY_FIELDS = (
    "isl",
    "osl",
    "max_concurrency",
    "image_height",
    "image_width",
    "images_per_prompt",
)
_OPTIONAL_IMAGE_IDENTITY_FIELDS = ("max_concurrency", "num_inference_steps")
_OPTIONAL_AUDIO_IDENTITY_FIELDS = ("max_concurrency", "num_eval_runs")
_OPTIONAL_EMBEDDING_IDENTITY_FIELDS = ("max_concurrency",)


def _resolve_perf_targets_map(
    perf_targets_map: Optional[Dict[str, float]],
    acceptance_criteria: Optional["AcceptanceCriteria"] = None,
) -> Dict[str, float]:
    if acceptance_criteria is not None and hasattr(
        acceptance_criteria, "resolved_perf_targets_map"
    ):
        return acceptance_criteria.resolved_perf_targets_map(perf_targets_map)
    return dict(perf_targets_map or DEFAULT_PERF_TARGETS_MAP)


def _resolve_regression_tolerance(
    acceptance_criteria: Optional["AcceptanceCriteria"] = None,
) -> float:
    regression_check = (
        getattr(acceptance_criteria, "regression_check", None)
        if acceptance_criteria is not None
        else None
    )
    tolerance = (
        getattr(regression_check, "tolerance", None)
        if regression_check is not None
        else None
    )
    return float(tolerance) if tolerance is not None else DEFAULT_REGRESSION_TOLERANCE


def _normalize_identity_value(field_name: str, value: Any) -> Any:
    if value is None or value in _MISSING_IDENTITY_VALUES:
        return None
    if field_name in _INT_IDENTITY_FIELDS:
        try:
            value = int(value)
        except (TypeError, ValueError):
            return value
        if value == 0:
            return None
    return value


def _measurement_identity_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        "task_type": row.get("task_type", "text"),
        "isl": _normalize_identity_value(
            "isl", row.get("isl", row.get("input_sequence_length", row.get("ISL")))
        ),
        "osl": _normalize_identity_value(
            "osl", row.get("osl", row.get("output_sequence_length"))
        ),
        "max_concurrency": _normalize_identity_value(
            "max_concurrency",
            row.get(
                "max_concurrency",
                row.get("max_con", row.get("concurrency")),
            ),
        ),
        "num_inference_steps": _normalize_identity_value(
            "num_inference_steps", row.get("num_inference_steps")
        ),
        "num_eval_runs": _normalize_identity_value(
            "num_eval_runs",
            row.get("num_eval_runs", row.get("num_requests")),
        ),
        "image_height": _normalize_identity_value(
            "image_height", row.get("image_height")
        ),
        "image_width": _normalize_identity_value("image_width", row.get("image_width")),
        "images_per_prompt": _normalize_identity_value(
            "images_per_prompt",
            row.get("images_per_prompt", row.get("images")),
        ),
    }


def _strict_identity_field_matches(
    reference_identity: Dict[str, Any],
    measurement_identity: Dict[str, Any],
    field_name: str,
) -> bool:
    return reference_identity.get(field_name) == measurement_identity.get(field_name)


def _optional_identity_field_matches(
    reference_identity: Dict[str, Any],
    measurement_identity: Dict[str, Any],
    field_name: str,
) -> bool:
    reference_value = reference_identity.get(field_name)
    measurement_value = measurement_identity.get(field_name)
    if reference_value is None or measurement_value is None:
        return True
    return reference_value == measurement_value


def _reference_wildcard_field_matches(
    reference_identity: Dict[str, Any],
    measurement_identity: Dict[str, Any],
    field_name: str,
) -> bool:
    reference_value = reference_identity.get(field_name)
    if reference_value is None:
        return True
    return reference_value == measurement_identity.get(field_name)


def benchmark_identities_match(
    reference_identity: Dict[str, Any], measurement_row: Dict[str, Any]
) -> bool:
    measurement_identity = _measurement_identity_from_row(measurement_row)
    if not measurement_identity:
        return False

    reference_task_type = reference_identity.get(
        "task_type"
    ) or measurement_identity.get("task_type", "text")
    measurement_task_type = measurement_identity.get("task_type", reference_task_type)
    if reference_task_type != measurement_task_type:
        return False

    if reference_task_type == "text":
        if not all(
            _strict_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in _STRICT_TEXT_IDENTITY_FIELDS
        ):
            return False
        return all(
            _optional_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in ("num_eval_runs", "num_inference_steps")
        )

    if reference_task_type == "vlm":
        if not all(
            _strict_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in _STRICT_VLM_IDENTITY_FIELDS
        ):
            return False
        return all(
            _optional_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in ("num_eval_runs", "num_inference_steps")
        )

    if reference_task_type in {"image", "cnn", "video"}:
        return all(
            _optional_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in _OPTIONAL_IMAGE_IDENTITY_FIELDS
        )

    if reference_task_type == "audio":
        return all(
            _optional_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in _OPTIONAL_AUDIO_IDENTITY_FIELDS
        )

    if reference_task_type == "embedding":
        if not _reference_wildcard_field_matches(
            reference_identity, measurement_identity, "isl"
        ):
            return False
        if not _reference_wildcard_field_matches(
            reference_identity, measurement_identity, "osl"
        ):
            return False
        return all(
            _optional_identity_field_matches(
                reference_identity, measurement_identity, field_name
            )
            for field_name in _OPTIONAL_EMBEDDING_IDENTITY_FIELDS
        )

    return all(
        _optional_identity_field_matches(
            reference_identity, measurement_identity, field_name
        )
        for field_name in (
            "isl",
            "osl",
            "max_concurrency",
            "num_eval_runs",
            "num_inference_steps",
        )
    )


@dataclass
class BenchmarkTaskParams:
    isl: int = None
    osl: int = None
    max_concurrency: int = None
    num_prompts: int = None
    num_eval_runs: int = None
    image_height: int = None
    image_width: int = None
    images_per_prompt: int = 0
    task_type: str = "text"
    theoretical_ttft_ms: float = None
    theoretical_tput_user: float = None
    targets: Dict[str, PerfTarget] = field(default_factory=dict)
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.10,
            "customer_complete": 0.50,
            "customer_sellable": 0.80,
        }
    )
    num_inference_steps: int = None

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        for target_name, peak_perf in self.target_peak_perf.items():
            if target_name not in self.targets and (
                self.theoretical_ttft_ms or self.theoretical_tput_user
            ):
                self.targets[target_name] = PerfTarget(
                    ttft_ms=self.theoretical_ttft_ms / peak_perf
                    if self.theoretical_ttft_ms is not None
                    else None,
                    tput_user=self.theoretical_tput_user * peak_perf
                    if self.theoretical_tput_user is not None
                    else None,
                    target_name=target_name,
                    is_derived=True,
                )


@dataclass
class BenchmarkTaskParamsCNN(BenchmarkTaskParams):
    num_eval_runs: int = 15
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.30,
            "customer_complete": 0.70,
            "customer_sellable": 0.80,
        }
    )

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        super()._infer_data()


def get_perf_targets_path() -> Path:
    return (
        get_repo_root_path()
        / "benchmarking"
        / "benchmark_targets"
        / "model_performance_reference.json"
    )


def get_release_perf_targets_path() -> Path:
    return (
        get_repo_root_path()
        / "benchmarking"
        / "benchmark_targets"
        / "release_performance.json"
    )


def load_perf_targets_json(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    filepath = Path(
        os.getenv("OVERRIDE_BENCHMARK_TARGETS", path or get_perf_targets_path())
    )
    assert filepath.exists(), f"Override benchmark file not found: {filepath}"
    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_release_perf_targets_json(path: Optional[Path] = None) -> Dict[str, Any]:
    filepath = Path(path or get_release_perf_targets_path())
    if not filepath.exists():
        return {"schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION, "models": {}}
    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("models"), dict):
        return {
            "schema_version": data.get(
                "schema_version", RELEASE_PERFORMANCE_SCHEMA_VERSION
            ),
            "models": {},
        }
    return data


model_performance_reference = load_perf_targets_json()
release_performance_reference = load_release_perf_targets_json()


@dataclass(frozen=True)
class PerfTarget:
    isl: int = None
    osl: int = None
    max_concurrency: int = None
    num_prompts: int = None
    task_type: str = "text"
    image_height: int = None
    image_width: int = None
    images_per_prompt: int = 0
    ttft_ms: float = None
    ttft_streaming_ms: float = None
    tput_user: float = None
    tput_prefill: float = None
    e2el_ms: float = None
    tput: float = None
    rtr: float = None
    tolerance: float = 0.05
    num_eval_runs: int = None
    num_inference_steps: int = None
    target_name: Optional[str] = None
    is_derived: bool = False
    is_summary: bool = False

    @classmethod
    def from_dict(cls, row: Dict[str, Any], is_summary: bool = False) -> "PerfTarget":
        theoretical = row.get("targets", {}).get("theoretical", {})
        return cls(
            isl=row.get("isl"),
            osl=row.get("osl"),
            max_concurrency=row.get("max_concurrency"),
            num_prompts=row.get("num_prompts"),
            task_type=row.get("task_type", "text"),
            image_height=row.get("image_height"),
            image_width=row.get("image_width"),
            images_per_prompt=row.get("images_per_prompt", 0),
            ttft_ms=theoretical.get("ttft_ms"),
            ttft_streaming_ms=theoretical.get("ttft_streaming_ms"),
            tput_user=theoretical.get("tput_user"),
            tput_prefill=theoretical.get("tput_prefill"),
            e2el_ms=theoretical.get(
                "e2el_ms", theoretical.get("end_to_end_latency_ms")
            ),
            tput=theoretical.get("tput", theoretical.get("inference_steps_per_second")),
            rtr=theoretical.get("rtr"),
            tolerance=theoretical.get("tolerance", 0.05),
            num_eval_runs=row.get("num_eval_runs"),
            num_inference_steps=row.get("num_inference_steps"),
            target_name=None,
            is_derived=False,
            is_summary=is_summary,
        )

    def build_threshold_targets(
        self, perf_targets_map: Dict[str, float], tolerance: Optional[float] = None
    ) -> Dict[str, "PerfTarget"]:
        targets: Dict[str, PerfTarget] = {}
        latency_metrics = ("ttft_ms", "ttft_streaming_ms", "e2el_ms")
        throughput_metrics = ("tput_user", "tput_prefill", "tput", "rtr")

        for target_name, percentage in perf_targets_map.items():
            target_values = {
                "isl": self.isl,
                "osl": self.osl,
                "max_concurrency": self.max_concurrency,
                "num_prompts": self.num_prompts,
                "task_type": self.task_type,
                "image_height": self.image_height,
                "image_width": self.image_width,
                "images_per_prompt": self.images_per_prompt,
                "num_eval_runs": self.num_eval_runs,
                "num_inference_steps": self.num_inference_steps,
                "tolerance": self.tolerance if tolerance is None else tolerance,
                "target_name": target_name,
                "is_derived": True,
                "is_summary": False,
            }
            for metric_name in latency_metrics:
                metric_value = getattr(self, metric_name)
                target_values[metric_name] = (
                    metric_value / percentage if metric_value is not None else None
                )
            for metric_name in throughput_metrics:
                metric_value = getattr(self, metric_name)
                target_values[metric_name] = (
                    metric_value * percentage if metric_value is not None else None
                )
            targets[target_name] = PerfTarget(**target_values)
        return targets

    def to_benchmark_task_params(
        self,
        perf_targets_map: Dict[str, float],
        tolerance: Optional[float] = None,
    ) -> BenchmarkTaskParams:
        return BenchmarkTaskParams(
            isl=self.isl,
            osl=self.osl,
            max_concurrency=self.max_concurrency,
            num_prompts=self.num_prompts,
            num_eval_runs=self.num_eval_runs,
            task_type=self.task_type,
            image_height=self.image_height,
            image_width=self.image_width,
            images_per_prompt=self.images_per_prompt,
            targets=self.build_threshold_targets(perf_targets_map, tolerance=tolerance),
            num_inference_steps=self.num_inference_steps,
        )

    def scale_llm_for_data_parallel(self, data_parallel: int) -> "PerfTarget":
        max_concurrency = self.max_concurrency
        if max_concurrency and max_concurrency != 1:
            max_concurrency *= data_parallel
        return replace(
            self,
            max_concurrency=max_concurrency,
            tput=self.tput * data_parallel if self.tput else None,
        )

    def benchmark_identity(self) -> Dict[str, Any]:
        identity = {
            "task_type": self.task_type,
            "isl": self.isl,
            "osl": self.osl,
            "max_concurrency": self.max_concurrency,
        }
        if self.num_inference_steps is not None:
            identity["num_inference_steps"] = self.num_inference_steps
        if self.num_eval_runs is not None:
            identity["num_eval_runs"] = self.num_eval_runs
        if self.task_type == "vlm":
            identity.update(
                {
                    "image_height": self.image_height,
                    "image_width": self.image_width,
                    "images_per_prompt": self.images_per_prompt,
                }
            )
        return identity

    def summary_targets(self) -> Dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "ttft_streaming_ms": self.ttft_streaming_ms,
            "tput_user": self.tput_user,
            "tput_prefill": self.tput_prefill,
            "e2el_ms": self.e2el_ms,
            "tput": self.tput,
            "rtr": self.rtr,
            "tolerance": self.tolerance,
        }

    def matches_measurement(self, row: Dict[str, Any]) -> bool:
        return benchmark_identities_match(self.benchmark_identity(), row)


@dataclass(frozen=True)
class PerfTargetSet:
    model_name: str
    device: DeviceTypes
    perf_targets: List[PerfTarget]

    def __post_init__(self):
        if not self.perf_targets:
            return
        summary_targets = [
            perf_target for perf_target in self.perf_targets if perf_target.is_summary
        ]
        if len(summary_targets) != 1:
            raise ValueError(
                "PerfTargetSet must contain exactly one summary PerfTarget"
            )

    @property
    def summary_perf_target(self) -> Optional[PerfTarget]:
        if not self.perf_targets:
            return None
        return next(
            (
                perf_target
                for perf_target in self.perf_targets
                if perf_target.is_summary
            ),
            None,
        )

    @property
    def summary_data_point(self) -> Optional[PerfTarget]:
        return self.summary_perf_target

    def to_benchmark_task_params(
        self,
        perf_targets_map: Optional[Dict[str, float]] = None,
        regression_perf_target_set: Optional["PerfTargetSet"] = None,
        acceptance_criteria: Optional["AcceptanceCriteria"] = None,
    ) -> List[BenchmarkTaskParams]:
        return [
            self._build_benchmark_task_params(
                perf_target=perf_target,
                perf_targets_map=perf_targets_map,
                regression_perf_target_set=regression_perf_target_set,
                acceptance_criteria=acceptance_criteria,
            )
            for perf_target in self.perf_targets
        ]

    def _build_benchmark_task_params(
        self,
        perf_target: PerfTarget,
        perf_targets_map: Optional[Dict[str, float]],
        regression_perf_target_set: Optional["PerfTargetSet"],
        acceptance_criteria: Optional["AcceptanceCriteria"],
    ) -> BenchmarkTaskParams:
        task_params = perf_target.to_benchmark_task_params(
            _resolve_perf_targets_map(perf_targets_map, acceptance_criteria)
        )
        regression_target = None
        if regression_perf_target_set is not None:
            regression_target = regression_perf_target_set.find_matching_perf_target(
                perf_target
            )
        if regression_target is None:
            return task_params

        task_params.targets[REGRESSION_TARGET_NAME] = replace(
            regression_target,
            target_name=REGRESSION_TARGET_NAME,
            is_derived=True,
            tolerance=(
                _resolve_regression_tolerance(acceptance_criteria)
                if acceptance_criteria is not None
                else regression_target.tolerance
                if regression_target.tolerance is not None
                else DEFAULT_REGRESSION_TOLERANCE
            ),
        )
        return task_params

    def scaled_for_data_parallel(self, data_parallel: int) -> "PerfTargetSet":
        return replace(
            self,
            perf_targets=[
                perf_target.scale_llm_for_data_parallel(data_parallel)
                for perf_target in self.perf_targets
            ],
        )

    def find_matching_row(
        self,
        rows: Iterable[Dict[str, Any]],
        perf_target: Optional[PerfTarget] = None,
    ) -> Optional[Dict[str, Any]]:
        target_perf_target = perf_target or self.summary_perf_target
        if target_perf_target is None:
            return None
        return find_matching_benchmark_row(rows, target_perf_target)

    def find_matching_perf_target(
        self, perf_target: PerfTarget
    ) -> Optional[PerfTarget]:
        return next(
            (
                candidate
                for candidate in self.perf_targets
                if perf_target.matches_measurement(candidate.benchmark_identity())
            ),
            None,
        )


def perf_target_from_benchmark_task(benchmark_task: BenchmarkTaskParams) -> PerfTarget:
    return PerfTarget(
        isl=benchmark_task.isl,
        osl=benchmark_task.osl,
        max_concurrency=benchmark_task.max_concurrency,
        task_type=benchmark_task.task_type,
        image_height=benchmark_task.image_height,
        image_width=benchmark_task.image_width,
        images_per_prompt=benchmark_task.images_per_prompt,
        num_eval_runs=getattr(benchmark_task, "num_eval_runs", None),
        num_inference_steps=benchmark_task.num_inference_steps,
    )


def find_matching_benchmark_row(
    rows: Iterable[Dict[str, Any]], perf_target: PerfTarget
) -> Optional[Dict[str, Any]]:
    for row in rows:
        if isinstance(row, dict) and perf_target.matches_measurement(row):
            return row
    return None


def _normalize_device(device: Union[str, DeviceTypes]) -> str:
    if isinstance(device, DeviceTypes):
        return device.name.lower()
    return str(device).lower()


def _build_perf_targets(rows: List[Dict[str, Any]]) -> List[PerfTarget]:
    return [
        PerfTarget.from_dict(row, is_summary=index == 0)
        for index, row in enumerate(rows)
    ]


def _iter_release_device_entries(
    model_name: str, device_key: str
) -> Iterable[Dict[str, Any]]:
    model_entries = release_performance_reference.get("models", {}).get(model_name, {})
    device_entries = model_entries.get(device_key, {})
    if not isinstance(device_entries, dict):
        return []

    entries = []
    for impl_id in sorted(device_entries):
        inference_engines = device_entries.get(impl_id, {})
        if not isinstance(inference_engines, dict):
            continue
        for inference_engine in sorted(inference_engines):
            entry = inference_engines.get(inference_engine)
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _perf_target_from_release_result(result: Dict[str, Any]) -> Optional[PerfTarget]:
    if not isinstance(result, dict):
        return None

    config = result.get("config", {})
    benchmark_summary = result.get("benchmark_summary", {})
    measured_metrics = result.get("measured_metrics", {})
    reference_targets = result.get("targets", {})
    if not isinstance(config, dict) or not isinstance(measured_metrics, dict):
        return None

    merged_identity = (
        dict(benchmark_summary) if isinstance(benchmark_summary, dict) else {}
    )
    merged_identity.update(config)
    tolerance = reference_targets.get("tolerance")
    tolerance = (
        float(tolerance) if tolerance is not None else DEFAULT_REGRESSION_TOLERANCE
    )

    perf_target = PerfTarget(
        isl=merged_identity.get("isl", merged_identity.get("input_sequence_length")),
        osl=merged_identity.get("osl", merged_identity.get("output_sequence_length")),
        max_concurrency=merged_identity.get(
            "max_concurrency", merged_identity.get("max_con")
        ),
        task_type=merged_identity.get("task_type", "text"),
        image_height=merged_identity.get("image_height"),
        image_width=merged_identity.get("image_width"),
        images_per_prompt=merged_identity.get("images_per_prompt", 0),
        ttft_ms=_coerce_optional_float(
            measured_metrics.get("ttft", benchmark_summary.get("ttft"))
        ),
        ttft_streaming_ms=_coerce_optional_float(
            measured_metrics.get(
                "ttft_streaming_ms", benchmark_summary.get("ttft_streaming_ms")
            )
        ),
        tput_user=_coerce_optional_float(
            measured_metrics.get("tput_user", benchmark_summary.get("tput_user"))
        ),
        tput_prefill=_coerce_optional_float(
            measured_metrics.get(
                "tput_prefill",
                benchmark_summary.get(
                    "tput_prefill", benchmark_summary.get("tps_prefill_throughput")
                ),
            )
        ),
        e2el_ms=_coerce_optional_float(
            measured_metrics.get("e2el_ms", benchmark_summary.get("e2el_ms"))
        ),
        tput=_coerce_optional_float(
            measured_metrics.get("tput", benchmark_summary.get("tput"))
        ),
        rtr=_coerce_optional_float(
            measured_metrics.get("rtr", benchmark_summary.get("rtr"))
        ),
        tolerance=tolerance,
        num_eval_runs=merged_identity.get("num_eval_runs"),
        num_inference_steps=merged_identity.get("num_inference_steps"),
        is_summary=bool(result.get("is_summary_data_point", False)),
    )
    if all(
        getattr(perf_target, metric_name) is None
        for metric_name in (
            "ttft_ms",
            "ttft_streaming_ms",
            "tput_user",
            "tput_prefill",
            "e2el_ms",
            "tput",
            "rtr",
        )
    ):
        return None
    return perf_target


def _build_release_perf_target_set(
    model_name: str, device_key: str
) -> Optional[PerfTargetSet]:
    for entry in _iter_release_device_entries(model_name, device_key):
        perf_target_results = entry.get("perf_target_results", [])
        if not isinstance(perf_target_results, list):
            continue

        perf_targets = [
            perf_target
            for perf_target in (
                _perf_target_from_release_result(result)
                for result in perf_target_results
            )
            if perf_target is not None
        ]
        if not perf_targets:
            continue

        return PerfTargetSet(
            model_name=model_name,
            device=DeviceTypes.from_string(device_key),
            perf_targets=perf_targets,
        )

    return None


def get_perf_target_map(model_name: str) -> Dict[DeviceTypes, PerfTargetSet]:
    model_data = model_performance_reference.get(model_name, {})
    perf_target_map: Dict[DeviceTypes, PerfTargetSet] = {}
    for device_str, rows in model_data.items():
        perf_target_map[DeviceTypes.from_string(device_str)] = PerfTargetSet(
            model_name=model_name,
            device=DeviceTypes.from_string(device_str),
            perf_targets=_build_perf_targets(rows),
        )
    return perf_target_map


def get_perf_target(
    model_name: str, device: Union[str, DeviceTypes]
) -> Optional[PerfTargetSet]:
    model_data = model_performance_reference.get(model_name, {})
    device_key = _normalize_device(device)
    rows = model_data.get(device_key, [])
    if not rows:
        return None
    return PerfTargetSet(
        model_name=model_name,
        device=DeviceTypes.from_string(device_key),
        perf_targets=_build_perf_targets(rows),
    )


def get_regression_perf_target(
    model_name: str, device: Union[str, DeviceTypes]
) -> Optional[PerfTargetSet]:
    device_key = _normalize_device(device)
    return _build_release_perf_target_set(model_name, device_key)


def get_perf_target_rows(
    model_name: str, device: Union[str, DeviceTypes]
) -> List[Dict[str, Any]]:
    model_data = model_performance_reference.get(model_name, {})
    return list(model_data.get(_normalize_device(device), []))


def get_performance_targets(
    model_name: str, device_str: str, model_type: str = None
) -> PerfTarget:
    del model_type  # retained for compatibility with older callers
    perf_target_set = get_perf_target(model_name, device_str)
    if perf_target_set and perf_target_set.summary_perf_target:
        logger.info(
            f"Found performance targets for model '{model_name}' on device '{device_str}'"
        )
        return perf_target_set.summary_perf_target

    logger.warning(
        f"No performance targets found for model '{model_name}' on device '{device_str}'"
    )
    return PerfTarget()


def get_perf_reference_map(
    model_name: str,
    perf_targets_map: Optional[Dict[str, float]] = None,
    acceptance_criteria: Optional["AcceptanceCriteria"] = None,
) -> Dict[DeviceTypes, List[BenchmarkTaskParams]]:
    return {
        device: perf_target_set.to_benchmark_task_params(
            perf_targets_map=_resolve_perf_targets_map(
                perf_targets_map, acceptance_criteria
            ),
            regression_perf_target_set=get_regression_perf_target(model_name, device),
            acceptance_criteria=acceptance_criteria,
        )
        for device, perf_target_set in get_perf_target_map(model_name).items()
    }


def get_named_perf_reference(
    model_name: str,
    device: Union[str, DeviceTypes],
    perf_targets_map: Optional[Dict[str, float]] = None,
    acceptance_criteria: Optional["AcceptanceCriteria"] = None,
) -> List[BenchmarkTaskParams]:
    perf_target_set = get_perf_target(model_name, device)
    if perf_target_set is None:
        return []

    return perf_target_set.to_benchmark_task_params(
        perf_targets_map=_resolve_perf_targets_map(
            perf_targets_map, acceptance_criteria
        ),
        regression_perf_target_set=get_regression_perf_target(model_name, device),
        acceptance_criteria=acceptance_criteria,
    )


def scale_llm_perf_targets(
    task: BenchmarkTaskParams, data_parallel: int
) -> BenchmarkTaskParams:
    scaled_targets = {
        target_name: replace(
            target,
            tput=target.tput * data_parallel if target.tput is not None else None,
            tput_prefill=target.tput_prefill * data_parallel
            if target.tput_prefill is not None
            else None,
        )
        for target_name, target in task.targets.items()
    }
    return BenchmarkTaskParams(
        isl=task.isl,
        osl=task.osl,
        max_concurrency=task.max_concurrency
        if task.max_concurrency == 1
        else task.max_concurrency * data_parallel,
        num_prompts=task.num_prompts,
        num_eval_runs=task.num_eval_runs,
        image_height=task.image_height,
        image_width=task.image_width,
        images_per_prompt=task.images_per_prompt,
        task_type=task.task_type,
        theoretical_ttft_ms=task.theoretical_ttft_ms,
        theoretical_tput_user=task.theoretical_tput_user,
        targets=scaled_targets,
        target_peak_perf=task.target_peak_perf,
        num_inference_steps=task.num_inference_steps,
    )


def get_perf_reference_for_device(
    device: DeviceTypes,
    override_tt_config: Optional[Dict[str, Any]],
    perf_reference_map: Dict[DeviceTypes, List[BenchmarkTaskParams]],
) -> List[BenchmarkTaskParams]:
    override_tt_config = override_tt_config or {}
    data_parallel = override_tt_config.get("data_parallel")

    if data_parallel:
        dp_device = device.get_data_parallel_subdevice(data_parallel)
        perf_reference = perf_reference_map.get(dp_device, [])
        if perf_reference:
            return [
                scale_llm_perf_targets(task, data_parallel) for task in perf_reference
            ]

    return perf_reference_map.get(device, [])

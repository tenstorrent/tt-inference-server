#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Helpers for release performance baselines and markdown rendering."""

import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from release_diff import ReleaseDiffRecord
except ImportError:
    from scripts.release.release_diff import ReleaseDiffRecord

from workflows.perf_targets import get_perf_target


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RELEASE_PERFORMANCE_PATH = (
    PROJECT_ROOT / "benchmarking" / "benchmark_targets" / "release_performance.json"
)
RELEASE_PERFORMANCE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "benchmarking"
    / "benchmark_targets"
    / "release_performance_schema.json"
)
RELEASE_PERFORMANCE_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class ReleasePerformanceRecordArtifacts:
    """Rich and stripped release-performance entries for one merged record."""

    model_spec: Any
    rich_entry: Dict[str, Any]
    baseline_entry: Dict[str, Any]


@dataclass(frozen=True)
class ReleasePerformanceArtifacts:
    """Release-performance payloads shared across write modes."""

    rich_data: Dict[str, Any]
    baseline_data: Dict[str, Any]
    records_with_entries: Tuple[ReleasePerformanceRecordArtifacts, ...]


@dataclass(frozen=True)
class ReleasePerformanceUpdateResult:
    """Shared updater result for Step-5 and backfill callers."""

    artifacts: ReleasePerformanceArtifacts
    final_baseline_data: Dict[str, Any]
    updated_count: int


class ReleasePerformanceWriteMode(str, Enum):
    """Persistence strategy for the checked-in release baseline."""

    REPLACE = "replace"
    MERGE_NEWER_ONLY = "merge_newer_only"


def _empty_release_performance_data() -> Dict[str, Any]:
    return {
        "schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION,
        "models": {},
    }


def get_release_performance_path() -> Path:
    """Return the checked-in release performance baseline path."""
    return RELEASE_PERFORMANCE_PATH


def _import_jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError(
            "Release performance schema validation requires the 'jsonschema' "
            "package. Install the workflow dependencies that include it before "
            "writing benchmarking/benchmark_targets/release_performance.json."
        ) from exc
    return jsonschema


@lru_cache(maxsize=1)
def load_release_performance_schema() -> Dict[str, Any]:
    with RELEASE_PERFORMANCE_SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _format_release_performance_validation_error(error: Exception) -> str:
    absolute_path = getattr(error, "absolute_path", [])
    location = " -> ".join(str(part) for part in absolute_path) or "<root>"
    message = getattr(error, "message", str(error))
    return f"release_performance schema validation failed at {location}: {message}"


def validate_release_performance_data(
    release_performance_data: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> None:
    jsonschema = _import_jsonschema()
    active_schema = load_release_performance_schema() if schema is None else schema

    try:
        jsonschema.validate(instance=release_performance_data, schema=active_schema)
    except jsonschema.ValidationError as exc:
        raise RuntimeError(_format_release_performance_validation_error(exc)) from exc


def normalize_device_name(device: Any) -> str:
    """Convert a DeviceTypes enum or string into the release JSON key form."""
    if hasattr(device, "name"):
        return str(device.name).lower()
    return str(device).lower()


def normalize_inference_engine_name(inference_engine: Any) -> str:
    """Convert an inference engine enum or string into a stable string."""
    if hasattr(inference_engine, "value"):
        return str(inference_engine.value)
    return str(inference_engine)


def load_release_performance_data(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the checked-in release performance baseline if it exists."""
    data_path = path or get_release_performance_path()
    if not data_path.exists():
        return _empty_release_performance_data()

    with data_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "models" not in data or not isinstance(data["models"], dict):
        return _empty_release_performance_data()
    return data


def write_release_performance_data(
    release_performance_data: Dict[str, Any], path: Optional[Path] = None
) -> Path:
    """Write the release performance baseline with deterministic formatting."""
    validate_release_performance_data(release_performance_data)
    data_path = path or get_release_performance_path()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(release_performance_data, indent=4) + "\n",
        encoding="utf-8",
    )
    return data_path


def _benchmark_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        row.get("task_type", "text"),
        row.get("isl", row.get("input_sequence_length", 0)),
        row.get("osl", row.get("output_sequence_length", 0)),
        row.get("image_height", 0),
        row.get("image_width", 0),
        row.get("images_per_prompt", 0),
        row.get("max_concurrency", row.get("max_con", 0)),
    )


def _normalize_benchmark_rows(rows: Any) -> List[Dict[str, Any]]:
    normalized_rows = []
    if not isinstance(rows, list):
        return normalized_rows

    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_row = deepcopy(row)
        if "task_type" not in normalized_row:
            is_vlm = any(
                key in normalized_row
                for key in ("image_height", "image_width", "images_per_prompt")
            )
            normalized_row["task_type"] = "vlm" if is_vlm else "text"
        normalized_rows.append(normalized_row)

    return sorted(normalized_rows, key=_benchmark_sort_key)


def _normalize_release_report_json(release_report_json: Any) -> Dict[str, Any]:
    if not isinstance(release_report_json, dict):
        return {}

    normalized_report = deepcopy(release_report_json)
    benchmarks_summary = normalized_report.get("benchmarks_summary")
    if isinstance(benchmarks_summary, list):
        normalized_report["benchmarks_summary"] = _normalize_benchmark_rows(
            benchmarks_summary
        )
    return normalized_report


def _extract_measured_metrics(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "ttft": None,
            "tput_user": None,
            "tput": None,
            "ttft_streaming_ms": None,
            "tput_prefill": None,
            "e2el_ms": None,
            "rtr": None,
        }

    return {
        "ttft": row.get("ttft", row.get("mean_ttft_ms")),
        "tput_user": row.get("tput_user", row.get("mean_tps")),
        "tput": row.get("tput", row.get("tps_decode_throughput")),
        "ttft_streaming_ms": row.get("ttft_streaming_ms"),
        "tput_prefill": row.get("tput_prefill", row.get("tps_prefill_throughput")),
        "e2el_ms": row.get("e2el_ms", row.get("mean_e2el_ms")),
        "rtr": row.get("rtr"),
    }


def _build_perf_target_results(
    model_name: str,
    device: str,
    benchmarks_summary: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    perf_target_set = get_perf_target(model_name, device)
    if not perf_target_set:
        return []

    results: List[Dict[str, Any]] = []
    for perf_target in perf_target_set.perf_targets:
        if not perf_target.is_summary:
            continue
        matched_row = perf_target_set.find_matching_row(benchmarks_summary, perf_target)
        results.append(
            {
                "is_summary_data_point": perf_target.is_summary,
                "config": perf_target.benchmark_identity(),
                "measured_metrics": _extract_measured_metrics(matched_row),
            }
        )
    return results


def _has_non_null_measured_metrics(perf_target_results: List[Dict[str, Any]]) -> bool:
    for result in perf_target_results:
        measured_metrics = result.get("measured_metrics")
        if not isinstance(measured_metrics, dict):
            continue
        if any(value is not None for value in measured_metrics.values()):
            return True
    return False


def _benchmark_identity_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Build a stable identity for one benchmark datapoint row."""

    def _normalize_identity_value(value: Any) -> Any:
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value.isdigit():
                return int(stripped_value)
            return stripped_value
        return value

    return (
        _normalize_identity_value(row.get("task_type", "text")),
        _normalize_identity_value(row.get("isl", row.get("input_sequence_length"))),
        _normalize_identity_value(row.get("osl", row.get("output_sequence_length"))),
        _normalize_identity_value(row.get("max_concurrency", row.get("max_con"))),
        _normalize_identity_value(row.get("image_height")),
        _normalize_identity_value(row.get("image_width")),
        _normalize_identity_value(row.get("images_per_prompt")),
        _normalize_identity_value(row.get("num_requests")),
    )


def _filter_benchmark_rows_for_perf_targets(
    perf_target_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only benchmark rows that correspond to evaluated perf targets."""
    filtered_rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for result in perf_target_results:
        config = result.get("config")
        measured_metrics = result.get("measured_metrics")
        if not isinstance(config, dict) or not isinstance(measured_metrics, dict):
            continue
        benchmark_row = deepcopy(config)
        benchmark_row.update(
            {
                "ttft": measured_metrics.get("ttft"),
                "tput_user": measured_metrics.get("tput_user"),
                "tput": measured_metrics.get("tput"),
                "ttft_streaming_ms": measured_metrics.get("ttft_streaming_ms"),
                "tput_prefill": measured_metrics.get("tput_prefill"),
                "e2el_ms": measured_metrics.get("e2el_ms"),
                "rtr": measured_metrics.get("rtr"),
            }
        )
        filtered_rows[_benchmark_identity_key(benchmark_row)] = benchmark_row
    return sorted(filtered_rows.values(), key=_benchmark_sort_key)


def _filter_report_benchmark_rows(
    rows: Any, allowed_keys: Sequence[Tuple[Any, ...]]
) -> List[Dict[str, Any]]:
    """Filter one report benchmark row list down to the allowed benchmark identities."""
    if not isinstance(rows, list):
        return []
    allowed_key_set = set(allowed_keys)
    filtered_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _benchmark_identity_key(row) not in allowed_key_set:
            continue
        filtered_rows.append(deepcopy(row))
    return filtered_rows


def build_release_performance_entry(
    model_spec: Any,
    ci_data: Dict[str, Any],
    *,
    include_report_data: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build one release-performance entry from merged CI/model-spec data."""
    report_data = _normalize_release_report_json(ci_data.get("release_report") or {})
    if not report_data:
        return None
    metadata = report_data.get("metadata") if isinstance(report_data, dict) else {}
    release_version = (
        metadata.get("release_version") if isinstance(metadata, dict) else None
    )
    report_data_json_path = ci_data.get("release_report_json_path")
    normalized_release_version = normalize_release_version(release_version)
    if normalized_release_version is not None:
        release_version = normalized_release_version
    benchmarks_summary = report_data.get("benchmarks_summary", [])
    perf_target_results = _build_perf_target_results(
        model_name=model_spec.model_name,
        device=normalize_device_name(model_spec.device_type),
        benchmarks_summary=benchmarks_summary,
    )
    if perf_target_results and not _has_non_null_measured_metrics(perf_target_results):
        return None
    entry = {
        "perf_status": ci_data.get("perf_status"),
        "release_version": release_version,
        "ci_run_number": ci_data.get("ci_run_number"),
        "ci_run_url": ci_data.get("ci_run_url"),
        "ci_job_url": ci_data.get("ci_job_url"),
        "perf_target_results": perf_target_results,
    }
    if include_report_data:
        entry["report_data"] = report_data
        if report_data_json_path:
            entry["report_data_json_path"] = report_data_json_path
    return entry


def _set_release_performance_entry(
    models: Dict[str, Any],
    model_name: str,
    device: str,
    impl_id: str,
    inference_engine: str,
    entry: Dict[str, Any],
) -> None:
    models.setdefault(model_name, {}).setdefault(device, {}).setdefault(impl_id, {})[
        inference_engine
    ] = entry


def _sort_release_performance_records(records: Iterable[Any]) -> List[Any]:
    return sorted(
        records,
        key=lambda record: (
            getattr(record.model_spec, "model_name", ""),
            normalize_device_name(getattr(record.model_spec, "device_type", "")),
            getattr(getattr(record.model_spec, "impl", None), "impl_id", ""),
            normalize_inference_engine_name(
                getattr(record.model_spec, "inference_engine", "")
            ),
            getattr(record, "model_id", ""),
        ),
    )


def _build_release_performance_record_artifacts(
    record: Any,
) -> Optional[ReleasePerformanceRecordArtifacts]:
    rich_entry = build_release_performance_entry(
        record.model_spec,
        record.ci_data,
        include_report_data=True,
    )
    if not rich_entry:
        return None
    baseline_entry = deepcopy(rich_entry)
    baseline_entry.pop("report_data", None)
    baseline_entry.pop("report_data_json_path", None)
    baseline_entry.pop("report_markdown", None)
    return ReleasePerformanceRecordArtifacts(
        model_spec=record.model_spec,
        rich_entry=rich_entry,
        baseline_entry=baseline_entry,
    )


def build_release_performance_artifacts(
    records: Iterable[Any],
) -> ReleasePerformanceArtifacts:
    """Build shared rich/baseline release-performance payloads."""
    rich_models: Dict[str, Any] = {}
    baseline_models: Dict[str, Any] = {}
    record_artifacts: List[ReleasePerformanceRecordArtifacts] = []

    for record in _sort_release_performance_records(records):
        record_artifacts_entry = _build_release_performance_record_artifacts(record)
        if record_artifacts_entry is None:
            continue
        record_artifacts.append(record_artifacts_entry)
        _set_release_performance_entry(
            rich_models,
            record.model_spec.model_name,
            normalize_device_name(record.model_spec.device_type),
            record.model_spec.impl.impl_id,
            normalize_inference_engine_name(record.model_spec.inference_engine),
            deepcopy(record_artifacts_entry.rich_entry),
        )
        _set_release_performance_entry(
            baseline_models,
            record.model_spec.model_name,
            normalize_device_name(record.model_spec.device_type),
            record.model_spec.impl.impl_id,
            normalize_inference_engine_name(record.model_spec.inference_engine),
            deepcopy(record_artifacts_entry.baseline_entry),
        )

    return ReleasePerformanceArtifacts(
        rich_data={
            "schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION,
            "models": rich_models,
        },
        baseline_data={
            "schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION,
            "models": baseline_models,
        },
        records_with_entries=tuple(record_artifacts),
    )


def apply_release_performance_updates(
    existing_baseline_data: Optional[Dict[str, Any]],
    artifacts: ReleasePerformanceArtifacts,
    mode: ReleasePerformanceWriteMode,
) -> ReleasePerformanceUpdateResult:
    """Apply the requested baseline update strategy without writing to disk."""
    if mode == ReleasePerformanceWriteMode.REPLACE:
        return ReleasePerformanceUpdateResult(
            artifacts=artifacts,
            final_baseline_data=deepcopy(artifacts.baseline_data),
            updated_count=len(artifacts.records_with_entries),
        )

    if mode != ReleasePerformanceWriteMode.MERGE_NEWER_ONLY:
        raise ValueError(f"Unsupported release performance write mode: {mode}")

    final_baseline_data = deepcopy(
        existing_baseline_data
        if existing_baseline_data is not None
        else _empty_release_performance_data()
    )
    final_baseline_data.setdefault("schema_version", RELEASE_PERFORMANCE_SCHEMA_VERSION)
    final_baseline_data.setdefault("models", {})

    updated_count = 0
    for record_artifacts in artifacts.records_with_entries:
        if merge_release_performance_entry(
            final_baseline_data,
            record_artifacts.model_spec,
            deepcopy(record_artifacts.baseline_entry),
        ):
            updated_count += 1

    return ReleasePerformanceUpdateResult(
        artifacts=artifacts,
        final_baseline_data=final_baseline_data,
        updated_count=updated_count,
    )


def update_release_performance_outputs(
    records: Iterable[Any],
    *,
    mode: ReleasePerformanceWriteMode,
    existing_baseline_data: Optional[Dict[str, Any]] = None,
) -> ReleasePerformanceUpdateResult:
    """Build shared release-performance artifacts and apply the chosen mode."""
    artifacts = build_release_performance_artifacts(records)
    return apply_release_performance_updates(
        existing_baseline_data, artifacts, mode=mode
    )


def build_release_performance_data(
    records: Iterable[Any],
    *,
    include_report_data: bool = False,
) -> Dict[str, Any]:
    """Build release performance data from merged model records."""
    artifacts = build_release_performance_artifacts(records)
    if include_report_data:
        return deepcopy(artifacts.rich_data)
    return deepcopy(artifacts.baseline_data)


def strip_release_report_data(
    release_performance_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of release performance data without embedded report payloads."""
    stripped_data = deepcopy(release_performance_data)
    models = stripped_data.get("models")
    if not isinstance(models, dict):
        return stripped_data

    for model_devices in models.values():
        if not isinstance(model_devices, dict):
            continue
        for device_impls in model_devices.values():
            if not isinstance(device_impls, dict):
                continue
            for impl_engines in device_impls.values():
                if not isinstance(impl_engines, dict):
                    continue
                for entry in impl_engines.values():
                    if isinstance(entry, dict):
                        entry.pop("report_data", None)
                        entry.pop("report_data_json_path", None)
                        entry.pop("report_markdown", None)

    return stripped_data


def parse_release_version(value: Any) -> Optional[Tuple[int, int, int]]:
    """Parse a release version string like `v0.11.0` or `0.11.0`."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("v"):
        text = text[1:]

    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def normalize_release_version(value: Any) -> Optional[str]:
    """Return a canonical `major.minor.patch` release version string."""
    parsed_version = parse_release_version(value)
    if parsed_version is None:
        return None
    return ".".join(str(part) for part in parsed_version)


def is_incoming_release_version_newer(
    existing_version: Any, incoming_version: Any
) -> bool:
    """Return whether an incoming release version is newer than an existing one."""
    parsed_incoming_version = parse_release_version(incoming_version)
    if parsed_incoming_version is None:
        return False

    parsed_existing_version = parse_release_version(existing_version)
    if parsed_existing_version is None:
        return True

    return parsed_incoming_version > parsed_existing_version


def get_release_performance_identity(model_spec: Any) -> Tuple[str, str, str, str]:
    """Return the nested release-performance key for one model spec."""
    return (
        model_spec.model_name,
        normalize_device_name(model_spec.device_type),
        model_spec.impl.impl_id,
        normalize_inference_engine_name(model_spec.inference_engine),
    )


def should_replace_release_performance_entry(
    existing_entry: Optional[Dict[str, Any]], incoming_entry: Dict[str, Any]
) -> bool:
    """Allow updates only when the incoming entry is for a newer release."""
    if existing_entry is None:
        return True
    return is_incoming_release_version_newer(
        existing_entry.get("release_version"),
        incoming_entry.get("release_version"),
    )


def merge_release_performance_entry(
    release_performance_data: Dict[str, Any],
    model_spec: Any,
    incoming_entry: Dict[str, Any],
) -> bool:
    """Merge one release-performance entry in place when it is newer."""
    release_performance_data.setdefault(
        "schema_version", RELEASE_PERFORMANCE_SCHEMA_VERSION
    )
    models = release_performance_data.setdefault("models", {})

    model_name, device, impl_id, inference_engine = get_release_performance_identity(
        model_spec
    )
    existing_entry = get_release_performance_entry(
        release_performance_data,
        model_name,
        device,
        impl_id,
        inference_engine,
    )
    if not should_replace_release_performance_entry(existing_entry, incoming_entry):
        return False

    _set_release_performance_entry(
        models,
        model_name,
        device,
        impl_id,
        inference_engine,
        incoming_entry,
    )
    return True


def get_release_performance_entry(
    release_performance_data: Dict[str, Any],
    model_name: str,
    device: Any,
    impl_id: str,
    inference_engine: Any,
) -> Optional[Dict[str, Any]]:
    """Look up a release-performance entry by its full identity."""
    return (
        release_performance_data.get("models", {})
        .get(model_name, {})
        .get(normalize_device_name(device), {})
        .get(impl_id, {})
        .get(normalize_inference_engine_name(inference_engine))
    )


def get_release_performance_summary(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    perf_target_results = entry.get("perf_target_results")
    if not isinstance(perf_target_results, list):
        return None
    for result in perf_target_results:
        if not isinstance(result, dict):
            continue
        if result.get("is_summary_data_point", False):
            return result
    return None


def get_release_performance_summary_metrics(
    entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    summary = get_release_performance_summary(entry)
    if not summary:
        return None
    metrics = summary.get("measured_metrics")
    if isinstance(metrics, dict) and any(
        value is not None for value in metrics.values()
    ):
        return metrics
    return None


def iter_release_performance_entries(
    release_performance_data: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Yield release-performance entries in deterministic order."""
    for _, _, _, _, entry in iter_release_performance_items(release_performance_data):
        yield entry


def iter_release_performance_items(
    release_performance_data: Dict[str, Any],
) -> Iterable[Tuple[str, str, str, str, Dict[str, Any]]]:
    """Yield release-performance entries with their nested map identity."""
    models = release_performance_data.get("models", {})
    for model_name in sorted(models.keys()):
        model_devices = models[model_name]
        for device in sorted(model_devices.keys()):
            device_impls = model_devices[device]
            for impl_id in sorted(device_impls.keys()):
                impl_engines = device_impls[impl_id]
                for inference_engine in sorted(impl_engines.keys()):
                    yield (
                        model_name,
                        device,
                        impl_id,
                        inference_engine,
                        impl_engines[inference_engine],
                    )


PerformanceEntry = Dict[str, Any]
PerformanceFullKey = Tuple[str, str, str, str]
PerformancePartialKey = Tuple[str, str, str]
_AMBIGUOUS_MATCH = object()


def _normalize_model_name(model_name: Any) -> str:
    return str(model_name or "").strip().lower()


def _normalize_engine_key(inference_engine: Any) -> str:
    return normalize_inference_engine_name(inference_engine).strip().lower()


def build_release_performance_entry_indices(
    release_performance_data: Dict[str, Any],
) -> Tuple[
    Dict[PerformanceFullKey, PerformanceEntry],
    Dict[PerformancePartialKey, PerformanceEntry],
]:
    """Build exact and fallback indices for release-performance entry matching."""
    full_index: Dict[PerformanceFullKey, PerformanceEntry] = {}
    partial_candidates: Dict[PerformancePartialKey, Any] = {}

    for (
        model_name,
        device,
        impl_id,
        inference_engine,
        entry,
    ) in iter_release_performance_items(release_performance_data):
        full_key = (
            _normalize_model_name(model_name),
            normalize_device_name(device),
            str(impl_id or "").strip(),
            _normalize_engine_key(inference_engine),
        )
        partial_key = (
            str(impl_id or "").strip(),
            normalize_device_name(device),
            _normalize_engine_key(inference_engine),
        )
        full_index[full_key] = entry

        existing_entry = partial_candidates.get(partial_key)
        if existing_entry is None:
            partial_candidates[partial_key] = entry
        else:
            partial_candidates[partial_key] = _AMBIGUOUS_MATCH

    partial_index = {
        key: value
        for key, value in partial_candidates.items()
        if value is not _AMBIGUOUS_MATCH
    }
    return full_index, partial_index


def lookup_release_performance_entry(
    full_index: Dict[PerformanceFullKey, PerformanceEntry],
    partial_index: Dict[PerformancePartialKey, PerformanceEntry],
    record: ReleaseDiffRecord,
    device: str,
) -> Optional[PerformanceEntry]:
    """Resolve one release diff record/device pair to a release-performance entry."""
    full_key = (
        _normalize_model_name(record.get("model_arch")),
        normalize_device_name(device),
        str(record.get("impl_id") or "").strip(),
        _normalize_engine_key(record.get("inference_engine")),
    )
    entry = full_index.get(full_key)
    if entry is not None:
        return entry

    partial_key = (
        str(record.get("impl_id") or "").strip(),
        normalize_device_name(device),
        _normalize_engine_key(record.get("inference_engine")),
    )
    return partial_index.get(partial_key)


def _index_perf_target_results(entry: Optional[PerformanceEntry]) -> Dict[str, str]:
    indexed_results = {}
    if entry is None:
        return indexed_results

    perf_target_results = entry.get("perf_target_results")
    if not isinstance(perf_target_results, list):
        return indexed_results

    for result in perf_target_results:
        if not isinstance(result, dict):
            continue
        config = result.get("config")
        if not isinstance(config, dict):
            continue
        indexed_results[json.dumps(config, sort_keys=True)] = json.dumps(
            result,
            sort_keys=True,
        )
    return indexed_results


def _format_change_counts(label: str, added: int, removed: int, updated: int) -> str:
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    if updated:
        parts.append(f"~{updated}")
    if not parts:
        return ""
    return f"{label} {' '.join(parts)}"


def _normalize_status_value(status_value: Any) -> Optional[str]:
    if status_value is None:
        return None
    status_text = str(status_value)
    if not status_text:
        return None
    if status_text.startswith("ModelStatusTypes."):
        return status_text.split(".", 1)[1]
    return status_text


def _format_status_change(
    before: Any, after: Any, unchanged_label: str = "No change"
) -> str:
    normalized_before = _normalize_status_value(before)
    normalized_after = _normalize_status_value(after)
    if normalized_before and normalized_after:
        if normalized_before != normalized_after:
            return f"{normalized_before} -> {normalized_after}"
        return f"{normalized_after} (no change)"
    if normalized_after:
        return f"New: {normalized_after}"
    if normalized_before:
        return f"Removed: {normalized_before}"
    return unchanged_label


def _summarize_benchmark_changes(
    before_entry: Optional[PerformanceEntry], after_entry: Optional[PerformanceEntry]
) -> str:
    before_results = _index_perf_target_results(before_entry)
    after_results = _index_perf_target_results(after_entry)
    if not before_results and not after_results:
        return ""
    if not before_results:
        return f"Perf targets added ({len(after_results)} row(s))"
    if not after_results:
        return f"Perf targets removed ({len(before_results)} row(s))"

    before_keys = set(before_results.keys())
    after_keys = set(after_results.keys())
    updated_count = sum(
        1
        for key in before_keys & after_keys
        if before_results[key] != after_results[key]
    )
    return _format_change_counts(
        "Perf targets",
        len(after_keys - before_keys),
        len(before_keys - after_keys),
        updated_count,
    )


def summarize_release_performance_diff(
    before_entry: Optional[PerformanceEntry], after_entry: Optional[PerformanceEntry]
) -> str:
    """Return the human-readable summary used by release notes and JSON output."""
    if before_entry is None and after_entry is None:
        return "No release data"
    if before_entry is None:
        return "New release data"
    if after_entry is None:
        return "Removed release data"

    changes = []
    if before_entry.get("perf_status") != after_entry.get("perf_status"):
        changes.append(
            "Perf status: "
            + _format_status_change(
                before_entry.get("perf_status"),
                after_entry.get("perf_status"),
                unchanged_label="No change",
            )
        )

    benchmark_change = _summarize_benchmark_changes(before_entry, after_entry)
    if benchmark_change:
        changes.append(benchmark_change)

    if not changes:
        return "No performance changes"
    return "; ".join(changes)


def _classify_release_performance_diff(
    before_entry: Optional[PerformanceEntry],
    after_entry: Optional[PerformanceEntry],
    summary: str,
) -> str:
    if before_entry is None and after_entry is None:
        return "no_data"
    if before_entry is None:
        return "new"
    if after_entry is None:
        return "removed"
    if summary == "No performance changes":
        return "unchanged"
    return "changed"


def build_release_performance_diff_records(
    release_diff_records: Sequence[ReleaseDiffRecord],
    release_performance_data: Dict[str, Any],
    base_release_performance_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build release-scoped performance diff records keyed by template/device."""
    current_indices = build_release_performance_entry_indices(release_performance_data)
    base_indices = build_release_performance_entry_indices(
        base_release_performance_data
    )
    diff_records: List[Dict[str, Any]] = []

    for record in release_diff_records:
        for device in record.get("devices", []):
            after_entry = lookup_release_performance_entry(
                current_indices[0], current_indices[1], record, device
            )
            before_entry = lookup_release_performance_entry(
                base_indices[0], base_indices[1], record, device
            )
            summary = summarize_release_performance_diff(before_entry, after_entry)
            diff_records.append(
                {
                    "template_key": record.get("template_key"),
                    "model_arch": record.get("model_arch"),
                    "impl_id": record.get("impl_id"),
                    "inference_engine": record.get("inference_engine"),
                    "weights": deepcopy(record.get("weights", [])),
                    "device": device,
                    "diff_status": _classify_release_performance_diff(
                        before_entry, after_entry, summary
                    ),
                    "summary": summary,
                    "before_entry": deepcopy(before_entry),
                    "after_entry": deepcopy(after_entry),
                }
            )

    return diff_records


def build_release_performance_diff_data(
    release_diff_records: Sequence[ReleaseDiffRecord],
    release_performance_data: Dict[str, Any],
    base_release_performance_data: Dict[str, Any],
    base_ref: str = "HEAD",
    compared_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the structured release-scoped performance diff artifact."""
    compared_path_value = compared_path or get_release_performance_path()
    try:
        compared_path_text = str(compared_path_value.relative_to(PROJECT_ROOT))
    except ValueError:
        compared_path_text = str(compared_path_value)

    diff_records = build_release_performance_diff_records(
        release_diff_records,
        release_performance_data,
        base_release_performance_data,
    )
    return {
        "schema_version": RELEASE_PERFORMANCE_SCHEMA_VERSION,
        "base_ref": base_ref,
        "compared_path": compared_path_text,
        "records": diff_records,
    }


def write_release_performance_diff_data(
    release_performance_diff_data: Dict[str, Any], path: Path
) -> Path:
    """Write the release performance diff JSON with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(release_performance_diff_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _split_benchmark_rows(entry: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rows = _filter_benchmark_rows_for_perf_targets(
        entry.get("perf_target_results")
        if isinstance(entry.get("perf_target_results"), list)
        else []
    )
    return {
        "text": [row for row in rows if row.get("task_type", "text") == "text"],
        "vlm": [row for row in rows if row.get("task_type", "text") == "vlm"],
    }


def _prepare_renderable_benchmark_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from workflows.workflow_types import ReportCheckTypes

    prepared_rows = deepcopy(rows)
    for row in prepared_rows:
        target_checks = row.get("target_checks", {})
        if not isinstance(target_checks, dict):
            continue
        for target_values in target_checks.values():
            if not isinstance(target_values, dict):
                continue
            for key, value in list(target_values.items()):
                if not key.endswith("_check") or not isinstance(value, int):
                    continue
                try:
                    target_values[key] = ReportCheckTypes(value)
                except ValueError:
                    continue
    return prepared_rows


def _flatten_target_checks(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened_rows = []
    for row in rows:
        flattened_row = {
            key: value for key, value in row.items() if key != "target_checks"
        }
        for target_name, checks in row.get("target_checks", {}).items():
            if not isinstance(checks, dict):
                continue
            for metric, value in checks.items():
                flattened_row[f"{target_name}_{metric}"] = value
        flattened_rows.append(flattened_row)
    return flattened_rows


def _sanitize_markdown_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _markdown_cell_width(text: str) -> int:
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in ("F", "W") else 1
    return width


def _pad_right(text: str, width: int) -> str:
    return text + " " * max(width - _markdown_cell_width(text), 0)


def _pad_left(text: str, width: int) -> str:
    return " " * max(width - _markdown_cell_width(text), 0) + text


def _pad_center(text: str, width: int) -> str:
    total_padding = width - _markdown_cell_width(text)
    left_padding = total_padding // 2
    return (
        " " * max(left_padding, 0) + text + " " * max(total_padding - left_padding, 0)
    )


def _format_numeric_table_value(
    value: str,
    header: str,
    max_left: Dict[str, int],
    max_right: Dict[str, int],
) -> str:
    left, _, right = value.partition(".")
    left = left.rjust(max_left[header])
    if max_right[header] > 0:
        right = right.ljust(max_right[header])
        return f"{left}.{right}"
    return left


def _render_markdown_table(display_rows: List[Dict[str, str]]) -> str:
    if not display_rows:
        return ""

    headers = list(display_rows[0].keys())
    numeric_columns = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(row.get(header, "")).strip())
            for row in display_rows
        )
        for header in headers
    }

    max_left: Dict[str, int] = {}
    max_right: Dict[str, int] = {}
    for header in headers:
        max_left[header] = 0
        max_right[header] = 0
        if not numeric_columns[header]:
            continue
        for row in display_rows:
            value = str(row.get(header, "")).strip()
            left, _, right = value.partition(".")
            max_left[header] = max(max_left[header], len(left))
            max_right[header] = max(max_right[header], len(right))

    column_widths = {}
    for header in headers:
        if numeric_columns[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            column_widths[header] = max(_markdown_cell_width(header), numeric_width)
            continue
        max_content_width = max(
            _markdown_cell_width(_sanitize_markdown_cell(row.get(header, "")))
            for row in display_rows
        )
        column_widths[header] = max(_markdown_cell_width(header), max_content_width)

    header_row = (
        "| "
        + " | ".join(
            _pad_center(_sanitize_markdown_cell(header), column_widths[header])
            for header in headers
        )
        + " |"
    )
    separator_row = (
        "|" + "|".join("-" * (column_widths[header] + 2) for header in headers) + "|"
    )

    value_rows = []
    for row in display_rows:
        formatted_cells = []
        for header in headers:
            raw_value = _sanitize_markdown_cell(str(row.get(header, "")).strip())
            if numeric_columns[header]:
                raw_value = _format_numeric_table_value(
                    raw_value, header, max_left, max_right
                )
                formatted_cells.append(_pad_left(raw_value, column_widths[header]))
            else:
                formatted_cells.append(_pad_right(raw_value, column_widths[header]))
        value_rows.append("| " + " | ".join(formatted_cells) + " |")

    def _clean_header(header: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", header).strip()

    explanation_map = {
        "ISL": "Input Sequence Length (tokens)",
        "OSL": "Output Sequence Length (tokens)",
        "Concurrency": "number of concurrent requests (batch size)",
        "Num Requests": "total number of requests (sample size, N)",
        "TTFT": "Time To First Token (ms)",
        "Tput User": "Throughput per user (TPS)",
        "Tput Decode": "Throughput for decode tokens, across all users (TPS)",
        "Max Concurrency": "number of concurrent requests (batch size)",
    }
    explanation_lines = [
        f"> {header}: {explanation_map[header]}"
        for header in (_clean_header(header) for header in headers)
        if header in explanation_map
    ]
    table = "\n".join([header_row, separator_row] + value_rows)
    note = "Note: all metrics are means across benchmark run unless otherwise stated."
    if explanation_lines:
        return f"{table}\n\n{note}\n" + "\n".join(explanation_lines)
    return f"{table}\n\n{note}"


def _build_benchmark_check_columns(target_checks: Any) -> List[tuple]:
    if not isinstance(target_checks, dict):
        return []

    check_columns = [
        (
            f"{target_name}_{metric}",
            " ".join(
                word.upper() if word.lower() == "ttft" else word.capitalize()
                for word in f"{target_name}_{metric}".split("_")
            )
            + (
                ""
                if metric.endswith("_check") or metric.endswith("_ratio")
                else " (ms)"
                if metric.startswith("ttft")
                else " (TPS)"
                if metric.startswith("tput")
                else ""
            ),
        )
        for target_name in target_checks.keys()
        for metric in ("ttft_check", "tput_user_check", "ttft", "tput_user")
    ]
    check_columns.sort(key=lambda column: not column[0].endswith("_check"))
    return check_columns


def _render_benchmark_table(
    rows: List[Dict[str, Any]], base_columns: List[tuple]
) -> str:
    from workflows.workflow_types import ReportCheckTypes

    prepared_rows = _prepare_renderable_benchmark_rows(rows)
    target_checks = prepared_rows[0].get("target_checks") if prepared_rows else None
    check_columns = _build_benchmark_check_columns(target_checks)
    display_columns = base_columns + check_columns
    round_columns = {column_name for column_name, _ in check_columns}
    flattened_rows = _flatten_target_checks(prepared_rows)
    display_rows = []

    for row in flattened_rows:
        display_row = {}
        for column_name, display_header in display_columns:
            value = row.get(column_name, "N/A")
            if isinstance(value, ReportCheckTypes):
                display_row[display_header] = ReportCheckTypes.to_display_string(value)
            elif column_name in round_columns and isinstance(value, float):
                display_row[display_header] = f"{value:.2f}"
            else:
                display_row[display_header] = str(value)
        display_rows.append(display_row)

    return _render_markdown_table(display_rows)


def _render_text_benchmark_table(rows: List[Dict[str, Any]]) -> str:
    base_columns = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    return _render_benchmark_table(rows, base_columns)


def _render_vlm_benchmark_table(rows: List[Dict[str, Any]]) -> str:
    base_columns = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    return _render_benchmark_table(rows, base_columns)


def _escape_message(message: Any) -> str:
    if not isinstance(message, str):
        message = str(message)
    message = message.replace("|", "\\|")
    message = message.replace("\n", " ").replace("\r", "")
    if len(message) > 250:
        return message[:250] + "..."
    return message


def _format_test_timestamp(timestamp: str) -> str:
    if not timestamp or timestamp == "N/A":
        return "N/A"
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return timestamp


def _build_parameter_support_summary(
    report_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    summary = {}
    results = report_data.get("results", {})
    if not isinstance(results, dict):
        return summary

    for test_case, tests in results.items():
        if not tests:
            summary[test_case] = {
                "status": "SKIP",
                "summary_text": "No tests run.",
                "all_tests": [],
            }
            continue

        total_tests = len(tests)
        passed_tests = [test for test in tests if test.get("status") == "passed"]
        failed_tests = [test for test in tests if test.get("status") == "failed"]
        summary[test_case] = {
            "status": "PASS" if not failed_tests else "FAIL",
            "summary_text": f"{len(passed_tests)}/{total_tests} passed",
            "all_tests": tests,
        }
    return summary


def _render_parameter_support_markdown(
    report_data: Dict[str, Any], heading_level: int
) -> str:
    summary = _build_parameter_support_summary(report_data)
    heading_prefix = "#" * heading_level
    lines = [
        f"{heading_prefix} LLM API Test Metadata",
        "",
        "| Attribute | Value |",
        "| --- | --- |",
        f"| **Endpoint URL** | `{report_data.get('endpoint_url', 'N/A')}` |",
        f"| **Test Timestamp** | {_format_test_timestamp(report_data.get('test_run_timestamp_utc', 'N/A'))} |",
        "",
        f"{heading_prefix} Parameter Conformance Summary",
        "",
        "| Test Case | Status | Summary |",
        "| --- | :---: | --- |",
    ]

    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        lines.append(
            f"| `{test_case}` | {result['status']} | {result['summary_text']} |"
        )

    lines.extend(
        [
            "",
            f"{heading_prefix} Detailed Test Results",
            "",
            "| Test Case | Parametrization | Status | Message |",
            "| --- | --- | :---: | --- |",
        ]
    )

    has_results = False
    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        if not result["all_tests"]:
            continue
        has_results = True
        sorted_tests = sorted(
            result["all_tests"],
            key=lambda test: (
                test.get("status") == "passed",
                test.get("test_node_name", ""),
            ),
        )
        for test in sorted_tests:
            status = str(test.get("status", "")).upper()
            message = (
                _escape_message(test.get("message", "")) if status == "FAILED" else ""
            )
            lines.append(
                f"| `{test_case}` | `{test.get('test_node_name', '')}` | {status} | {message} |"
            )

    if not has_results:
        lines.append("| No results | | | |")

    return "\n".join(lines)


def render_benchmark_summary_markdown(
    entry: Dict[str, Any], heading_level: int = 3
) -> str:
    """Render benchmark target tables for one release-performance entry."""
    sections = []
    split_rows = _split_benchmark_rows(entry)
    heading_prefix = "#" * heading_level
    subheading_prefix = "#" * (heading_level + 1)

    if split_rows["text"] or split_rows["vlm"]:
        sections.append(f"{heading_prefix} Performance Benchmark Targets")

    if split_rows["text"]:
        sections.append(
            f"{subheading_prefix} Text-to-Text\n\n"
            f"{_render_text_benchmark_table(split_rows['text'])}"
        )
    if split_rows["vlm"]:
        sections.append(
            f"{subheading_prefix} VLM\n\n"
            f"{_render_vlm_benchmark_table(split_rows['vlm'])}"
        )

    return "\n\n".join(sections)


def render_test_results_markdown(entry: Dict[str, Any], heading_level: int = 3) -> str:
    """Render parameter support test results for one release-performance entry."""
    return ""


def get_release_report_data(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    report_data = entry.get("report_data")
    if isinstance(report_data, dict) and report_data:
        return report_data
    return None


def _format_release_report_source_line(entry: Dict[str, Any]) -> Optional[str]:
    ci_run_url = entry.get("ci_run_url")
    ci_job_url = entry.get("ci_job_url")
    source_links = []
    if ci_run_url:
        source_links.append(f"[CI run]({ci_run_url})")
    if ci_job_url:
        source_links.append(f"[CI job]({ci_job_url})")
    if not source_links:
        return None
    return f"Source: {', '.join(source_links)}"


def _inject_release_report_source_line(
    report_markdown: str, source_line: Optional[str]
) -> str:
    if not report_markdown or not source_line:
        return report_markdown

    lines = report_markdown.splitlines()
    if len(lines) >= 2 and lines[1] == "":
        return "\n".join([lines[0], "", source_line, ""] + lines[2:]).strip()
    return "\n".join([lines[0], "", source_line, ""] + lines[1:]).strip()


def _render_legacy_release_report_section(
    entry: Dict[str, Any],
    section_heading_level: int,
    title: str,
) -> str:
    """Preserve the old perf-target-only renderer for legacy baseline entries."""
    sections = []
    benchmark_markdown = render_benchmark_summary_markdown(
        entry, heading_level=section_heading_level + 1
    )
    test_results_markdown = render_test_results_markdown(
        entry, heading_level=section_heading_level + 1
    )
    if benchmark_markdown:
        sections.append(benchmark_markdown)
    if test_results_markdown:
        sections.append(test_results_markdown)
    if not sections:
        return ""

    heading_prefix = "#" * section_heading_level
    report_lines = [f"{heading_prefix} {title}", ""]
    source_line = _format_release_report_source_line(entry)
    if source_line:
        report_lines.append(source_line)
        report_lines.append("")
    report_lines.append("\n\n".join(sections))
    return "\n".join(report_lines).strip()


def render_release_report_section(
    entry: Dict[str, Any],
    section_heading_level: int = 2,
    title: str = "Release Report",
) -> str:
    """Render the combined release report section for one entry."""
    report_markdown = str(entry.get("report_markdown") or "").strip()
    if report_markdown:
        return _inject_release_report_source_line(
            report_markdown, _format_release_report_source_line(entry)
        )
    return _render_legacy_release_report_section(entry, section_heading_level, title)


def render_release_notes_performance_markdown(
    release_performance_data: Dict[str, Any],
) -> str:
    """Render the release-notes performance section from the baseline data."""
    sections = []
    for (
        model_name,
        device,
        impl_id,
        inference_engine,
        entry,
    ) in iter_release_performance_items(release_performance_data):
        body_sections = []
        benchmark_markdown = render_benchmark_summary_markdown(entry, heading_level=4)
        test_results_markdown = render_test_results_markdown(entry, heading_level=4)
        if benchmark_markdown:
            body_sections.append(benchmark_markdown)
        if test_results_markdown:
            body_sections.append(test_results_markdown)
        if not body_sections:
            continue

        sections.append(
            "\n\n".join(
                [
                    (f"### {model_name} on {device} ({impl_id}, {inference_engine})"),
                    "\n\n".join(body_sections),
                ]
            )
        )

    return "\n\n".join(sections)

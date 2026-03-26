#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

try:
    from release_diff import ReleaseDiffRecord
except ImportError:
    from scripts.release.release_diff import ReleaseDiffRecord


def _require_single_unique_value(
    values: Iterable[Optional[str]], field_name: str, source_label: str
) -> str:
    """Require exactly one non-empty unique value for one release field."""
    unique_values = sorted(
        {value.strip() for value in values if isinstance(value, str) and value.strip()}
    )
    if not unique_values:
        raise ValueError(
            f"{source_label} does not contain any non-empty {field_name} values."
        )
    if len(unique_values) > 1:
        formatted_values = ", ".join(unique_values)
        raise ValueError(
            f"{source_label} contains multiple {field_name} values: {formatted_values}"
        )
    return unique_values[0]


def load_release_diff_records(release_diff_path: Path) -> List[ReleaseDiffRecord]:
    """Load the pre-release diff JSON that drives release dispatch."""
    if not release_diff_path.exists():
        raise FileNotFoundError(f"Pre-release diff JSON not found: {release_diff_path}")

    with release_diff_path.open("r", encoding="utf-8") as file:
        records = cast(List[ReleaseDiffRecord], json.load(file))

    if not isinstance(records, list):
        raise ValueError(
            f"Pre-release diff JSON must contain a list of records: {release_diff_path}"
        )

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                f"Pre-release diff JSON contains a non-object record: {release_diff_path}"
            )
        if "template_key" not in record or "inference_engine" not in record:
            raise ValueError(
                "Release diff JSON is missing `template_key` or `inference_engine`. "
                "Regenerate it with update_model_spec.py before dispatch."
            )

    return records


def resolve_release_workflow_refs(release_diff_path: Path) -> Tuple[str, str]:
    """Resolve the single tt-metal and vLLM refs required by release.yml."""
    records = load_release_diff_records(release_diff_path)
    source_label = str(release_diff_path)
    tt_metal_ref = _require_single_unique_value(
        (record.get("tt_metal_commit_after") for record in records),
        "tt_metal_commit_after",
        source_label,
    )
    vllm_ref = _require_single_unique_value(
        (record.get("vllm_commit_after") for record in records),
        "vllm_commit_after",
        source_label,
    )
    return tt_metal_ref, vllm_ref


def _load_models_ci_config(models_ci_config_path: Path) -> Dict[str, object]:
    """Load the Models CI configuration JSON object."""
    if not models_ci_config_path.exists():
        raise FileNotFoundError(
            f"Models CI config JSON not found: {models_ci_config_path}"
        )

    with models_ci_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Models CI config must be a JSON object: {models_ci_config_path}"
        )
    return cast(Dict[str, object], config)


def _load_models_ci_models(models_ci_config_path: Path) -> Dict[str, object]:
    """Load the `models` mapping from the Models CI configuration JSON."""
    config = _load_models_ci_config(models_ci_config_path)

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError(
            f"Models CI config is missing a top-level `models` object: {models_ci_config_path}"
        )
    return cast(Dict[str, object], models)


def _normalize_device_list(devices: object, context: str) -> List[str]:
    """Validate one device list and normalize whitespace."""
    if not isinstance(devices, list) or not devices:
        raise ValueError(f"{context} must be a non-empty list of device names.")

    normalized_devices = []
    for device in devices:
        if not isinstance(device, str) or not device.strip():
            raise ValueError(f"{context} contains an invalid device name: {device!r}")
        normalized_devices.append(device.strip())
    return normalized_devices


def _require_model_config_entry(
    entry_name: str, config_entry: object, models_ci_config_path: Path
) -> Dict[str, object]:
    """Require one model config entry to be a JSON object."""
    if not isinstance(config_entry, dict):
        raise ValueError(
            f"Models CI config entry {entry_name!r} must be a JSON object: "
            f"{models_ci_config_path}"
        )
    return cast(Dict[str, object], config_entry)


def _require_ci_config(
    entry_name: str, config_entry: Dict[str, object], models_ci_config_path: Path
) -> Dict[str, object]:
    """Require one model config entry to contain a `ci` object."""
    ci_config = config_entry.get("ci")
    if not isinstance(ci_config, dict):
        raise ValueError(
            f"Models CI config entry {entry_name!r} is missing a `ci` object: "
            f"{models_ci_config_path}"
        )
    return cast(Dict[str, object], ci_config)


def _filter_release_device_args(
    release_config: Dict[str, object], allowed_devices: List[str], entry_name: str
) -> None:
    """Drop release device-args that do not apply to the kept release devices."""
    device_args = release_config.get("device-args")
    if device_args is None:
        return
    if not isinstance(device_args, dict):
        raise ValueError(
            f"Models CI release config for {entry_name!r} has an invalid "
            "`device-args` object."
        )

    filtered_device_args = {
        device_name: value
        for device_name, value in device_args.items()
        if device_name in allowed_devices
    }
    if filtered_device_args:
        release_config["device-args"] = filtered_device_args
        return
    release_config.pop("device-args", None)


def _build_model_name_candidates(record: ReleaseDiffRecord) -> List[str]:
    """Return config-key candidates derived from one release diff record."""
    candidate_names: List[str] = []
    for weight_name in record.get("weights", []):
        candidate_name = Path(weight_name).name.strip()
        if candidate_name and candidate_name not in candidate_names:
            candidate_names.append(candidate_name)

    model_arch = str(record.get("model_arch", "")).strip()
    if model_arch and model_arch not in candidate_names:
        candidate_names.append(model_arch)
    return candidate_names


def _find_matching_models_ci_entry_name(
    record: ReleaseDiffRecord, models: Dict[str, object], models_ci_config_path: Path
) -> str:
    """Match one release diff record to exactly one Models CI config entry."""
    inference_engine = str(record.get("inference_engine", "")).strip()
    if not inference_engine:
        raise ValueError(
            f"Release diff record is missing inference_engine: {record.get('template_key')}"
        )

    candidate_names = _build_model_name_candidates(record)
    matched_names = []
    for model_name, config_entry in models.items():
        if model_name not in candidate_names:
            continue
        config_entry = _require_model_config_entry(
            model_name, config_entry, models_ci_config_path
        )
        if str(config_entry.get("inference_engine", "")).strip() == inference_engine:
            matched_names.append(model_name)

    if len(matched_names) != 1:
        candidate_text = ", ".join(candidate_names) or "<none>"
        matched_text = ", ".join(matched_names) or "<none>"
        raise ValueError(
            "Failed to match exactly one Models CI config entry for release diff record "
            f"{record.get('template_key')}. Candidates={candidate_text}; "
            f"matched={matched_text}; inference_engine={inference_engine}; "
            f"config={models_ci_config_path}"
        )
    return matched_names[0]


def collect_release_devices_by_config_entry(
    release_diff_path: Path, models_ci_config_path: Path
) -> Dict[str, List[str]]:
    """Collect expected release devices for each matched Models CI config entry."""
    records = load_release_diff_records(release_diff_path)
    models = _load_models_ci_models(models_ci_config_path)
    devices_by_entry_name: Dict[str, List[str]] = {}

    for record in records:
        entry_name = _find_matching_models_ci_entry_name(
            record, models, models_ci_config_path
        )
        expected_devices = devices_by_entry_name.setdefault(entry_name, [])
        for device in _normalize_device_list(
            record.get("devices"),
            f"Release diff devices for {record.get('template_key')}",
        ):
            if device not in expected_devices:
                expected_devices.append(device)

    return devices_by_entry_name


def prune_release_models_ci_config(
    release_diff_path: Path, models_ci_config_path: Path
) -> Dict[str, List[str]]:
    """Keep `ci.release` only for models/devices needed by the release dispatch."""
    devices_by_entry_name = collect_release_devices_by_config_entry(
        release_diff_path, models_ci_config_path
    )
    config = _load_models_ci_config(models_ci_config_path)
    models = cast(Dict[str, object], config["models"])

    for entry_name, config_entry in models.items():
        config_entry = _require_model_config_entry(
            entry_name, config_entry, models_ci_config_path
        )
        ci_config = _require_ci_config(entry_name, config_entry, models_ci_config_path)
        expected_devices = devices_by_entry_name.get(entry_name)
        if expected_devices is None:
            ci_config.pop("release", None)
            continue

        release_config = ci_config.get("release")
        if release_config is None:
            release_config = {}
            ci_config["release"] = release_config
        elif not isinstance(release_config, dict):
            raise ValueError(
                f"Models CI config entry {entry_name!r} has a non-object "
                f"`ci.release`: {models_ci_config_path}"
            )

        release_config = cast(Dict[str, object], release_config)
        release_config["devices"] = list(expected_devices)
        _filter_release_device_args(release_config, expected_devices, entry_name)

    models_ci_config_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return devices_by_entry_name


def validate_release_models_ci_config(
    release_diff_path: Path, models_ci_config_path: Path
) -> Dict[str, List[str]]:
    """Require the Models CI release config to match the pre-release diff exactly."""
    devices_by_entry_name = collect_release_devices_by_config_entry(
        release_diff_path, models_ci_config_path
    )
    models = _load_models_ci_models(models_ci_config_path)
    unexpected_release_entries = []

    for entry_name, expected_devices in devices_by_entry_name.items():
        config_entry = _require_model_config_entry(
            entry_name, models[entry_name], models_ci_config_path
        )
        ci_config = _require_ci_config(entry_name, config_entry, models_ci_config_path)
        release_config = ci_config.get("release")
        if not isinstance(release_config, dict):
            raise ValueError(
                f"Models CI config entry {entry_name!r} is missing `ci.release`: "
                f"{models_ci_config_path}"
            )
        actual_devices = _normalize_device_list(
            release_config.get("devices"),
            f"Models CI release devices for {entry_name!r}",
        )
        if actual_devices != expected_devices:
            raise ValueError(
                f"Models CI release devices for {entry_name!r} do not match "
                f"{release_diff_path}. Expected {expected_devices}, found {actual_devices}."
            )

    for entry_name, config_entry in models.items():
        if entry_name in devices_by_entry_name:
            continue
        config_entry = _require_model_config_entry(
            entry_name, config_entry, models_ci_config_path
        )
        ci_config = _require_ci_config(entry_name, config_entry, models_ci_config_path)
        if "release" in ci_config:
            unexpected_release_entries.append(entry_name)

    if unexpected_release_entries:
        formatted_entries = ", ".join(sorted(unexpected_release_entries))
        raise ValueError(
            "Models CI config contains unexpected `ci.release` entries outside the "
            f"dispatch set from {release_diff_path}: {formatted_entries}."
        )

    return devices_by_entry_name

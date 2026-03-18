# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to generate CI matrix from models-ci-config.json.
This script reads model CI configuration and generates
GitHub Actions matrix JSON for the specified workflow type (nightly, weekly, release, etc.)
The config file defines which models run on which devices for each schedule.
"""

import sys
import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

from load_config import load_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Exclusion file paths (relative to script location)
EXCLUSIONS_FILE_PATH = os.path.join(SCRIPT_DIR, "ci_exclusions.yml")
EXCLUSIONS_SCHEMA_PATH = os.path.join(SCRIPT_DIR, "ci-exclusions-schema.json")

# Runner mappings file path
RUNNER_MAPPINGS_FILE_PATH = os.path.join(SCRIPT_DIR, "runner_mappings.yml")
RUNNER_MAPPINGS_SCHEMA_PATH = os.path.join(SCRIPT_DIR, "runner-mappings-schema.json")

SERVER_TYPE_MAPPING = {
    "vLLM": "tt-inference-server",
    "MEDIA": "media-inference-server",
    "FORGE": "forge-media-inference-server",
}


def is_github_ci() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true"


def load_exclusions() -> List[Dict]:
    try:
        config = load_config(EXCLUSIONS_FILE_PATH, EXCLUSIONS_SCHEMA_PATH)
        return config.get('exclusions') or []
    except FileNotFoundError:
        return []


def load_ci_config(config_path: str, schema_path: Optional[str] = None) -> Dict:
    config = load_config(config_path, schema_path)
    return config["models"]


def load_runner_mappings() -> Dict:
    return load_config(RUNNER_MAPPINGS_FILE_PATH, RUNNER_MAPPINGS_SCHEMA_PATH)


def get_server_type(inference_engine: str) -> str:
    """
    Map inference engine to server type.

    Raises ValueError if inference engine is not found in SERVER_TYPE_MAPPING.
    """
    server_type = SERVER_TYPE_MAPPING.get(inference_engine)
    if server_type:
        return server_type

    raise ValueError(
        f"Unknown inference engine '{inference_engine}'. "
        f"Please add it to SERVER_TYPE_MAPPING in generate_ci_matrix.py"
    )


def get_runner_config(schedule: str, device_name: str, model_name: Optional[str] = None, runner_mappings: Optional[Dict] = None) -> Dict[str, str]:
    """
    Get runner configuration for a device and schedule type with model-specific override support.

    Priority order:
    1. Overrides with matching schedule and model (if model_name provided)
    2. Overrides with matching schedule (no model filter)
    3. Default mapping

    Args:
        schedule: Schedule type (nightly, weekly, etc.)
        device_name: Device name (N150, N300, etc.)
        model_name: Optional model name for model-specific overrides
        runner_mappings: Optional pre-loaded runner mappings (for performance)

    Returns:
        Dict with 'label' and 'type' keys

    Raises:
        ValueError if device is not found in any mapping
    """
    mappings = runner_mappings if runner_mappings is not None else load_runner_mappings()

    # Check overrides for this schedule
    for override in mappings.get('overrides', []):
        if override.get('schedule') != schedule:
            continue

        # Priority 1: Model-specific overrides (has 'models' field and model matches)
        if model_name and 'models' in override:
            if model_name in override.get('models', []):
                device_overrides = override.get('devices', {})
                if device_name in device_overrides:
                    return device_overrides[device_name]

    # Priority 2: Schedule-wide overrides (no 'models' field = applies to all models)
    for override in mappings.get('overrides', []):
        if override.get('schedule') != schedule:
            continue

        if 'models' not in override:
            device_overrides = override.get('devices', {})
            if device_name in device_overrides:
                return device_overrides[device_name]

    # Priority 3: Default mapping
    defaults = mappings.get('defaults', {})
    if device_name in defaults:
        return defaults[device_name]

    raise ValueError(
        f"Unknown device '{device_name}'. Please add it to the 'defaults' section "
        f"in {RUNNER_MAPPINGS_FILE_PATH}"
    )


def is_excluded(model_name: str, device_name: str, schedule: str, server_type: str, exclusions: List[Dict]) -> bool:
    """Check if a model/device combination should be excluded."""
    for rule in exclusions:
        if not rule:
            continue

        # Skip rules with no filtering criteria
        if not any(key in rule for key in ['model', 'device', 'schedule', 'server_type']):
            continue

        matches = True

        if 'model' in rule:
            if rule['model'] not in model_name and model_name != rule['model']:
                matches = False

        if 'device' in rule and matches:
            if rule['device'].upper() != device_name.upper():
                matches = False

        if 'schedule' in rule and matches:
            if rule['schedule'].lower() != schedule.lower():
                matches = False

        if 'server_type' in rule and matches:
            if rule['server_type'] != server_type:
                matches = False

        if matches:
            reason = rule.get('reason', 'No reason specified')
            print(f"Excluding {model_name} on {device_name} ({schedule}): {reason}", file=sys.stderr)
            return True

    return False


def generate_matrix(
    schedule: str,
    config_path: str,
    schema_path: Optional[str] = None,
    device_filter: Optional[str] = None,
    server_type_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Generate CI matrix organized by server type and device.
    Args:
        schedule: Schedule type (nightly, weekly, release, bi_weekly)
        config_path: Path to CI config file
        schema_path: Path to CI config schema file
        device_filter: Filter by specific device type (optional)
        server_type_filter: Filter by server type (optional)
    Returns:
        Dict structure: {server_type: {device: {models: [...], runner: {...}}}}
    """
    exclusions = load_exclusions()
    runner_mappings = load_runner_mappings()
    models = load_ci_config(config_path, schema_path)

    # Structure: server_type -> device -> {models: [], runner: {}, model-args: {}}
    matrix: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {"models": [], "runner": None, "model-args": {}}))

    for model_name, model_config in models.items():
        server_type = get_server_type(model_config["inference_engine"])

        if server_type_filter and server_type != server_type_filter:
            continue

        ci_config = model_config["ci"]
        schedule_config = ci_config.get(schedule)  # This particular schedule may not exist for this model

        if not schedule_config:
            continue

        devices = schedule_config["devices"]
        device_args = schedule_config.get("device-args", {})

        if devices == "ALL":
            devices = list(runner_mappings.get('defaults', {}).keys())

        for device_name in devices:
            if device_filter and device_name.upper() != device_filter.upper():
                continue

            if is_excluded(model_name, device_name, schedule, server_type, exclusions):
                continue

            # Get runner config for this specific model/device combination
            runner_config = get_runner_config(schedule, device_name, model_name, runner_mappings)

            # Create a unique key that combines device and runner label
            # This allows different models on the same device to have different runners
            device_key = f"{device_name}_{runner_config['label']}"

            # Add runner config and model to the appropriate group
            if matrix[server_type][device_key]["runner"] is None:
                matrix[server_type][device_key]["runner"] = runner_config

            matrix[server_type][device_key]["models"].append(model_name)

            per_device_args = device_args.get(device_name, {})
            if per_device_args:
                matrix[server_type][device_key]["model-args"][model_name] = per_device_args

    # Convert defaultdict to regular dict for cleaner output
    return {server: dict(devices) for server, devices in matrix.items()}


def _flatten_matrix(nested_matrix: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """
    Helper to flatten nested matrix for GitHub Actions.
    Converts {server_type: {device: {models: [...], runner: {...}}}} to [{model, runner, server_type}, ...].
    Each model gets its own entry in the flat list.

    Args:
        nested_matrix: Nested matrix from generate_matrix()

    Returns:
        List[Dict]: Flat list suitable for GitHub Actions matrix with one entry per model
    """
    flat_list = []
    for server_type, devices in nested_matrix.items():
        for _, config in devices.items():
            model_args = config.get("model-args", {})
            for model in config["models"]:
                entry = {
                    "model": model,
                    "runner": config["runner"],
                    "server_type": server_type,
                }
                per_model_args = model_args.get(model, {})
                if "additional-args" in per_model_args:
                    entry["additional-args"] = per_model_args["additional-args"]
                if "throttle-perf" in per_model_args:
                    entry["throttle-perf"] = per_model_args["throttle-perf"]
                flat_list.append(entry)
    return sorted(flat_list, key=lambda entry: entry["model"])


def main() -> None:
    # Load runner mappings to get valid device choices
    mappings = load_runner_mappings()
    valid_devices = list(mappings.get('defaults', {}).keys())

    parser = argparse.ArgumentParser(description="Generate CI matrix from models-ci-config.json")
    parser.add_argument("--schedule", required=True, choices=["nightly", "weekly", "release", "bi_weekly"],
                        help="Schedule type")
    parser.add_argument("--device", choices=valid_devices,
                        help="Filter by device type")
    parser.add_argument("--server-type", dest="server_type", choices=list(SERVER_TYPE_MAPPING.values()),
                        help="Filter by server type")
    parser.add_argument("--config", dest="config_path", required=True,
                        help="Path to models-ci-config.json")
    parser.add_argument("--schema", dest="schema_path",
                        help="Path to models-ci-config-schema.json (optional, skips validation if omitted)")

    args = parser.parse_args()

    try:
        output = generate_matrix(
            schedule=args.schedule,
            config_path=args.config_path,
            schema_path=args.schema_path,
            device_filter=args.device,
            server_type_filter=args.server_type,
        )

        # GH job matrix expects a list, not a dict, hence we need to flatten the output
        if is_github_ci():
            output = _flatten_matrix(output)
            output_str = json.dumps(output, separators=(',', ':'))
        else:
            output_str = json.dumps(output, indent=2)

        # Print for CLI usage (so shell can capture it)
        print(output_str)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

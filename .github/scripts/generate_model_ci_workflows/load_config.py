# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility module for loading and validating config files (JSON and YAML).
"""

import json
import os
import sys
import yaml
from typing import Optional, Any
from jsonschema import validate, ValidationError


def load_config(config_path: str, schema_path: Optional[str] = None) -> Any:
    """
    Load and optionally validate a config file (JSON or YAML).
    File type is automatically detected from the extension.

    Args:
        config_path: Path to the config file (.json, .yml, .yaml)
        schema_path: Optional path to JSON schema file for validation

    Returns:
        Any: The loaded config data (dict, list, etc.)

    Raises:
        FileNotFoundError: If the config or schema file doesn't exist
        ValueError: If validation fails or unsupported file type
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Determine file type from extension
    ext = os.path.splitext(config_path)[1].lower()

    with open(config_path, 'r') as f:
        if ext == '.json':
            data = json.load(f, object_pairs_hook=_detect_duplicate_keys)
        elif ext in ['.yml', '.yaml']:
            data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .json, .yml, .yaml")

    # Validate against schema if provided
    if schema_path:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path, 'r') as f:
            schema = json.load(f)

        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            error_msg = f"Config validation failed for {config_path}: {e.message}"
            print(f"ERROR: {error_msg}", file=sys.stderr)
            raise ValueError(error_msg)

    return data


def _detect_duplicate_keys(pairs):
    """
    Object hook for json.load() that detects duplicate keys.

    Args:
        pairs: List of (key, value) tuples from JSON parsing

    Returns:
        Dict with no duplicates

    Raises:
        ValueError: If duplicate keys are found
    """
    seen_keys = {}
    for key, value in pairs:
        if key in seen_keys:
            raise ValueError(f"Duplicate key found in JSON: '{key}'")
        seen_keys[key] = value
    return seen_keys

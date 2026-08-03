# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
This module provides utilities to load suite files from /test_suites/*.json.

Supports two suite definition formats:
- "test_suites": Explicit suite definitions (original format)
- "test_matrices": Compact model × device matrix definitions that are
  automatically expanded into the same format as test_suites
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
SERVER_TESTS_CONFIG_FILE = "server_tests_config.json"
TEST_SUITES_DIR = "test_suites"
TEST_SUITES_KEY = "test_suites"
TEST_MATRICES_KEY = "test_matrices"
TEST_SUITE_CATEGORY_KEY = "_category"
CONFIG_DIR = Path(__file__).parent.parent
SUITES_DIR = CONFIG_DIR / TEST_SUITES_DIR
CONFIG_PATH = CONFIG_DIR / SERVER_TESTS_CONFIG_FILE


def load_server_tests_config():
    """
    Load server tests configuration from server_tests_config.json.
    """
    logger.info(f"Loading server tests configuration from {SERVER_TESTS_CONFIG_FILE}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def expand_test_matrices(matrices: list[dict], model_configs: dict) -> list[dict]:
    """
    Expand compact test matrix definitions into individual test suites.

    Each matrix entry defines a cross product of models × devices, producing
    one suite per (model, device) pair. Model properties (weights, compatibility)
    are resolved from model_configs in server_tests_config.json.

    Per-model (and optionally per-model+device) targets are specified via
    ``model_targets`` within each test_case, keeping the override co-located
    with the test it applies to.

    Args:
        matrices: List of matrix definitions from a suite file's "test_matrices" key.
        model_configs: The "model_configs" section from server_tests_config.json.

    Returns:
        List of suite dicts in the same format as manually-defined "test_suites" entries.

    Raises:
        ValueError: If a referenced model is not found in model_configs.
    """
    suites = []

    for matrix in matrices:
        id_pattern = matrix.get("id_pattern", "{model}-{device}")
        models = matrix.get("models", [])
        devices = matrix.get("devices", [])
        matrix_num_devices = matrix.get("num_of_devices")
        base_test_cases = matrix.get("test_cases", [])

        for model_key in models:
            model_cfg = model_configs.get(model_key)
            if not model_cfg:
                raise ValueError(
                    f"Model '{model_key}' referenced in test_matrix "
                    f"but not found in model_configs"
                )

            compatible = model_cfg.get("compatible_devices", [])

            for device in devices:
                if compatible and device not in compatible:
                    logger.warning(
                        f"Skipping {model_key} on {device}: "
                        f"not in compatible_devices {compatible}"
                    )
                    continue

                id_name = model_cfg.get("id_name", model_key)
                suite_id = id_pattern.format(model=id_name, device=device)
                test_cases = deepcopy(base_test_cases)

                _resolve_model_targets(test_cases, model_key, device)

                suite = {
                    "id": suite_id,
                    "weights": list(model_cfg["weights"]),
                    "device": device,
                    "model_marker": model_key,
                    "test_cases": test_cases,
                }

                num_devices = _resolve_num_devices(matrix_num_devices, model_cfg)
                if num_devices is not None:
                    suite["num_of_devices"] = num_devices

                suites.append(suite)

    logger.info(f"Expanded {len(matrices)} test matrices into {len(suites)} suites")
    return suites


def _resolve_model_targets(test_cases: list[dict], model_key: str, device: str) -> None:
    """
    Resolve ``model_targets`` within each test case in-place.

    Each test_case may carry an optional ``model_targets`` dict with
    per-model (or per-model+device) target values. These are merged on
    top of the test_case's base ``targets``, then ``model_targets`` is
    removed from the output.

    Lookup priority (highest wins):
        1. ``model_targets["{model}+{device}"]``
        2. ``model_targets["{model}"]``
        3. Base ``targets`` on the test_case
    """
    for tc in test_cases:
        model_targets = tc.pop("model_targets", None)
        if not model_targets:
            continue

        resolved = dict(tc.get("targets", {}))

        if model_key in model_targets:
            resolved.update(model_targets[model_key])

        specific_key = f"{model_key}+{device}"
        if specific_key in model_targets:
            resolved.update(model_targets[specific_key])

        if resolved:
            tc["targets"] = resolved
        elif "targets" in tc and not tc["targets"]:
            del tc["targets"]


def _resolve_num_devices(matrix_value: Optional[int], model_cfg: dict) -> Optional[int]:
    """Resolve num_of_devices: matrix-level takes priority over model config."""
    if matrix_value is not None:
        return matrix_value
    return model_cfg.get("num_of_devices")


def _load_suites_from_file(
    suite_file: Path, model_configs: dict
) -> tuple[list[dict], dict]:
    """
    Load suites from a single JSON file, expanding any test_matrices.

    Returns (suites, raw_suite_data) tuple: expanded suites list and the
    original parsed JSON for metadata access (e.g. _category).
    """
    with open(suite_file, "r") as f:
        suite_data = json.load(f)

    suites = list(suite_data.get(TEST_SUITES_KEY, []))

    matrices = suite_data.get(TEST_MATRICES_KEY, [])
    if matrices:
        expanded = expand_test_matrices(matrices, model_configs)
        suites.extend(expanded)

    return suites, suite_data


def load_suite_files() -> list[dict]:
    """
    Load and merge test suite files from test_suites/ directory.

    Supports both explicit "test_suites" and compact "test_matrices" definitions.
    Matrices are expanded using model_configs from server_tests_config.json.
    """
    logger.info(f"Loading suite files from {SUITES_DIR}")
    if not SUITES_DIR.exists():
        raise FileNotFoundError(f"test_suites directory not found: {SUITES_DIR}")

    config = load_server_tests_config()
    model_configs = config.get("model_configs", {})
    suite_files = list(SUITES_DIR.glob("*.json"))

    test_suites = []
    for suite_file in sorted(suite_files):
        try:
            suites, suite_data = _load_suites_from_file(suite_file, model_configs)
            if suites:
                category = suite_data.get(TEST_SUITE_CATEGORY_KEY, suite_file.stem)
                logger.info(
                    f"Loaded {len(suites)} suites from {suite_file.name} ({category})"
                )
                test_suites.extend(suites)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in suite file {suite_file}: {e}")
            raise ValueError(f"Invalid JSON in suite file {suite_file}: {e}") from e
        except Exception as e:
            logger.error(f"Error loading suite file {suite_file}: {e}")
            raise RuntimeError(f"Error loading suite file {suite_file}: {e}") from e

    return test_suites


def load_suite_files_by_category(category: str) -> list[dict]:
    """
    Load suites from a specific category file (e.g., 'image' -> image.json).

    Supports both explicit "test_suites" and compact "test_matrices" definitions.

    Args:
        category: Category name (e.g., "image", "audio", "video").

    Returns:
        List of test suite dictionaries from that category file.

    Raises:
        FileNotFoundError: If the category file doesn't exist.
    """
    logger.info(f"Loading suite file for category: {category}")
    suite_file = SUITES_DIR / f"{category.lower()}.json"

    if not suite_file.exists():
        available = [f.stem for f in SUITES_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Suite file not found: {suite_file}. Available categories: {available}"
        )

    config = load_server_tests_config()
    model_configs = config.get("model_configs", {})
    suites, _ = _load_suites_from_file(suite_file, model_configs)

    logger.info(f"Loaded {len(suites)} suites from {suite_file.name}")
    return suites

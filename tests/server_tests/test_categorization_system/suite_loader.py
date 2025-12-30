# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
This module provides utilities to load suite files from test_suites/*.py (preferred)
or test_suites/*.json (fallback for backward compatibility).
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Constants
SERVER_TESTS_CONFIG_FILE = "server_tests_config.json"
TEST_SUITES_DIR = "test_suites"
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


def _load_suites_from_python_module(
    py_file: Path, raise_on_error: bool = False
) -> List[dict]:
    """
    Load test suites from a Python module.

    Looks for a variable ending with '_SUITES' (e.g., AUDIO_SUITES, IMAGE_SUITES).

    Args:
        py_file: Path to the Python file.
        raise_on_error: If True, re-raise import errors instead of returning empty list.

    Returns:
        List of test suite dictionaries, or empty list if not found/import failed.
    """
    if py_file.name.startswith("_"):
        return []

    try:
        module_name = f"server_tests.test_suites.{py_file.stem}"
        module = importlib.import_module(module_name)

        # Look for *_SUITES variable (e.g., AUDIO_SUITES, IMAGE_SUITES)
        for attr_name in dir(module):
            if attr_name.endswith("_SUITES"):
                suites = getattr(module, attr_name)
                if isinstance(suites, list):
                    logger.info(f"Loaded {len(suites)} suites from {py_file.name}")
                    return suites

        logger.warning(f"No *_SUITES variable found in {py_file.name}")
        return []

    except ImportError as e:
        # Missing dependencies - log but don't fail
        logger.warning(f"Import error loading {py_file.name}: {e}")
        if raise_on_error:
            raise
        return []
    except Exception as e:
        logger.warning(f"Failed to load Python suite {py_file}: {e}")
        if raise_on_error:
            raise
        return []


def _load_suites_from_json_file(json_file: Path) -> List[dict]:
    """
    Load test suites from a JSON file.

    Args:
        json_file: Path to the JSON file.

    Returns:
        List of test suite dictionaries.
    """
    try:
        with open(json_file, "r") as f:
            suite_data = json.load(f)

        suites = suite_data.get(TEST_SUITES_DIR, [])
        if suites:
            category = suite_data.get(TEST_SUITE_CATEGORY_KEY, json_file.stem)
            logger.info(
                f"Loaded {len(suites)} suites from {json_file.name} ({category})"
            )
        return suites

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in suite file {json_file}: {e}")
        raise ValueError(f"Invalid JSON in suite file {json_file}: {e}") from e
    except Exception as e:
        logger.error(f"Error loading suite file {json_file}: {e}")
        raise RuntimeError(f"Error loading suite file {json_file}: {e}")


def load_suite_files() -> List[dict]:
    """
    Load and merge test suite files from test_suites/ directory.

    Prioritizes Python modules (*.py) over JSON files (*.json).
    For each category, if a Python module exists, the JSON file is skipped.
    """
    logger.info(f"Loading suite files from {SUITES_DIR}")
    if not SUITES_DIR.exists():
        raise FileNotFoundError(f"test_suites directory not found: {SUITES_DIR}")

    test_suites = []
    loaded_categories = set()

    # First, try to load from Python modules
    py_files = sorted(SUITES_DIR.glob("*.py"))
    for py_file in py_files:
        suites = _load_suites_from_python_module(py_file)
        if suites:
            test_suites.extend(suites)
            loaded_categories.add(py_file.stem)

    # Fallback to JSON for categories not loaded from Python
    json_files = sorted(SUITES_DIR.glob("*.json"))
    for json_file in json_files:
        if json_file.stem in loaded_categories:
            logger.debug(f"Skipping {json_file.name} (already loaded from Python)")
            continue

        suites = _load_suites_from_json_file(json_file)
        if suites:
            test_suites.extend(suites)
            loaded_categories.add(json_file.stem)

    logger.info(
        f"Total suites loaded: {len(test_suites)} from categories: {loaded_categories}"
    )
    return test_suites


def load_suite_files_by_category(category: str) -> List[dict]:
    """
    Load suites from a specific category (Python module preferred, JSON fallback).

    Args:
        category: Category name (e.g., "image", "audio", "forge").

    Returns:
        List of test suite dictionaries from that category.

    Raises:
        FileNotFoundError: If neither Python nor JSON file exists for the category.
        ImportError: If Python module exists but has missing dependencies.
    """
    category_lower = category.lower()

    # Try Python module first
    py_file = SUITES_DIR / f"{category_lower}.py"
    if py_file.exists():
        # When loading by category, raise errors so user knows about missing deps
        suites = _load_suites_from_python_module(py_file, raise_on_error=True)
        if suites:
            return suites

    # Fallback to JSON
    json_file = SUITES_DIR / f"{category_lower}.json"
    if json_file.exists():
        return _load_suites_from_json_file(json_file)

    # Neither found
    available_py = [
        f.stem for f in SUITES_DIR.glob("*.py") if not f.name.startswith("_")
    ]
    available_json = [f.stem for f in SUITES_DIR.glob("*.json")]
    available = list(set(available_py + available_json))

    raise FileNotFoundError(
        f"Suite file not found for category: {category}. Available categories: {available}"
    )

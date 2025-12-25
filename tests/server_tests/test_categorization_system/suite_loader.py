# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
This module provides utilities to load suite files from /test_suites/*.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def load_suite_files() -> list[dict]:
    """
    Load and merge test suite files from test_suites/ directory.
    """
    if not SUITES_DIR.exists():
        raise FileNotFoundError(f"test_suites directory not found: {SUITES_DIR}")

    logger.info(f"Auto-discovering all suite files in {SUITES_DIR}")
    suite_files = list(SUITES_DIR.glob("*.json"))

    test_suites = []
    for suite_file in sorted(suite_files):
        try:
            with open(suite_file, "r") as f:
                suite_data = json.load(f)

            suites = suite_data.get(TEST_SUITES_DIR, [])
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
            raise RuntimeError(f"Error loading suite file {suite_file}: {e}")

    return test_suites


def load_suite_files_by_category(category: str) -> list[dict]:
    """
    Load suites from a specific category file (e.g., 'image' -> image.json).

    Args:
        category: Category name (e.g., "image", "audio", "forge")

    Returns:
        List of test suite dictionaries from that category file

    Raises:
        FileNotFoundError: If the category file doesn't exist
    """
    suite_file = SUITES_DIR / f"{category.lower()}.json"

    if not suite_file.exists():
        available = [f.stem for f in SUITES_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Suite file not found: {suite_file}. Available categories: {available}"
        )

    logger.info(f"Loading suite file for category: {category}")
    with open(suite_file, "r") as f:
        suite_data = json.load(f)

    suites = suite_data.get(TEST_SUITES_DIR, [])
    logger.info(f"Loaded {len(suites)} suites from {suite_file.name}")
    return suites

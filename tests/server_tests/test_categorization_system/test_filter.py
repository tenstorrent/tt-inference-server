# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test filtering utilities for server tests configuration.

This module provides utilities to filter and select tests from server_tests_config.json
based on markers, models, devices, and other criteria. It handles:
- Template expansion from test_templates
- Auto-derivation of markers from model category and hardware
- Prerequisite test injection (DeviceLivenessTest)
- Hardware defaults for num_of_devices and retry_attempts
- Auto-discovery and merging of suite files from test_suites/*.json
"""

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Constants
SERVER_TESTS_CONFIG_FILE = "server_tests_config.json"
TEST_SUITES_DIR = "test_suites"
TEST_SUITE_CATEGORY_KEY = "_category"
TEST_CONFIG = "test_config"
NUM_OF_DEVICES = "num_of_devices"
TARGETS = "targets"


@dataclass
class ExpandedTestCase:
    """Represents a fully expanded test case with all markers and config."""

    name: str
    module: str
    enabled: bool
    description: str
    markers: Set[str]
    test_config: Dict
    targets: Dict
    suite_id: str
    model: str
    device: str


class TestFilter:
    """
    Filter tests from server_tests_config.json based on various criteria.

    The filter handles config format with:
    - test_templates: Reusable test definitions
    - test_suites: Model/device specific test configurations
    - prerequisite_tests: Tests that run before all others (e.g., DeviceLivenessTest)
    - hardware_defaults: Device-specific defaults
    - model_categories: Category to model mappings

    Markers are automatically derived from:
    - Template markers (from test_templates)
    - Model category (IMAGE -> "image", AUDIO -> "audio", etc.)
    - Specific model marker (from suite's model_marker field)
    - Hardware target (device field -> "n150", "t3k", etc.)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config_dict: Optional[Dict] = None,
        suite_files: Optional[List[Path]] = None,
    ):
        """
        Initialize TestFilter with server test configuration.

        Args:
            config_path: Path to server_tests_config.json. If None, uses default location.
            config_dict: Optional dictionary config (overrides config_path).
            suite_files: Optional list of specific suite files to load.
        """
        # Config files are in parent directory (server_tests/)
        self._config_dir = Path(__file__).parent.parent

        if config_dict is not None:
            self.config = config_dict
        elif config_path is None:
            config_path = self._config_dir / SERVER_TESTS_CONFIG_FILE
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            config_path = Path(config_path)
            self._config_dir = config_path.parent
            with open(config_path, "r") as f:
                self.config = json.load(f)

        self.model_categories = self.config.get("model_categories", {})
        self.hardware_defaults = self.config.get("hardware_defaults", {})
        self.test_templates = self.config.get("test_templates", {})
        self.prerequisite_tests = self.config.get("prerequisite_tests", [])
        self.test_suites = self.config.get("test_suites", [])

        # Load suite files: either specific files or auto-discover
        self._load_suite_files(suite_files)

        # Build reverse mapping: model -> category
        self._model_to_category = {}
        for category, models in self.model_categories.items():
            for model in models:
                self._model_to_category[model] = category

        # Expand all test suites
        self.expanded_suites = self._expand_all_suites()
        self.filtered_suites = list(self.expanded_suites)

        # Track if prerequisites should be included
        self._include_prerequisites = True

    def _load_suite_files(self, suite_files: Optional[List[Path]] = None):
        """
        Load and merge test suite files from test_suites/ directory.

        Args:
            suite_files: Optional list of specific suite files to load.
                        If None, discovers all *.json files in test_suites/.
        """
        suites_dir = self._config_dir / TEST_SUITES_DIR
        if not suites_dir.exists():
            raise FileNotFoundError(f"test_suites directory not found: {suites_dir}")

        if suite_files is None:
            logger.info(f"Auto-discovering all suite files in {suites_dir}")
            suite_files = list(suites_dir.glob("*.json"))

        for suite_file in sorted(suite_files):
            if not suite_file.exists():
                logger.error(f"Suite file not found: {suite_file}")
                raise FileNotFoundError(f"Suite file not found: {suite_file}")

            try:
                with open(suite_file, "r") as f:
                    suite_data = json.load(f)

                suites = suite_data.get(TEST_SUITES_DIR, [])
                if suites:
                    category = suite_data.get(TEST_SUITE_CATEGORY_KEY, suite_file.stem)
                    logger.info(
                        f"Loaded {len(suites)} suites from {suite_file.name} ({category})"
                    )
                    self.test_suites.extend(suites)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in suite file {suite_file}: {e}")
                raise json.JSONDecodeError(
                    f"Invalid JSON in suite file {suite_file}: {e}"
                )
            except Exception as e:
                logger.error(f"Error loading suite file {suite_file}: {e}")
                raise RuntimeError(f"Error loading suite file {suite_file}: {e}")

    @classmethod
    def from_category(cls, category: str) -> "TestFilter":
        """
        Create a TestFilter loading only suites from a specific category file.

        This is useful for running tests for a single category (IMAGE, AUDIO, FORGE, etc.)
        without loading other category suites.

        Args:
            category: Category name (e.g., "image", "audio", "forge")

        Returns:
            TestFilter instance with only the specified category's suites

        Example:
            filter = TestFilter.from_category("image")
            filter.filter_by_device("n150")
            tests = filter.get_tests()
        """
        # Config files are in parent directory (server_tests/)
        config_dir = Path(__file__).parent.parent
        suite_file = config_dir / "test_suites" / f"{category.lower()}.json"

        if not suite_file.exists():
            raise FileNotFoundError(
                f"Suite file not found: {suite_file}. "
                f"Available: {list((config_dir / 'test_suites').glob('*.json'))}"
            )

        return cls(suite_files=[suite_file])

    def _get_category_marker(self, model: str) -> Optional[str]:
        """Get the category marker for a model (e.g., 'image', 'audio')."""
        category = self._model_to_category.get(model)
        if category:
            return category.lower()
        return None

    def _expand_test_case(self, test_case: Dict, suite: Dict) -> Dict:
        """
        Expand a test case by merging template defaults with overrides.
        Auto-derives markers from model category, model marker, and hardware.
        """
        template_name = test_case.get("template")
        if not template_name:
            # Direct test case definition (no template)
            return test_case

        template = self.test_templates.get(template_name, {})
        if not template:
            raise ValueError(f"Template '{template_name}' not found in test_templates")

        # Start with template defaults
        expanded = {
            "name": template_name,
            "module": template.get("module"),
            "enabled": test_case.get("enabled", True),
            "description": test_case.get("description", ""),
            "test_config": deepcopy(template.get("test_config", {})),
            "targets": {},
        }

        # Merge test_config overrides
        if TEST_CONFIG in test_case:
            expanded["test_config"].update(test_case["test_config"])

        # Set targets with hardware defaults
        device = suite.get("device", "")
        hw_defaults = self.hardware_defaults.get(device, {})

        # Start with hardware default num_of_devices
        if NUM_OF_DEVICES in hw_defaults:
            expanded["targets"]["num_of_devices"] = hw_defaults["num_of_devices"]

        # Override with test case specific targets
        if TARGETS in test_case:
            expanded["targets"].update(test_case["targets"])

        # Build markers: template + category + model_marker + hardware
        markers = set(template.get("markers", []))

        # Add category marker (e.g., "image", "audio")
        for model in suite.get("weights", []):
            category_marker = self._get_category_marker(model)
            if category_marker:
                markers.add(category_marker)

        # Add model-specific marker (e.g., "sdxl", "whisper")
        model_marker = suite.get("model_marker")
        if model_marker:
            markers.add(model_marker)

        # Add hardware marker
        if device:
            markers.add(device.lower())

        expanded["markers"] = list(markers)

        return expanded

    def _expand_prerequisite_test(self, prereq: Dict, suite: Dict) -> Dict:
        """Expand a prerequisite test with hardware-specific values."""
        logger.info(f"Expanding prerequisite test: {prereq}")
        expanded = deepcopy(prereq)
        device = suite.get("device", "")
        hw_defaults = self.hardware_defaults.get(device, {})

        # Set hardware-specific retry_attempts for liveness test
        if "liveness_retry_attempts" in hw_defaults:
            if "test_config" not in expanded:
                expanded["test_config"] = {}
            expanded["test_config"]["retry_attempts"] = hw_defaults[
                "liveness_retry_attempts"
            ]

        # Set num_of_devices from hardware defaults
        if "targets" not in expanded:
            expanded["targets"] = {}
        if "num_of_devices" in hw_defaults:
            expanded["targets"]["num_of_devices"] = hw_defaults["num_of_devices"]

        # Add markers for model category, model marker, and hardware
        markers = set(expanded.get("markers", []))

        for model in suite.get("weights", []):
            category_marker = self._get_category_marker(model)
            if category_marker:
                markers.add(category_marker)

        model_marker = suite.get("model_marker")
        if model_marker:
            markers.add(model_marker)

        if device:
            markers.add(device.lower())

        expanded["markers"] = list(markers)

        return expanded

    def _expand_all_suites(self) -> List[Dict]:
        """Expand all test suites, injecting prerequisite tests and applying templates."""
        expanded_suites = []

        for suite in self.test_suites:
            expanded_suite = {
                "id": suite.get("id", ""),
                "weights": suite.get("weights", []),
                "device": suite.get("device", ""),
                "model_marker": suite.get("model_marker", ""),
                "test_cases": [],
            }

            # Expand each test case
            for test_case in suite.get("test_cases", []):
                expanded = self._expand_test_case(test_case, suite)
                expanded_suite["test_cases"].append(expanded)

            expanded_suites.append(expanded_suite)

        return expanded_suites

    def _get_prerequisite_for_suite(self, suite: Dict) -> List[Dict]:
        """Get prerequisite tests configured for a specific suite."""
        prereqs = []
        for prereq in self.prerequisite_tests:
            expanded = self._expand_prerequisite_test(prereq, suite)
            prereqs.append(expanded)
        return prereqs

    def include_prerequisites(self, include: bool = True) -> "TestFilter":
        """
        Control whether prerequisite tests (DeviceLivenessTest) are included.

        Args:
            include: If True, prerequisite tests are prepended to results.

        Returns:
            Self for method chaining
        """
        self._include_prerequisites = include
        return self

    def filter_by_model_category(self, categories: List[str]) -> "TestFilter":
        """
        Filter tests by model category (IMAGE, AUDIO, CNN, etc.).

        Args:
            categories: List of category names (e.g., ["IMAGE", "AUDIO"])

        Returns:
            Self for method chaining
        """
        if not categories:
            return self

        # Get all models in the specified categories
        target_models = set()
        for category in categories:
            target_models.update(self.model_categories.get(category, []))

        # Filter suites that match these models
        self.filtered_suites = [
            suite
            for suite in self.filtered_suites
            if any(model in target_models for model in suite.get("weights", []))
        ]

        return self

    def filter_by_model(self, model_name: str) -> "TestFilter":
        """
        Filter tests by specific model name.

        Args:
            model_name: Model name (e.g., "stable-diffusion-xl-base-1.0")

        Returns:
            Self for method chaining
        """
        self.filtered_suites = [
            suite
            for suite in self.filtered_suites
            if model_name in suite.get("weights", [])
        ]

        return self

    def filter_by_device(self, device: str) -> "TestFilter":
        """
        Filter tests by device type.

        Args:
            device: Device name (e.g., "n150", "t3k", "galaxy")

        Returns:
            Self for method chaining
        """
        self.filtered_suites = [
            suite
            for suite in self.filtered_suites
            if suite.get("device", "").lower() == device.lower()
        ]

        return self

    def filter_by_markers(
        self, markers: List[str], match_all: bool = False
    ) -> "TestFilter":
        """
        Filter tests by markers (tags).

        Args:
            markers: List of marker names (e.g., ["load", "slow"])
            match_all: If True, test must have ALL markers. If False, ANY marker matches.

        Returns:
            Self for method chaining
        """
        if not markers:
            return self

        marker_set = set(m.lower() for m in markers)
        filtered_suites = []

        for suite in self.filtered_suites:
            filtered_suite = deepcopy(suite)
            filtered_suite["test_cases"] = []

            for test in suite.get("test_cases", []):
                test_markers = set(m.lower() for m in test.get("markers", []))

                if match_all:
                    if marker_set.issubset(test_markers):
                        filtered_suite["test_cases"].append(test)
                else:
                    if marker_set.intersection(test_markers):
                        filtered_suite["test_cases"].append(test)

            if filtered_suite["test_cases"]:
                filtered_suites.append(filtered_suite)

        self.filtered_suites = filtered_suites
        return self

    def filter_by_test_name(self, test_name: str) -> "TestFilter":
        """
        Filter tests by test class name.

        Args:
            test_name: Test class name (e.g., "ImageGenerationLoadTest")

        Returns:
            Self for method chaining
        """
        filtered_suites = []

        for suite in self.filtered_suites:
            filtered_suite = deepcopy(suite)
            filtered_suite["test_cases"] = [
                test
                for test in suite.get("test_cases", [])
                if test.get("name") == test_name
            ]

            if filtered_suite["test_cases"]:
                filtered_suites.append(filtered_suite)

        self.filtered_suites = filtered_suites
        return self

    def exclude_markers(self, markers: List[str]) -> "TestFilter":
        """
        Exclude tests that have any of the specified markers.

        Args:
            markers: List of marker names to exclude (e.g., ["slow", "heavy"])

        Returns:
            Self for method chaining
        """
        if not markers:
            return self

        exclude_set = set(m.lower() for m in markers)
        filtered_suites = []

        for suite in self.filtered_suites:
            filtered_suite = deepcopy(suite)
            filtered_suite["test_cases"] = []

            for test in suite.get("test_cases", []):
                test_markers = set(m.lower() for m in test.get("markers", []))

                if not exclude_set.intersection(test_markers):
                    filtered_suite["test_cases"].append(test)

            if filtered_suite["test_cases"]:
                filtered_suites.append(filtered_suite)

        self.filtered_suites = filtered_suites
        return self

    def get_tests(self) -> List[Dict]:
        """
        Get the filtered test suites with prerequisite tests prepended.

        Returns:
            List of test suites with expanded test cases.
            Each suite has prerequisite tests (e.g., DeviceLivenessTest) prepended
            if include_prerequisites is True.
        """
        result = []

        for suite in self.filtered_suites:
            result_suite = deepcopy(suite)

            if self._include_prerequisites:
                prereqs = self._get_prerequisite_for_suite(suite)
                result_suite["test_cases"] = prereqs + result_suite["test_cases"]

            result.append(result_suite)

        return result

    def get_flat_tests(self) -> List[Dict]:
        """
        Get a flat list of all test cases (without suite grouping).
        Useful for counting or iterating over all tests.

        Returns:
            List of individual test case dictionaries
        """
        tests = []
        for suite in self.get_tests():
            for test in suite.get("test_cases", []):
                test_copy = deepcopy(test)
                test_copy["_suite_id"] = suite.get("id", "")
                test_copy["_weights"] = suite.get("weights", [])
                test_copy["_device"] = suite.get("device", "")
                tests.append(test_copy)
        return tests

    def print_summary(self):
        """Print a summary of filtered tests."""
        suites = self.get_tests()
        flat_tests = self.get_flat_tests()

        logger.info("Filtered Test Summary:")
        logger.info(f"Total test suites: {len(suites)}")
        logger.info(f"Total individual tests: {len(flat_tests)}")
        logger.info("Test breakdown:")

        for suite in suites:
            models = ", ".join(suite.get("weights", []))
            device = suite.get("device", "unknown")
            test_count = len(suite.get("test_cases", []))
            logger.info(
                f"  [{suite.get('id', 'unknown')}] {models} on {device}: {test_count} tests"
            )

            for test in suite.get("test_cases", []):
                markers = ", ".join(sorted(test.get("markers", [])))
                enabled = "✓" if test.get("enabled", True) else "✗"
                logger.info(f"    {enabled} {test.get('name')}: [{markers}]")
                if test.get("description"):
                    logger.info(f"      └─ {test.get('description')}")

    def get_available_markers(self) -> Dict:
        """Get all available markers defined in config."""
        return self.config.get("available_markers", {})

    def get_all_devices(self) -> List[str]:
        """Get list of all devices in test suites."""
        return list(set(suite.get("device", "") for suite in self.test_suites))

    def get_all_models(self) -> List[str]:
        """Get list of all models in test suites."""
        models = set()
        for suite in self.test_suites:
            models.update(suite.get("weights", []))
        return list(models)


def main():
    """Example usage of TestFilter demonstrating all 4 use cases."""
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Test Filter - Modular Suite Architecture")
    logger.info("=" * 70)

    # Show loaded suites
    filter_base = TestFilter()
    logger.info(f"Total suites loaded: {len(filter_base.test_suites)}")
    logger.info(f"Available devices: {filter_base.get_all_devices()}")
    logger.info(f"Available models: {filter_base.get_all_models()}")

    # Case 1: Run all IMAGE/AUDIO/CNN tests on specific hardware
    logger.info("=" * 70)
    logger.info("CASE 1a: All IMAGE/AUDIO/CNN tests on N150")
    logger.info("=" * 70)
    filter1a = TestFilter()
    filter1a.filter_by_model_category(["IMAGE", "AUDIO", "CNN"]).filter_by_device(
        "n150"
    )
    filter1a.print_summary()

    logger.info("=" * 70)
    logger.info("CASE 1b: All IMAGE tests on ALL hardware")
    logger.info("=" * 70)
    filter1b = TestFilter()
    filter1b.filter_by_model_category(["IMAGE"])
    filter1b.print_summary()

    # Case 2: Run specific model on specific hardware with specific test category
    logger.info("=" * 70)
    logger.info("CASE 2: Img2Img model on N150, only 'load' tests")
    logger.info("=" * 70)
    filter2 = TestFilter()
    filter2.filter_by_model("stable-diffusion-xl-base-1.0-img-2-img").filter_by_device(
        "n150"
    ).filter_by_markers(["load"])
    filter2.print_summary()

    # Case 3: Run specific model + hardware + category + specific test
    logger.info("=" * 70)
    logger.info(
        "CASE 3: Img2Img on N150, 'load' category, ImageGenerationLoadTest only"
    )
    logger.info("=" * 70)
    filter3 = TestFilter()
    filter3.filter_by_model("stable-diffusion-xl-base-1.0-img-2-img").filter_by_device(
        "n150"
    ).filter_by_markers(["load"]).filter_by_test_name("ImageGenerationLoadTest")
    filter3.print_summary()

    # Case 4: Run all tests of a type across all models
    logger.info("=" * 70)
    logger.info("CASE 4a: All 'load' tests across all models")
    logger.info("=" * 70)
    filter4a = TestFilter()
    filter4a.filter_by_markers(["load"])
    filter4a.print_summary()

    logger.info("=" * 70)
    logger.info("CASE 4b: All 'param' tests across all models")
    logger.info("=" * 70)
    filter4b = TestFilter()
    filter4b.filter_by_markers(["param"])
    filter4b.print_summary()

    # Modular loading: Load only specific category file (efficient for CI)
    logger.info("=" * 70)
    logger.info("MODULAR: Load only IMAGE category (from image.json)")
    logger.info("=" * 70)
    try:
        filter5 = TestFilter.from_category("image")
        filter5.print_summary()
    except FileNotFoundError as e:
        logger.warning(f"  (Skipped: {e})")

    logger.info("=" * 70)
    logger.info("MODULAR: Load only AUDIO category, filter to galaxy")
    logger.info("=" * 70)
    try:
        filter6 = TestFilter.from_category("audio")
        filter6.filter_by_device("galaxy")
        filter6.print_summary()
    except FileNotFoundError as e:
        logger.warning(f"  (Skipped: {e})")


if __name__ == "__main__":
    main()

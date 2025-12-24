# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test filtering utilities for server tests configuration.

This module provides utilities to filter and select tests from /test_suites/*.json
based on markers, models, devices, and other criteria.

It handles the following:
- Auto-derivation of markers from model category and hardware
- Prerequisite test injection (DeviceLivenessTest)
- Hardware defaults for num_of_devices and retry_attempts
- Auto-discovery and merging of suite files from test_suites/*.json
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List, Optional

from .suite_loader import (
    load_server_tests_config,
    load_suite_files,
    load_suite_files_by_category,
)

logger = logging.getLogger(__name__)

# Constants
TEST_CONFIG = "test_config"
NUM_OF_DEVICES = "num_of_devices"
TARGETS = "targets"


class TestFilter:
    """
    Filter tests from /test_suites/*.json based on various criteria.

    The filter handles config format with:
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

    def __init__(self, suites: list[dict] = None):
        """
        Initialize TestFilter with server test configuration from /test_suites/*.json.
        """
        logger.info("Loading server tests configuration")
        self.config = load_server_tests_config()

        self.model_categories = self.config.get("model_categories", {})
        self.hardware_defaults = self.config.get("hardware_defaults", {})
        self.test_templates = self.config.get("test_templates", {})
        self.prerequisite_tests = self.config.get("prerequisite_tests", [])

        if suites is None:
            logger.info("Load test suites")
            self.test_suites = load_suite_files()
        else:
            logger.info(f"Using {len(suites)} pre-loaded test suites")
            self.test_suites = suites

        # Build reverse mapping: model -> category
        self._model_to_category = {}
        for category, models in self.model_categories.items():
            for model in models:
                self._model_to_category[model] = category

        # Expand all test suites
        self.expanded_suites = self._expand_all_suites()
        self.filtered_suites = list(self.expanded_suites)

        # Include prerequisites by default
        self._include_prerequisites = True

    @classmethod
    def from_category(cls, category: str) -> TestFilter:
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
        suites = load_suite_files_by_category(category)
        return cls(suites=suites)

    def _get_category_marker(self, model: str) -> Optional[str]:
        """Get the category marker for a model.

        Args:
            model: Model name (e.g., "stable-diffusion-xl-base-1.0")

        Returns:
            Category marker in lowercase (e.g., "image", "audio") or None if not found.
        """
        category = self._model_to_category.get(model)
        if category:
            return category.lower()
        return None

    def _expand_test_case(self, test_case: Dict, suite: Dict) -> Dict:
        """Expand a test case by merging template defaults with overrides.

        Applies template config, hardware defaults, and auto-derives markers
        from model category, model marker, and device.

        Args:
            test_case: Raw test case dict with template reference
            suite: Parent suite containing device and weights info

        Returns:
            Expanded test case dict with all fields populated.
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
        """Expand all test suites by applying templates to each test case.

        Returns:
            List of expanded suite dicts with fully populated test cases.
        """
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
        """Get expanded prerequisite tests for a specific suite.

        Args:
            suite: Suite dict for hardware context

        Returns:
            List of expanded prerequisite test dicts.
        """
        prereqs = []
        for prereq in self.prerequisite_tests:
            expanded = self._expand_prerequisite_test(prereq, suite)
            prereqs.append(expanded)
        return prereqs

    def include_prerequisites(self, include: bool = True) -> TestFilter:
        """
        Control whether prerequisite tests (DeviceLivenessTest) are included.

        Args:
            include: If True, prerequisite tests are prepended to results.

        Returns:
            Self for method chaining
        """
        self._include_prerequisites = include
        return self

    def filter_by_model_category(self, categories: List[str]) -> TestFilter:
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

    def filter_by_model(self, model_name: str) -> TestFilter:
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

    def filter_by_device(self, device: str) -> TestFilter:
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
    ) -> TestFilter:
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

    def filter_by_test_name(self, test_name: str) -> TestFilter:
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

    def exclude_markers(self, markers: List[str]) -> TestFilter:
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
                enabled = "✅" if test.get("enabled", True) else "❌"
                logger.info(f"    {enabled} {test.get('name')}: [{markers}]")
                if test.get("description"):
                    logger.info(f"      └─ {test.get('description')}")

    def get_available_markers(self) -> Dict:
        """Get all available markers defined in server_tests_config.json.

        Returns:
            Dict of marker categories and their descriptions.
        """
        return self.config.get("available_markers", {})

    def get_all_devices(self) -> List[str]:
        """Get list of all unique devices across all test suites.

        Returns:
            List of device names (e.g., ["n150", "t3k", "galaxy"]).
        """
        return list(set(suite.get("device", "") for suite in self.test_suites))

    def get_all_models(self) -> List[str]:
        """Get list of all unique models across all test suites.

        Returns:
            List of model names from suite weights.
        """
        models = set()
        for suite in self.test_suites:
            models.update(suite.get("weights", []))
        return list(models)

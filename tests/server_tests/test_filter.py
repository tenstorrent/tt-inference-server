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
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


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

    The filter handles config v2.0 format with:
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

    Examples:
        # Case 1: Run all IMAGE/AUDIO tests on N150
        filter = TestFilter()
        tests = filter.filter_by_model_category(["IMAGE", "AUDIO"]).filter_by_device("n150").get_tests()

        # Case 2: Run specific model on specific hardware with specific category
        filter = TestFilter()
        tests = filter.filter_by_model("stable-diffusion-xl-base-1.0") \\
                      .filter_by_device("n150") \\
                      .filter_by_markers(["load"]).get_tests()

        # Case 3: Run specific tests
        filter = TestFilter()
        tests = filter.filter_by_model("stable-diffusion-xl-base-1.0") \\
                      .filter_by_device("n150") \\
                      .filter_by_markers(["load"]) \\
                      .filter_by_test_name("ImageGenerationLoadTest").get_tests()

        # Case 4: Run all performance tests
        filter = TestFilter()
        tests = filter.filter_by_markers(["load"]).get_tests()
    """

    def __init__(
        self, config_path: Optional[Path] = None, config_dict: Optional[Dict] = None
    ):
        """
        Initialize TestFilter with server test configuration.

        Supports both v2.0 format (with test_suites and test_templates) and
        legacy format (flat test_cases array).

        Args:
            config_path: Path to server_tests_config.json. If None, uses default location.
            config_dict: Optional dictionary config (overrides config_path). Useful for
                        loading legacy configs or testing.
        """
        if config_dict is not None:
            self.config = config_dict
        elif config_path is None:
            config_path = Path(__file__).parent / "server_tests_config.json"
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            with open(config_path, "r") as f:
                self.config = json.load(f)

        # Detect legacy format (has top-level "test_cases" without "test_suites")
        self._is_legacy_format = self._detect_legacy_format()

        if self._is_legacy_format:
            self._convert_legacy_format()

        self.model_categories = self.config.get("model_categories", {})
        self.hardware_defaults = self.config.get("hardware_defaults", {})
        self.test_templates = self.config.get("test_templates", {})
        self.prerequisite_tests = self.config.get("prerequisite_tests", [])
        self.test_suites = self.config.get("test_suites", [])

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

    def _detect_legacy_format(self) -> bool:
        """Detect if config is in legacy format (flat test_cases array)."""
        has_test_cases = "test_cases" in self.config
        has_test_suites = "test_suites" in self.config
        # Legacy format has test_cases at top level but no test_suites
        return has_test_cases and not has_test_suites

    def _convert_legacy_format(self):
        """Convert legacy format to v2.0 format."""
        legacy_test_cases = self.config.get("test_cases", [])

        # Add markers to legacy test cases based on test name patterns
        for test_case in legacy_test_cases:
            if "markers" not in test_case:
                test_case["markers"] = self._infer_markers_for_legacy_test(test_case)

        # Wrap in a single suite
        self.config["test_suites"] = [
            {
                "id": "legacy",
                "weights": [],
                "device": "",
                "model_marker": "",
                "test_cases": legacy_test_cases,
            }
        ]

        # Disable prerequisite injection for legacy format since DeviceLivenessTest
        # might already be in the test_cases
        if not self.config.get("prerequisite_tests"):
            self.config["prerequisite_tests"] = []

    def _infer_markers_for_legacy_test(self, test_case: Dict) -> List[str]:
        """Infer markers for a legacy test case based on its name and characteristics."""
        markers = []
        name = test_case.get("name", "")

        # Infer test type markers based on test name patterns
        name_lower = name.lower()
        if "liveness" in name_lower:
            markers.extend(["smoke", "functional", "fast", "prerequisite"])
        elif "load" in name_lower:
            markers.extend(["load", "e2e", "slow"])
        elif "param" in name_lower:
            markers.extend(["param", "e2e"])
        elif "stability" in name_lower:
            markers.extend(["stability", "e2e", "slow", "heavy"])
        elif "eval" in name_lower:
            markers.extend(["eval", "e2e", "slow"])

        # Infer model category markers based on test name
        if "image" in name_lower:
            markers.append("image")
        elif "audio" in name_lower or "transcription" in name_lower:
            markers.append("audio")

        return markers

    @classmethod
    def from_legacy_config(cls, config_path: Path) -> "TestFilter":
        """
        Create a TestFilter from a legacy config file.

        Legacy format example:
        {
            "test_cases": [
                {"name": "DeviceLivenessTest", "module": "...", ...},
                {"name": "ImageGenerationLoadTest", "module": "...", ...}
            ]
        }

        Args:
            config_path: Path to legacy config JSON file

        Returns:
            TestFilter instance configured for legacy format
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config_dict=config)

    @classmethod
    def from_test_cases(cls, test_cases: List[Dict]) -> "TestFilter":
        """
        Create a TestFilter from a list of test case dictionaries.

        Useful for programmatic test configuration without a config file.

        Args:
            test_cases: List of test case dictionaries

        Returns:
            TestFilter instance
        """
        config = {"test_cases": test_cases}
        return cls(config_dict=config)

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
        if "test_config" in test_case:
            expanded["test_config"].update(test_case["test_config"])

        # Set targets with hardware defaults
        device = suite.get("device", "")
        hw_defaults = self.hardware_defaults.get(device, {})

        # Start with hardware default num_of_devices
        if "num_of_devices" in hw_defaults:
            expanded["targets"]["num_of_devices"] = hw_defaults["num_of_devices"]

        # Override with test case specific targets
        if "targets" in test_case:
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

    def filter_by_suite_id(self, suite_id: str) -> "TestFilter":
        """
        Filter tests by suite ID.

        Args:
            suite_id: Suite identifier (e.g., "sdxl-n150")

        Returns:
            Self for method chaining
        """
        self.filtered_suites = [
            suite for suite in self.filtered_suites if suite.get("id", "") == suite_id
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

    def get_test_count(self) -> int:
        """
        Get the total count of individual tests (including prerequisites).

        Returns:
            Total number of individual test cases
        """
        return len(self.get_flat_tests())

    def print_summary(self):
        """Print a summary of filtered tests."""
        suites = self.get_tests()
        flat_tests = self.get_flat_tests()

        print("\nFiltered Test Summary:")
        print(f"Total test suites: {len(suites)}")
        print(f"Total individual tests: {len(flat_tests)}")
        print("\nTest breakdown:")

        for suite in suites:
            models = ", ".join(suite.get("weights", []))
            device = suite.get("device", "unknown")
            test_count = len(suite.get("test_cases", []))
            print(
                f"\n  [{suite.get('id', 'unknown')}] {models} on {device}: {test_count} tests"
            )

            for test in suite.get("test_cases", []):
                markers = ", ".join(sorted(test.get("markers", [])))
                enabled = "✓" if test.get("enabled", True) else "✗"
                print(f"    {enabled} {test.get('name')}: [{markers}]")
                if test.get("description"):
                    print(f"      └─ {test.get('description')}")

    def reset(self) -> "TestFilter":
        """Reset all filters."""
        self.filtered_suites = list(self.expanded_suites)
        self._include_prerequisites = True
        return self

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
    print("=" * 70)
    print("Test Filter Examples - All 4 Use Cases")
    print("=" * 70)

    # Use Case 1: Run all IMAGE/AUDIO tests on specific hardware
    print("\n" + "=" * 70)
    print("USE CASE 1: All IMAGE tests on N150")
    print("=" * 70)
    filter1 = TestFilter()
    filter1.filter_by_model_category(["IMAGE"]).filter_by_device("n150")
    filter1.print_summary()

    # Use Case 2: Run specific model on specific hardware with specific category
    print("\n" + "=" * 70)
    print("USE CASE 2: SDXL load tests on T3K")
    print("=" * 70)
    filter2 = TestFilter()
    filter2.filter_by_model("stable-diffusion-xl-base-1.0").filter_by_device(
        "t3k"
    ).filter_by_markers(["load"])
    filter2.print_summary()

    # Use Case 3: Run specific test with full filtering
    print("\n" + "=" * 70)
    print("USE CASE 3: Specific ImageGenerationLoadTest for SDXL on N150")
    print("=" * 70)
    filter3 = TestFilter()
    filter3.filter_by_model("stable-diffusion-xl-base-1.0").filter_by_device(
        "n150"
    ).filter_by_markers(["load"]).filter_by_test_name("ImageGenerationLoadTest")
    filter3.print_summary()

    # Use Case 4: Run all performance/load tests across all models
    print("\n" + "=" * 70)
    print("USE CASE 4: All load tests across all models")
    print("=" * 70)
    filter4 = TestFilter()
    filter4.filter_by_markers(["load"])
    filter4.print_summary()

    # Additional examples
    print("\n" + "=" * 70)
    print("EXTRA: Fast smoke tests only (excluding slow/heavy)")
    print("=" * 70)
    filter5 = TestFilter()
    filter5.filter_by_markers(["smoke"]).exclude_markers(["slow", "heavy"])
    filter5.print_summary()

    print("\n" + "=" * 70)
    print("EXTRA: All audio tests on Galaxy")
    print("=" * 70)
    filter6 = TestFilter()
    filter6.filter_by_model_category(["AUDIO"]).filter_by_device("galaxy")
    filter6.print_summary()


if __name__ == "__main__":
    main()

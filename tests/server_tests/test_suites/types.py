# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Type definitions for test suites.

This module provides Python-native types for defining test suites,
enabling IDE features like Find Usages, Go to Definition, and type checking.

Uses TYPE_CHECKING to allow IDE features without triggering runtime imports
of test classes (which may have heavy dependencies like aiohttp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    # These imports only happen during static analysis/IDE features
    pass


class Device(str, Enum):
    """Hardware device targets."""

    N150 = "n150"
    N300 = "n300"
    T3K = "t3k"
    GALAXY = "galaxy"


@dataclass
class TestConfig:
    """Configuration for test execution."""

    test_timeout: int = 3600
    retry_attempts: int = 1
    retry_delay: int = 60
    break_on_failure: bool = False
    mock_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_timeout": self.test_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "break_on_failure": self.break_on_failure,
            "mock_mode": self.mock_mode,
        }


@dataclass
class TestCase:
    """
    A single test case definition.

    Uses actual class references instead of string module paths,
    enabling IDE features like Find Usages and Go to Definition.

    The test_class can be:
    - A Type reference (for IDE features + runtime)
    - A string "module.ClassName" (for lazy loading without imports)
    """

    test_class: Union[Type, str]  # Class reference or "module.ClassName" string
    description: str
    targets: Dict[str, Any] = field(default_factory=dict)
    config: Optional[TestConfig] = None
    enabled: bool = True
    markers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by TestFilter/run.py."""
        # Handle both Type references and string paths
        if isinstance(self.test_class, str):
            # String format: "module.ClassName"
            parts = self.test_class.rsplit(".", 1)
            if len(parts) == 2:
                module, name = parts
            else:
                module = ""
                name = self.test_class
        else:
            # Type reference
            module = self.test_class.__module__
            name = self.test_class.__name__

        result = {
            "name": name,
            "module": module,
            "description": self.description,
            "targets": self.targets.copy(),
            "enabled": self.enabled,
            "markers": self.markers.copy(),
        }

        if self.config:
            result["test_config"] = self.config.to_dict()

        return result


@dataclass
class TestSuite:
    """
    A test suite grouping multiple test cases for a model/device combination.
    """

    id: str
    weights: List[str]  # Model weight names
    device: Device
    model_marker: str
    test_cases: List[TestCase]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by TestFilter."""
        return {
            "id": self.id,
            "weights": self.weights.copy(),
            "device": self.device.value,
            "model_marker": self.model_marker,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }


def suites_to_dicts(suites: List[TestSuite]) -> List[Dict[str, Any]]:
    """Convert a list of TestSuite objects to dict format."""
    return [suite.to_dict() for suite in suites]


# =========================================================================
# Test class references using string paths (avoids import-time dependencies)
# These can be used in suite definitions for lazy loading
# =========================================================================


class TestClasses:
    """
    String references to test classes for lazy loading.

    Use these in TestCase definitions to avoid importing test modules
    at suite definition time. IDE features like Find All References
    will still work on these string constants.
    """

    # Audio tests
    AudioTranscriptionLoadTest = "server_tests.test_cases.audio_transcription_load_test.AudioTranscriptionLoadTest"
    AudioTranscriptionParamTest = "server_tests.test_cases.audio_transcription_param_test.AudioTranscriptionParamTest"

    # Image tests
    ImageGenerationLoadTest = (
        "server_tests.test_cases.image_generation_load_test.ImageGenerationLoadTest"
    )
    ImageGenerationParamTest = (
        "server_tests.test_cases.image_generation_param_test.ImageGenerationParamTest"
    )

    # Utility tests
    DeviceLivenessTest = (
        "server_tests.test_cases.device_liveness_test.DeviceLivenessTest"
    )
    DeviceStabilityTest = (
        "server_tests.test_cases.device_stability_test.DeviceStabilityTest"
    )

    # Mock tests
    MockPassingTest = "server_tests.test_cases.mock_test.MockPassingTest"
    MockFailingTest = "server_tests.test_cases.mock_test.MockFailingTest"
    MockConditionalTest = "server_tests.test_cases.mock_test.MockConditionalTest"

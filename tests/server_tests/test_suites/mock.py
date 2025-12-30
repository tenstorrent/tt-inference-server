# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Mock test suites for testing the test framework itself.

Uses Python-native types with TestClasses references for:
- Cmd+Click to navigate to TestClasses definition
- Find Usages on TestClasses constants
- No import-time dependency loading
"""

from server_tests.test_suites.types import (
    Device,
    TestCase,
    TestClasses,
    TestConfig,
    TestSuite,
    suites_to_dicts,
)

# Shared config for fast mock tests
MOCK_CONFIG = TestConfig(
    test_timeout=10,
    retry_attempts=0,
    retry_delay=1,
    mock_mode=True,
)

# Define suites using proper Python types
_MOCK_SUITE_OBJECTS = [
    TestSuite(
        id="mock-tests-n150",
        weights=["mock-model"],
        device=Device.N150,
        model_marker="mock",
        test_cases=[
            TestCase(
                test_class=TestClasses.MockPassingTest,  # ← Find Usages works!
                description="Simple passing test",
                targets={"expected_value": 42, "delay_seconds": 0.1},
                config=MOCK_CONFIG,
                markers=["mock", "fast"],
            ),
            TestCase(
                test_class=TestClasses.MockConditionalTest,  # ← Cmd+Click works!
                description="Conditional test that passes (value > threshold)",
                targets={"value": 75, "threshold": 50},
                config=MOCK_CONFIG,
                markers=["mock", "fast"],
            ),
            TestCase(
                test_class=TestClasses.MockConditionalTest,
                description="Conditional test that fails (value < threshold)",
                targets={"value": 30, "threshold": 50},
                config=MOCK_CONFIG,
                markers=["mock", "fast"],
            ),
        ],
    ),
]

# Export as dict format for backward compatibility with suite_loader
MOCK_SUITES = suites_to_dicts(_MOCK_SUITE_OBJECTS)

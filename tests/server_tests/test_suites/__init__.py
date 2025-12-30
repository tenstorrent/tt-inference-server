# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test suite definitions using Python-native types.

This package provides type-safe test suite definitions that enable:
- Cmd+Click / Go to Definition
- Find Usages / Find All References
- Type checking with mypy
- IDE autocomplete

Usage:
    from server_tests.test_suites.types import TestCase, TestSuite, TestClasses, Device

    suite = TestSuite(
        id="my-suite",
        weights=["model-weights"],
        device=Device.N150,
        model_marker="my_model",
        test_cases=[
            TestCase(
                test_class=TestClasses.MockPassingTest,
                description="My test",
                targets={"key": "value"},
            ),
        ],
    )
"""

from server_tests.test_suites.types import (
    Device,
    TestCase,
    TestClasses,
    TestConfig,
    TestSuite,
    suites_to_dicts,
)

__all__ = [
    "Device",
    "TestCase",
    "TestClasses",
    "TestConfig",
    "TestSuite",
    "suites_to_dicts",
]

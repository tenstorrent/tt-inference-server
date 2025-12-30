# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Mock test cases for testing the test framework itself.
These tests don't require real hardware and are used for verification.
"""

import logging
import time

from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)


class MockPassingTest(BaseTest):
    """A mock test that always passes."""

    async def _run_specific_test_async(self):
        logger.info(f"MockPassingTest running with targets: {self.targets}")

        # Simulate some work
        delay = self.targets.get("delay_seconds", 0.1)
        time.sleep(delay)

        expected_value = self.targets.get("expected_value", 42)
        actual_value = expected_value  # Always matches

        logger.info(f"MockPassingTest completed: expected={expected_value}, actual={actual_value}")

        return {
            "success": True,
            "expected_value": expected_value,
            "actual_value": actual_value,
            "delay_seconds": delay,
        }


class MockFailingTest(BaseTest):
    """A mock test that always fails."""

    async def _run_specific_test_async(self):
        logger.info(f"MockFailingTest running with targets: {self.targets}")

        expected_value = self.targets.get("expected_value", 100)
        actual_value = self.targets.get("actual_value", 0)  # Intentionally wrong

        logger.info(f"MockFailingTest completed: expected={expected_value}, actual={actual_value}")

        return {
            "success": False,
            "expected_value": expected_value,
            "actual_value": actual_value,
            "reason": "Intentional failure for testing",
        }


class MockConditionalTest(BaseTest):
    """A mock test that passes or fails based on targets."""

    async def _run_specific_test_async(self):
        logger.info(f"MockConditionalTest running with targets: {self.targets}")

        threshold = self.targets.get("threshold", 50)
        value = self.targets.get("value", 75)

        success = value >= threshold

        logger.info(f"MockConditionalTest: value={value}, threshold={threshold}, success={success}")

        return {
            "success": success,
            "value": value,
            "threshold": threshold,
        }


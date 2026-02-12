# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TestCase:
    """Represents a test case with its configuration"""

    test: Any  # BaseTest instance
    targets: list = None


@dataclass
class TestTarget:
    """Represents a test target (e.g., endpoint, service, etc.)"""

    name: str
    url: Optional[str] = None
    port: Optional[int] = None


class TestConfig:
    """Configuration for test execution"""

    def __init__(self, config_dict=None):
        self.config = config_dict or {}
        self.test_timeout = self.config.get("timeout", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 3)
        self.break_on_failure = self.config.get("break_on_failure", True)

    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return self.config.get(key, default)

    def __str__(self):
        """String representation of TestConfig"""
        return (
            f"TestConfig(timeout={self.test_timeout}s, "
            f"retry_attempts={self.retry_attempts}, "
            f"retry_delay={self.retry_delay}s, "
            f"break_on_failure={self.break_on_failure})"
        )

    @classmethod
    def create_default(cls):
        """Create default test configuration"""
        return cls(
            {
                "timeout": 300,  # 5 minutes
                "max_retries": 3,  # 3 retry attempts
                "retry_delay": 5,  # 5 seconds between retries
                "break_on_failure": True,
                "retry_attempts": 10,
            }
        )


class TestReport:
    """Represents the result of a test execution"""

    def __init__(
        self,
        test_name: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
        targets=None,
        result: Any = None,
        logs: List = None,
        attempts: int = 1,
        descrtiption: str = "",
    ):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.error = error
        self.targets = targets
        self.result = result  # Store the actual test result
        self.logs = logs or []  # Store test execution logs
        self.attempts = attempts  # Store number of attempts made
        self.timestamp = time.time()
        self.description = descrtiption

    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        return f"{status} {self.test_name} ({self.duration:.2f}s, {self.attempts} attempts)"

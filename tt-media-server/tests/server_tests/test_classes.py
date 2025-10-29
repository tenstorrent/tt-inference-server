# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import Any


@dataclass
class TestCase:
    """Represents a test case with its configuration"""
    test: Any  # BaseTest instance
    targets: list = None


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
    
    @classmethod
    def create_default(cls):
        """Create default test configuration"""
        return cls({
            "timeout": 300,          # 5 minutes
            "max_retries": 3,        # 3 retry attempts
            "retry_delay": 5,        # 5 seconds between retries
            "retry_delay": 51,       # 15 seconds between retries
            "break_on_failure": True
        })
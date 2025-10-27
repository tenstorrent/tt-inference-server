#!/usr/bin/env python3

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.server_tests.server_runner import ServerRunner
from tests.server_tests.test_config import TestConfig, TestCase
from tests.server_tests.media_server_liveness_test import MediaServerLivenessTest


@dataclass
class TestTarget:
    """Represents a test target (e.g., endpoint, service, etc.)"""
    name: str
    url: Optional[str] = None
    port: Optional[int] = None


class TestReport:
    """Represents the result of a test execution"""
    def __init__(self, test_name: str, success: bool, duration: float, error: Optional[str] = None, targets=None):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.error = error
        self.targets = targets
        self.timestamp = time.time()

    def __str__(self):
        status = "✓ PASS" if self.success else "✗ FAIL"
        return f"{status} {self.test_name} ({self.duration:.2f}s)"


def create_test_config():
    """Create a test configuration"""
    config = TestConfig()
    config.test_timeout = int(os.getenv("TEST_TIMEOUT", "60"))  # 1 minute default
    config.retry_attempts = int(os.getenv("TEST_RETRIES", "20"))  # 2 retries
    config.break_on_failure = os.getenv("BREAK_ON_FAILURE", "false").lower() == "true"

    return config


def create_test_cases():
    """Create test cases to run"""
    config = create_test_config()
    
    # Create test targets
    targets = [
        TestTarget(name="media_server", port=int(os.getenv("SERVICE_PORT", "8000")))
    ]
    
    # Create test cases
    test_cases = []
    
    # Add liveness test
    liveness_test = MediaServerLivenessTest(config, targets)
    liveness_case = TestCase()
    liveness_case.test = liveness_test
    liveness_case.config = config
    liveness_case.targets = targets
    test_cases.append(liveness_case)

    return test_cases


def print_summary(reports: List[TestReport]):
    """Print test execution summary"""
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    
    total = len(reports)
    passed = sum(1 for report in reports if report.success)
    failed = total - passed
    total_duration = sum(report.duration for report in reports)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration:.2f}s")
    
    print("\nDetailed Results:")
    print("-" * 40)
    for report in reports:
        print(f"  {report}")
        if report.error:
            print(f"    Error: {report.error}")
    
    print("="*60)
    
    return failed == 0


def main():
    """Main entry point"""
    print("Starting server tests...")
    print(f"Service port: {os.getenv('SERVICE_PORT', '8000')}")
    print(f"Test timeout: {os.getenv('TEST_TIMEOUT', '60')}s")
    print(f"Test retries: {os.getenv('TEST_RETRIES', '2')}")
    
    try:
        # Create test cases
        test_cases = create_test_cases()
        print(f"Created {len(test_cases)} test case(s)")
        
        # Initialize ServerRunner
        runner = ServerRunner(test_cases)
        
        # Run tests
        start_time = time.perf_counter()
        reports = runner.run()
        total_duration = time.perf_counter() - start_time
        
        print(f"\nAll tests completed in {total_duration:.2f}s")
        
        # Print summary
        success = print_summary(reports)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

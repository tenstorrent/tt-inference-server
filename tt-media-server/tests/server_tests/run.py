#!/usr/bin/env python3

import os
import sys
import time
import json
import importlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tests.server_tests.tests_runner import ServerRunner
from tests.server_tests.test_classes import TestConfig
from tests.server_tests.test_cases.media_server_liveness_test import MediaServerLivenessTest


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


def load_test_cases_from_json(json_file_path: str) -> List:
    """Load test cases from JSON configuration file"""
    try:
        with open(json_file_path, 'r') as f:
            json_config = json.load(f)
        
        # Create test cases from JSON
        test_cases = []
        for test_case_data in json_config.get("test_cases", []):
            if not test_case_data.get("enabled", True):
                print(f"Skipping disabled test: {test_case_data['name']}")
                continue
            
            try:
                # Create config from test case's test_config
                config_dict = test_case_data.get("test_config", {})
                config = TestConfig(config_dict)
                
                # Create targets from test case's targets
                targets = test_case_data.get("targets", {})
                
                # Import the test class dynamically using name as class name
                module_name = test_case_data["module"]
                class_name = test_case_data["name"]  # Use name as class name
                
                module = importlib.import_module(module_name)
                test_class = getattr(module, class_name)
                
                # Create test instance
                test_instance = test_class(config, targets)
                test_instance.config = config
                test_instance.targets = targets
                
                test_cases.append(test_instance)
                
                print(f"✓ Loaded test: {test_case_data['name']} - {test_case_data.get('description', '')}")
                print(f"  Config: timeout={config.get('test_timeout')}, retries={config.get('retry_attempts')}")
                print(f"  Targets: {[t.name for t in targets]}")
                
            except Exception as e:
                print(f"✗ Failed to load test {test_case_data['name']}: {e}")
                continue
        
        return test_cases
        
    except FileNotFoundError:
        print(f"JSON config file not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return []
    except Exception as e:
        print(f"Error loading test cases from JSON: {e}")
        return []


def print_summary(reports: List[TestReport], test_cases):
    """Print test execution summary"""
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    
    total = len(test_cases)
    passed = sum(1 for report in reports if report.success)
    attempted = len(reports)
    skipped = total - attempted
    failed = total - passed
    total_duration = sum(report.duration for report in reports)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Attempted: {attempted}")
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
        json_file_path = os.getenv("TEST_CONFIG_JSON")
        if json_file_path:
            # Load test cases from JSON config
            test_cases = load_test_cases_from_json(json_file_path)
        
        print(f"Created {len(test_cases)} test case(s)")
        
        # Initialize ServerRunner
        runner = ServerRunner(test_cases)
        
        # Run tests
        start_time = time.perf_counter()
        reports = runner.run()
        total_duration = time.perf_counter() - start_time
        
        print(f"\nAll tests completed in {total_duration:.2f}s")
        
        # Print summary
        success = print_summary(reports, test_cases)
        
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

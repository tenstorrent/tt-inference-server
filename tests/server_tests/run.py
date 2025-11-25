# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/usr/bin/env python3

import importlib
import json
import os
import sys
import time
import logging
from typing import List

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from server_tests.test_classes import TestConfig, TestReport
from server_tests.tests_runner import ServerRunner


def configure_logging():
    """Configure logging to display to console with proper formatting"""
    # Set up root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific loggers to INFO level
    logging.getLogger("tests.server_tests").setLevel(logging.INFO)
    logging.getLogger("whisper_eval_test").setLevel(logging.INFO)


def load_test_cases_from_json(json_file_path: str) -> List:
    """Load test cases from JSON configuration file"""
    try:
        with open(json_file_path, "r") as f:
            json_config = json.load(f)

        # Create test cases from JSON
        test_cases = []
        for test_case_data in json_config.get("test_cases", []):
            if not test_case_data.get("enabled", True):
                logger.info(f"Skipping disabled test: {test_case_data['name']}")
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

                logger.info(
                    f"✓ Loaded test: {test_case_data['name']} - {test_case_data.get('description', '')}"
                )
                logger.info(
                    f"  Config: timeout={config.get('test_timeout')}, retries={config.get('retry_attempts')}"
                )
                logger.info(f"  Targets: {[t.name for t in targets]}")

            except Exception as e:
                logger.error(f"✗ Failed to load test {test_case_data['name']}: {e}")
                continue

        return test_cases

    except FileNotFoundError:
        logger.error(f"JSON config file not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading test cases from JSON: {e}")
        return []


def print_summary(reports: List[TestReport], test_cases):
    """Print test execution summary"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("=" * 60)

    total = len(test_cases)
    passed = sum(1 for report in reports if report.success)
    attempted = len(reports)
    skipped = total - attempted
    failed = total - passed
    total_duration = sum(report.duration for report in reports)

    logger.info(f"Total tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Attempted: {attempted}")
    logger.info(f"Total duration: {total_duration:.2f}s")

    logger.info("\nDetailed Results:")
    logger.info("-" * 40)
    for report in reports:
        logger.info(f"  {report}")
        if report.error:
            logger.error(f"    Error: {report.error}")

    logger.info("=" * 60)
    return failed == 0


def main():
    """Main entry point"""

    # Configure logging first
    configure_logging()

    logger.info("Starting server tests...")
    logger.info(f"Service port: {os.getenv('SERVICE_PORT', '8000')}")
    logger.info(f"Test timeout: {os.getenv('TEST_TIMEOUT', '60')}s")
    logger.info(f"Test retries: {os.getenv('TEST_RETRIES', '2')}")

    try:
        json_file_path = os.getenv("TEST_CONFIG_JSON")
        if json_file_path:
            # Load test cases from JSON config
            test_cases = load_test_cases_from_json(json_file_path)

        logger.info(f"Created {len(test_cases)} test case(s)")

        # Initialize ServerRunner
        runner = ServerRunner(test_cases)

        # Run tests
        start_time = time.perf_counter()
        reports = runner.run()
        total_duration = time.perf_counter() - start_time

        logger.info(f"\nAll tests completed in {total_duration:.2f}s")

        # Print summary
        success = print_summary(reports, test_cases)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\n\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nFatal error during test execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

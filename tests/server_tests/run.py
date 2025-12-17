# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/usr/bin/env python3

import argparse
import importlib
import json
import logging
import os
import sys
import time
from typing import List

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Add the tests directory to the Python path for server_tests imports
tests_dir = os.path.join(project_root, "tests")
sys.path.insert(0, tests_dir)

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

                description = test_case_data.get("description", "")

                # Import the test class dynamically using name as class name
                module_name = test_case_data["module"]
                class_name = test_case_data["name"]  # Use name as class name

                module = importlib.import_module(module_name)
                test_class = getattr(module, class_name)

                # Create test instance
                test_instance = test_class(config, targets)
                test_instance.config = config
                test_instance.targets = targets
                test_instance.description = description

                test_cases.append(test_instance)

                logger.info(
                    f"✅ Loaded test: {test_case_data['name']} - {test_case_data.get('description', '')}"
                )
                logger.info(
                    f"  Config: timeout={config.get('test_timeout')}, retries={config.get('retry_attempts')}"
                )
                logger.info(f"  Targets: {targets}")

            except Exception as e:
                logger.error(f"❌ Failed to load test {test_case_data['name']}: {e}")
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


def find_test_config_by_model_and_device(model: str, device: str) -> dict:
    """Find test configuration in server_tests_config.json by matching model and device"""
    config_path = os.path.join(os.path.dirname(__file__), "server_tests_config.json")

    try:
        with open(config_path, "r") as f:
            configs = json.load(f)

        # Find matching configuration
        for config in configs:
            # Check if model is in weights array and device matches
            if model in config.get("weights", []) and config.get("device") == device:
                logger.info(f"Found matching config for model={model}, device={device}")
                logger.info(f"  Weights: {config.get('weights')}")
                logger.info(f"  Device: {config.get('device')}")
                logger.info(f"  Test cases: {len(config.get('test_cases', []))}")
                return config

        logger.warning(f"No matching config found for model={model}, device={device}")
        return None

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return None


def print_summary(reports: List[TestReport], test_cases):
    """Print test execution summary as a formatted table"""
    total = len(test_cases)
    passed = sum(1 for report in reports if report.success)
    attempted = len(reports)
    skipped = total - attempted
    failed = attempted - passed
    total_duration = sum(report.duration for report in reports)

    # Summary table
    logger.info("=" * 70)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<20} {'Value':>10}")
    logger.info("-" * 32)
    logger.info(f"{'Total tests':<20} {total:>10}")
    logger.info(f"{'Passed':<20} {passed:>10}")
    logger.info(f"{'Failed':<20} {failed:>10}")
    logger.info(f"{'Skipped':<20} {skipped:>10}")
    logger.info(f"{'Attempted':<20} {attempted:>10}")
    logger.info(f"{'Total duration':<20} {total_duration:>9.2f}s")
    logger.info("=" * 70)

    # Detailed results table
    logger.info("Detailed Results:")
    logger.info("-" * 70)
    logger.info(f"{'Status':<8} {'Test Name':<40} {'Duration':>10} {'Attempts':>10}")
    logger.info("-" * 70)
    for report in reports:
        status = "✅ PASS" if report.success else "❌ FAIL"
        logger.info(
            f"{status:<8} {report.test_name:<40} {report.duration:>9.2f}s {report.attempts:>10}"
        )
        if report.error:
            logger.error(f"         Error: {report.error}")
    logger.info("=" * 70)

    return failed == 0


def main():
    """Main entry point"""

    # Configure logging first
    configure_logging()
    args = parse_args()

    logger.info("Starting server tests...")
    logger.info(f"Service port: {os.getenv('SERVICE_PORT', '8000')}")
    logger.info(f"Test timeout: {os.getenv('TEST_TIMEOUT', '60')}s")
    logger.info(f"Test retries: {os.getenv('TEST_RETRIES', '2')}")

    try:
        json_file_path = os.getenv("TEST_CONFIG_JSON")

        if json_file_path:
            # Load test cases from specified JSON config
            logger.info(f"Loading test config from: {json_file_path}")
            with open(json_file_path, "r") as f:
                json_config = json.load(f)
            test_cases_config = json_config
        elif args.model and args.device:
            # Find config by model and device in server_tests_config.json
            logger.info(
                f"Finding test config for model={args.model}, device={args.device}"
            )
            config = find_test_config_by_model_and_device(args.model, args.device)

            if config:
                # Use only the test_cases attribute from the matched config
                test_cases_config = {"test_cases": config.get("test_cases", {})}
            else:
                logger.warning(
                    f"No test configuration found for model={args.model}, device={args.device}"
                )
                logger.warning("Available configurations in server_tests_config.json:")
                try:
                    config_path = os.path.join(
                        os.path.dirname(__file__), "server_tests_config.json"
                    )
                    with open(config_path, "r") as f:
                        configs = json.load(f)
                        for cfg in configs:
                            logger.error(
                                f"  - weights={cfg.get('weights')}, device={cfg.get('device')}"
                            )
                except Exception:
                    logger.warning("  (Failed to load available configurations)")
                    # return success to not fail CI runs that don't have spec tests
                    return 0
                # gracefully exit
                return 0
        else:
            logger.warning("TEST_CONFIG_JSON environment variable not set")
            logger.warning("Please either:")
            logger.warning(
                "  1. Set TEST_CONFIG_JSON to point to your test configuration file"
            )
            logger.warning(
                "  2. Provide --model and --device arguments to auto-select from server_tests_config.json"
            )
            sys.exit(0)

        # Load test cases from the test_cases_config
        test_cases = []
        for test_case_data in test_cases_config.get("test_cases", []):
            if not test_case_data.get("enabled", True):
                logger.info(f"Skipping disabled test: {test_case_data['name']}")
                continue

            try:
                # Create config from test case's test_config
                config_dict = test_case_data.get("test_config", {})
                config = TestConfig(config_dict)

                # Create targets from test case's targets
                targets = test_case_data.get("targets", {})

                description = test_case_data.get("description", "")

                # Import the test class dynamically using name as class name
                module_name = test_case_data["module"]
                class_name = test_case_data["name"]  # Use name as class name

                module = importlib.import_module(module_name)
                test_class = getattr(module, class_name)

                # Create test instance
                test_instance = test_class(config, targets)
                test_instance.config = config
                test_instance.targets = targets
                test_instance.description = description

                test_cases.append(test_instance)

                logger.info(
                    f"✅ Loaded test: {test_case_data['name']} - {test_case_data.get('description', '')}"
                )
                logger.info(
                    f"  Config: timeout={config.get('test_timeout')}, retries={config.get('retry_attempts')}"
                )
                logger.info(f"  Targets: {targets}")

            except Exception as e:
                logger.error(f"❌ Failed to load test {test_case_data['name']}: {e}")
                continue

        if not test_cases:
            logger.error("No test cases loaded")
            sys.exit(1)

        logger.info(f"Created {len(test_cases)} test case(s)")

        # Initialize ServerRunner
        runner = ServerRunner(test_cases)

        # Run tests
        start_time = time.perf_counter()
        reports = runner.run()
        total_duration = time.perf_counter() - start_time

        logger.info(f"All tests completed in {total_duration:.2f}s")

        # Print summary
        success = print_summary(reports, test_cases)

        # Exit with appropriate code
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error during test execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for benchmark output",
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name",
        default=os.getenv("MODEL", ""),
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device name",
        default=os.getenv("DEVICE", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    ret_args = parser.parse_args()
    return ret_args


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/usr/bin/env python3

"""
Server tests runner with marker-based filtering support.

Usage examples:
    # Run all tests for a model/device
    python run.py --model stable-diffusion-xl-base-1.0 --device n150

    # Filter by test markers
    python run.py --model stable-diffusion-xl-base-1.0 --device n150 --markers load

    # Filter by model category on specific hardware
    python run.py --model-category IMAGE --device n150

    # Run all load tests across all models/devices
    python run.py --markers load

    # Exclude slow tests
    python run.py --model stable-diffusion-xl-base-1.0 --device n150 --exclude-markers slow heavy

    # Run specific test
    python run.py --model stable-diffusion-xl-base-1.0 --device n150 --test-name ImageGenerationLoadTest

    # Skip prerequisite tests (DeviceLivenessTest)
    python run.py --model stable-diffusion-xl-base-1.0 --device n150 --skip-prerequisites
"""

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
from server_tests.test_filter import TestFilter
from server_tests.tests_runner import ServerRunner

SERVER_TESTS_CONFIG_PATH = os.path.join(
    project_root, "tests", "server_tests", "server_tests_config.json"
)


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


def load_test_instances_from_suites(test_suites: List[dict]) -> List:
    """
    Load test instances from filtered test suites.

    Args:
        test_suites: List of test suite dictionaries from TestFilter.get_tests()

    Returns:
        List of instantiated test case objects
    """
    test_cases = []

    for suite in test_suites:
        suite_id = suite.get("id", "unknown")
        device = suite.get("device", "unknown")
        weights = suite.get("weights", [])

        logger.info(
            f"Loading tests for suite: {suite_id} (device={device}, models={weights})"
        )

        for test_case_data in suite.get("test_cases", []):
            if not test_case_data.get("enabled", True):
                logger.info(f"  Skipping disabled test: {test_case_data.get('name')}")
                continue

            try:
                config_dict = test_case_data.get("test_config", {})
                config = TestConfig(config_dict)
                targets = test_case_data.get("targets", {})
                description = test_case_data.get("description", "")

                module_name = test_case_data["module"]
                class_name = test_case_data["name"]

                module = importlib.import_module(module_name)
                test_class = getattr(module, class_name)

                test_instance = test_class(config, targets)
                test_instance.config = config
                test_instance.targets = targets
                test_instance.description = description
                test_instance.markers = test_case_data.get("markers", [])
                test_instance.suite_id = suite_id

                test_cases.append(test_instance)

                markers_str = ", ".join(test_case_data.get("markers", []))
                logger.info(f"  ✅ Loaded: {class_name} - {description}")
                logger.info(f"     Markers: [{markers_str}]")
                logger.info(
                    f"     Config: timeout={config.get('test_timeout')}, retries={config.get('retry_attempts')}"
                )
                logger.info(f"     Targets: {targets}")

            except Exception as e:
                logger.error(
                    f"  ❌ Failed to load test {test_case_data.get('name')}: {e}"
                )
                continue

    return test_cases


def print_summary(reports: List[TestReport], test_cases):
    """Print test execution summary as a formatted table"""
    total = len(test_cases)
    passed = sum(1 for report in reports if report.success)
    attempted = len(reports)
    skipped = total - attempted
    failed = attempted - passed
    total_duration = sum(report.duration for report in reports)

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


def apply_filters(args) -> List[dict]:
    """
    Apply CLI filters to get matching test suites.

    Args:
        args: Parsed CLI arguments

    Returns:
        List of filtered test suite dictionaries
    """
    from pathlib import Path

    # Determine how to load TestFilter based on arguments
    suite_files = None
    if hasattr(args, "suite_file") and args.suite_file:
        # Load specific suite file(s)
        suite_files = [Path(f) for f in args.suite_file]
        logger.info(f"Loading suite files: {[str(f) for f in suite_files]}")

    if hasattr(args, "suite_category") and args.suite_category:
        # Load by category name (e.g., "image", "audio")
        try:
            logger.info(f"Loading suite category: {args.suite_category}")
            test_filter = TestFilter.from_category(args.suite_category)
        except FileNotFoundError as e:
            logger.error(str(e))
            return []
    elif hasattr(args, "config") and args.config:
        # Load from custom config file
        config_path = Path(args.config)
        logger.info(f"Loading config from: {config_path}")
        try:
            test_filter = TestFilter(config_path=config_path)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return []
    else:
        # Default: load schema + auto-discover suite files
        test_filter = TestFilter(suite_files=suite_files)

    # Apply model category filter
    if args.model_category:
        logger.info(f"Filtering by model categories: {args.model_category}")
        test_filter.filter_by_model_category(args.model_category)

    # Apply model filter
    if args.model:
        logger.info(f"Filtering by model: {args.model}")
        test_filter.filter_by_model(args.model)

    # Apply device filter
    if args.device:
        logger.info(f"Filtering by device: {args.device}")
        test_filter.filter_by_device(args.device)

    # Apply marker filter
    if args.markers:
        match_all = args.match_all_markers
        logger.info(f"Filtering by markers: {args.markers} (match_all={match_all})")
        test_filter.filter_by_markers(args.markers, match_all=match_all)

    # Apply test name filter
    if args.test_name:
        logger.info(f"Filtering by test name: {args.test_name}")
        test_filter.filter_by_test_name(args.test_name)

    # Apply marker exclusion
    if args.exclude_markers:
        logger.info(f"Excluding markers: {args.exclude_markers}")
        test_filter.exclude_markers(args.exclude_markers)

    # Handle prerequisites
    if args.skip_prerequisites:
        logger.info("Skipping prerequisite tests (DeviceLivenessTest)")
        test_filter.include_prerequisites(False)

    # Get filtered tests
    test_suites = test_filter.get_tests()

    # Print filter summary
    test_filter.print_summary()

    return test_suites


def main():
    """Main entry point"""
    configure_logging()
    args = parse_args()

    logger.info("Starting server tests...")
    logger.info(f"Service port: {os.getenv('SERVICE_PORT', '8000')}")
    logger.info(f"Test timeout: {os.getenv('TEST_TIMEOUT', '60')}s")
    logger.info(f"Test retries: {os.getenv('TEST_RETRIES', '2')}")

    try:
        # Use TestFilter with CLI arguments
        test_suites = apply_filters(args)

        if not test_suites:
            logger.error("No test suites match the specified filters")
            logger.info("Available options:")
            test_filter = TestFilter()
            logger.info(f"  Devices: {test_filter.get_all_devices()}")
            logger.info(f"  Models: {test_filter.get_all_models()}")
            return 1

        # Load test instances from filtered suites
        test_cases = load_test_instances_from_suites(test_suites)

        if not test_cases:
            logger.error("No test cases loaded")
            return 1

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

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error during test execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


def parse_args():
    """Parse command line arguments with marker-based filtering support."""
    parser = argparse.ArgumentParser(
        description="Run server tests with marker-based filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for a model on specific device
  python run.py --model stable-diffusion-xl-base-1.0 --device n150

  # Run all IMAGE model tests on N150
  python run.py --model-category IMAGE --device n150

  # Run load tests only
  python run.py --model stable-diffusion-xl-base-1.0 --device n150 --markers load

  # Run all load tests across all models
  python run.py --markers load

  # Run fast smoke tests
  python run.py --markers smoke fast --match-all-markers

  # Exclude slow and heavy tests
  python run.py --device n150 --exclude-markers slow heavy

  # Run a specific test class
  python run.py --model stable-diffusion-xl-base-1.0 --device n150 --test-name ImageGenerationLoadTest

  # Skip prerequisite tests
  python run.py --model stable-diffusion-xl-base-1.0 --device n150 --skip-prerequisites

  # Load only IMAGE category suites (efficient for CI)
  python run.py --suite-category image --device n150

  # Load specific suite files
  python run.py --suite-file image.json audio.json --markers load
        """,
    )

    # Config file arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom test config JSON file",
        default=None,
    )
    parser.add_argument(
        "--suite-file",
        type=str,
        nargs="+",
        help="Load specific suite file(s) from test_suites/ (e.g., image.json audio.json)",
        default=None,
    )
    parser.add_argument(
        "--suite-category",
        type=str,
        help="Load suites for a specific category (e.g., image, audio, forge)",
        default=None,
    )

    # Workflow integration arguments
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Path to model specification JSON file (passed by workflow)",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for test output",
        required=False,
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token",
        default=os.getenv("HF_TOKEN", ""),
    )

    # Model/Device filtering
    parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name (e.g., stable-diffusion-xl-base-1.0)",
        default=os.getenv("MODEL", ""),
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Filter by device (n150, n300, t3k, galaxy)",
        default=os.getenv("DEVICE", ""),
    )
    parser.add_argument(
        "--model-category",
        type=str,
        nargs="+",
        help="Filter by model category (IMAGE, AUDIO, CNN)",
        default=None,
    )

    # Marker-based filtering
    parser.add_argument(
        "--markers",
        type=str,
        nargs="+",
        help="Filter by test markers (e.g., load smoke fast)",
        default=None,
    )
    parser.add_argument(
        "--match-all-markers",
        action="store_true",
        help="Require ALL markers to match (default: ANY marker matches)",
    )
    parser.add_argument(
        "--exclude-markers",
        type=str,
        nargs="+",
        help="Exclude tests with these markers (e.g., slow heavy)",
        default=None,
    )

    # Test selection
    parser.add_argument(
        "--test-name",
        type=str,
        help="Filter by specific test class name (e.g., ImageGenerationLoadTest)",
        default=None,
    )

    # Prerequisite control
    parser.add_argument(
        "--skip-prerequisites",
        action="store_true",
        help="Skip prerequisite tests (DeviceLivenessTest)",
    )

    # Utility arguments
    parser.add_argument(
        "--list-markers",
        action="store_true",
        help="List all available markers and exit",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List matching tests without running them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print test plan without executing",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_markers:
        test_filter = TestFilter()
        markers = test_filter.get_available_markers()
        logger.info("Available Markers:")
        logger.info("=" * 50)
        for category, items in markers.items():
            logger.info(f"{category}:")
            for marker, desc in items.items():
                logger.info(f"  {marker}: {desc}")
        sys.exit(0)

    if args.list_tests:
        # Create TestFilter respecting suite file/category arguments
        if args.suite_category:
            try:
                test_filter = TestFilter.from_category(args.suite_category)
            except FileNotFoundError as e:
                logger.error(f"Error: {e}")
                sys.exit(1)
        elif args.suite_file:
            from pathlib import Path

            suite_files = [Path(f) for f in args.suite_file]
            test_filter = TestFilter(suite_files=suite_files)
        else:
            test_filter = TestFilter()

        if args.model_category:
            test_filter.filter_by_model_category(args.model_category)
        if args.model:
            test_filter.filter_by_model(args.model)
        if args.device:
            test_filter.filter_by_device(args.device)
        if args.markers:
            test_filter.filter_by_markers(
                args.markers, match_all=args.match_all_markers
            )
        if args.test_name:
            test_filter.filter_by_test_name(args.test_name)
        if args.exclude_markers:
            test_filter.exclude_markers(args.exclude_markers)
        if args.skip_prerequisites:
            test_filter.include_prerequisites(False)

        test_filter.print_summary()
        sys.exit(0)

    return args


if __name__ == "__main__":
    sys.exit(main())

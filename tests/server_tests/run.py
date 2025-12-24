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
import logging
import os
import sys
import time
from typing import List

logger = logging.getLogger("server_tests.run")

# Add the project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Add the tests directory to the Python path for server_tests imports
tests_dir = os.path.join(project_root, "tests")
sys.path.insert(0, tests_dir)


from server_tests.test_categorization_system import TestFilter
from server_tests.test_classes import TestConfig
from server_tests.tests_runner import ServerRunner

SERVER_TESTS_CONFIG_PATH = os.path.join(
    project_root, "tests", "server_tests", "server_tests_config.json"
)


def _configure_logging():
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


def _load_test_instances_from_suites(test_suites: List[dict]) -> List:
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


def _apply_filters(args) -> List[dict]:
    """
    Apply CLI filters to get matching test suites.

    Args:
        Parsed CLI arguments.

    Returns:
        List of filtered test suite dictionaries.
    """
    logger.info(
        f"Determine how to load TestFilter based on arguments. Applying filters: {args}"
    )
    if hasattr(args, "suite_category") and args.suite_category:
        # Load by category name (e.g., "image", "audio")
        try:
            logger.info(f"Loading suite category: {args.suite_category}")
            test_filter = TestFilter.from_category(args.suite_category)
        except FileNotFoundError as e:
            logger.error(str(e))
            return []
    else:
        # Default: load schema + auto-discover suite files
        test_filter = TestFilter()

    if args.model_category:
        logger.info(f"Filtering by model categories: {args.model_category}")
        test_filter.filter_by_model_category(args.model_category)

    if args.model:
        logger.info(f"Filtering by model: {args.model}")
        test_filter.filter_by_model(args.model)

    if args.device:
        logger.info(f"Filtering by device: {args.device}")
        test_filter.filter_by_device(args.device)

    if args.markers:
        match_all = args.match_all_markers
        logger.info(f"Filtering by markers: {args.markers} (match_all={match_all})")
        test_filter.filter_by_markers(args.markers, match_all=match_all)

    if args.test_name:
        logger.info(f"Filtering by test name: {args.test_name}")
        test_filter.filter_by_test_name(args.test_name)

    if args.exclude_markers:
        logger.info(f"Excluding markers: {args.exclude_markers}")
        test_filter.exclude_markers(args.exclude_markers)

    if args.skip_prerequisites:
        logger.info("Skipping prerequisite tests (DeviceLivenessTest)")
        test_filter.include_prerequisites(False)

    # Get filtered tests
    test_suites = test_filter.get_tests()

    # Print filter summary
    test_filter.print_summary()

    return test_suites


def _parse_args():
    """Parse command line arguments with marker-based filtering support."""
    parser = argparse.ArgumentParser(
        description="Run server tests with marker-based filtering",
    )

    # Suite category filter
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
        help="Filter by test markers (e.g., load fast)",
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


def main():
    """Main entry point"""
    _configure_logging()
    args = _parse_args()

    logger.info("Starting server tests...")
    logger.info(f"Service port: {os.getenv('SERVICE_PORT', '8000')}")
    logger.info(f"Test timeout: {os.getenv('TEST_TIMEOUT', '60')}s")
    logger.info(f"Test retries: {os.getenv('TEST_RETRIES', '2')}")

    try:
        # Use TestFilter with CLI arguments
        test_suites = _apply_filters(args)

        if not test_suites:
            logger.error("No test suites match the specified filters")
            logger.info("Available options:")
            test_filter = TestFilter()
            logger.info(f"  Devices: {test_filter.get_all_devices()}")
            logger.info(f"  Models: {test_filter.get_all_models()}")
            return 1

        # Load test instances from filtered suites
        test_cases = _load_test_instances_from_suites(test_suites)

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

        # Determine success from reports (ServerRunner already prints summary)
        success = all(report.success for report in reports)
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error during test execution: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

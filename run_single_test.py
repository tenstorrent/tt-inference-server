#!/usr/bin/env python3
"""
Run a single server test directly without needing config files.

Usage:
    python run_single_test.py TTSLoadTest
    python run_single_test.py TTSLoadTest --targets '{"num_of_devices": 1, "sample_count": 5}'
    python run_single_test.py TTSLoadTest --port 8000
"""

import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

from tests.server_tests.test_classes import TestConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single_test(
    test_class_name: str, targets: dict = None, port: int = None, timeout: int = 3600
):
    """Run a single test class directly."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info(f"Starting test execution: {test_class_name}")
    logger.info("=" * 70)

    # Default targets
    if targets is None:
        targets = {
            "num_of_devices": 1,
            "tts_generation_time": 10,
            "sample_count": 5,
            "cleanup": True,
        }
        logger.info("Using default targets (no custom targets provided)")
    else:
        logger.info(f"Using custom targets: {targets}")

    # Set service port if provided
    if port:
        os.environ["SERVICE_PORT"] = str(port)
        logger.info(f"Service port set to: {port}")
    else:
        logger.info(f"Using default service port: {os.getenv('SERVICE_PORT', '8000')}")

    # Default config
    config_dict = {
        "test_timeout": timeout,
        "retry_attempts": 1,
        "retry_delay": 5,
        "break_on_failure": True,
        "mock_mode": False,
    }
    config = TestConfig(config_dict)
    logger.info(
        f"Test configuration: timeout={timeout}s, retries={config_dict['retry_attempts']}"
    )

    # Import test class
    # Convert class name to module name (e.g., TTSLoadTest -> tts_load_test)
    import re

    def class_to_module_name(class_name):
        # Insert underscore before uppercase letters that follow lowercase or other uppercase
        # But group consecutive uppercase together (TTS -> tts, not t_t_s)
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        # Insert underscore before uppercase that follows lowercase
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Convert to lowercase
        module_name = s2.lower()
        # Ensure it ends with _test
        if not module_name.endswith("_test"):
            module_name += "_test"
        return module_name

    module_name = class_to_module_name(test_class_name)

    # Try different import paths
    module_paths = [
        f"tests.server_tests.test_cases.{module_name}",
        f"server_tests.test_cases.{module_name}",
    ]

    logger.info(f"Searching for test class '{test_class_name}'...")
    logger.info(f"  Trying module paths: {', '.join(module_paths)}")

    test_class = None
    for module_path in module_paths:
        try:
            logger.debug(f"  Attempting to import: {module_path}")
            module = importlib.import_module(module_path)
            test_class = getattr(module, test_class_name)
            logger.info(f"✅ Found test class '{test_class_name}' in {module_path}")
            break
        except ImportError as e:
            logger.debug(f"  Import failed: {e}")
            continue
        except AttributeError as e:
            logger.debug(f"  Class not found in module: {e}")
            continue

    if test_class is None:
        logger.error(f"❌ Could not find test class '{test_class_name}'")
        logger.error(f"   Tried: {', '.join(module_paths)}")
        sys.exit(1)

    # Create test instance
    logger.info(f"Creating test instance for '{test_class_name}'...")
    test_instance = test_class(config, targets)
    test_instance.config = config
    test_instance.targets = targets
    test_instance.description = f"Direct run of {test_class_name}"
    logger.info("✅ Test instance created successfully")

    logger.info("=" * 70)
    logger.info(f"🚀 Starting test execution: {test_class_name}")
    logger.info(f"   Service URL: http://localhost:{os.getenv('SERVICE_PORT', '8000')}")
    logger.info(f"   Targets: {targets}")
    logger.info(f"   Timeout: {timeout}s")
    logger.info("=" * 70)

    # Run test
    test_start_time = time.time()
    try:
        logger.info("Calling test_instance.run_tests()...")
        result = test_instance.run_tests()
        test_duration = time.time() - test_start_time
        success = result.get("success", False)

        logger.info("=" * 70)
        logger.info(
            f"{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )
        logger.info(f"   Duration: {test_duration:.2f}s")
        logger.info(f"   Attempts: {result.get('attempts', 1)}")

        if result.get("result"):
            result_data = result["result"]
            logger.info("   Test Results:")
            if isinstance(result_data, dict):
                for key, value in result_data.items():
                    if key != "success":
                        logger.info(f"     {key}: {value}")
            else:
                logger.info(f"     {result_data}")

        total_duration = time.time() - start_time
        logger.info(f"   Total execution time: {total_duration:.2f}s")
        logger.info("=" * 70)

        return 0 if success else 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Test interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        test_duration = time.time() - test_start_time
        logger.error("=" * 70)
        logger.error(f"❌ Test failed with exception after {test_duration:.2f}s")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {e}")
        logger.error("=" * 70)
        import traceback

        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run a single server test directly")
    parser.add_argument("test_class", help="Test class name (e.g., TTSLoadTest)")
    parser.add_argument("--targets", type=str, help="JSON string with targets dict")
    parser.add_argument(
        "--port", type=int, default=8000, help="Service port (default: 8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Test timeout in seconds (default: 3600)",
    )

    args = parser.parse_args()

    targets = {}
    if args.targets:
        try:
            targets = json.loads(args.targets)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --targets: {e}")
            sys.exit(1)

    sys.exit(run_single_test(args.test_class, targets, args.port, args.timeout))


if __name__ == "__main__":
    main()

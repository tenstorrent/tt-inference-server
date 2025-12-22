# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from abc import ABC, abstractmethod

from tests.server_tests.test_cases.device_liveness_test import DeviceLivenessTest
from tests.server_tests.test_classes import TestConfig

# Import test framework components
from .test_status import BaseTestStatus

# BaseMediaStrategy constants
DEVICE_LIVENESS_TEST_ALIVE = "alive"

logger = logging.getLogger(__name__)


class BaseMediaStrategy(ABC):
    """Interface for media strategies."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.service_port = service_port
        self.base_url = f"http://localhost:{service_port}"
        self.test_payloads_path = "utils/test_payloads"

    @abstractmethod
    def run_eval(self) -> None:
        """Run evaluation workflow for this media type."""
        pass

    @abstractmethod
    def run_benchmark(self, num_calls: int) -> list[BaseTestStatus]:
        """Run benchmark workflow for this media type."""
        pass

    def get_health(self, attempt_number: int = 1, override_num_devices: int = None) -> tuple[bool, str]:
        """
        Check health status using DeviceLivenessTest.
        DeviceLivenessTest extends BaseTest, which provides run_tests() with retry logic.

        Returns:
            tuple[bool, str]: (health_status, runner_in_use)
        """
        logger.info("Checking server health using DeviceLivenessTest...")
        device_name = (
            self.device.name if hasattr(self.device, "name") else str(self.device)
        )
        num_devices = self.model_spec.device_model_spec.max_concurrency
        
        if override_num_devices is not None:
            num_devices = override_num_devices

        logger.info(
            f"Detected device: {device_name} with {num_devices} expected worker(s)"
        )

        # Configure test with retry logic
        test_config = TestConfig(
            {
                "test_timeout": 1200,  # 20 minutes
                "retry_attempts": 229,  # 230 total attempts (0-indexed)
                "retry_delay": 10,  # 10 seconds between attempts
                "break_on_failure": False,
            }
        )
        logger.info(f"TestConfig: {test_config}")

        # Set targets for device count validation
        targets = {
            "num_of_devices": num_devices if num_devices and num_devices > 0 else 1
        }
        logger.info(f"Test targets: {targets}")

        # Instantiate test
        liveness_test = DeviceLivenessTest(test_config, targets)
        liveness_test.service_port = self.service_port

        try:
            logger.info("Running DeviceLivenessTest...")
            test_result = liveness_test.run_tests()

            if isinstance(test_result, dict) and test_result.get("success"):
                result_data = test_result.get("result", {})
                runner_in_use = result_data.get("full_response", {}).get(
                    "runner_in_use", None
                )

                logger.info(
                    f"✅ Health check passed after {test_result.get('attempts', 1)} attempt(s)"
                )
                return (True, runner_in_use)
            else:
                logger.error("Health check failed after all retry attempts")
                return (False, None)

        except SystemExit as e:
            logger.error(f"Health check failed with SystemExit: {e}")
            return (False, None)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return (False, None)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

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

    def get_health(self, attempt_number: int = 1) -> tuple[bool, str]:
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
            "num_of_devices": num_devices if num_devices and num_devices > 0 else None
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

    def generate_report(
        self,
        status_list: list[BaseTestStatus],
        report_prefix: str = "benchmark",
    ) -> Optional[Path]:
        """
        Template Method - generates report with common structure.

        Subclasses override _create_specific_benchmarks_data() to add their metrics.

        Args:
            status_list: List of test status objects from benchmark run
            report_prefix: Prefix for the report filename (e.g., "benchmark", "eval")

        Returns:
            Path to generated report file, or None if generation failed
        """
        if not status_list:
            logger.warning("Empty status list, skipping report generation")
            return None

        logger.info("Generating benchmark report...")

        filename = self._create_report_filename(report_prefix)

        self._ensure_report_directory(filename)

        base_data = self._create_base_report_data()

        benchmarks_data = self._create_specific_benchmarks_data(status_list)

        report_data = {**base_data, "benchmarks": benchmarks_data}
        self._save_report(report_data, filename)

        logger.info(f"Report generated: {filename}")
        return filename

    def _create_report_filename(self, prefix: str = "benchmark") -> Path:
        """Create standardized report filename."""
        return (
            Path(self.output_path)
            / f"{prefix}_{self.model_spec.model_id}_{time.time()}.json"
        )

    def _ensure_report_directory(self, filepath: Path) -> None:
        """Ensure parent directory exists."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

    def _create_base_report_data(self) -> dict:
        """Create common report metadata."""
        return {
            "model": self.model_spec.model_name,
            "device": self.device.name
            if hasattr(self.device, "name")
            else str(self.device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": self._get_task_type(),
        }

    def _save_report(self, report_data: dict, filename: Path) -> None:
        """Save report to JSON file."""
        with open(filename, "w") as f:
            json.dump(report_data, f, indent=4)

    def _get_task_type(self) -> str:
        """
        Return task type for this client.
        Subclasses should override to return appropriate type.
        """
        return "unknown"

    @abstractmethod
    def _create_specific_benchmarks_data(
        self, status_list: list[BaseTestStatus]
    ) -> dict:
        """
        Create benchmark-specific data. Subclasses MUST implement.

        Args:
            status_list: List of test status objects

        Returns:
            Dict with benchmark metrics specific to this media type

        Example for audio:
            return {
                "num_requests": len(status_list),
                "ttft": self._calculate_ttft(status_list),
                "rtr": self._calculate_rtr(status_list),
                "t/s/u": self._calculate_tsu(status_list),
            }
        """
        pass

    def _calculate_ttft_value(self, status_list: list[BaseTestStatus]) -> float:
        """
        Can be overridden if needed.
        """
        if not status_list:
            return 0.0
        return sum(
            getattr(s, "elapsed", 0) or getattr(s, "ttft", 0) for s in status_list
        ) / len(status_list)

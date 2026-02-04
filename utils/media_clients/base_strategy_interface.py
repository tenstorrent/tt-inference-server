# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from tests.server_tests.test_cases.device_liveness_test import DeviceLivenessTest
from tests.server_tests.test_classes import TestConfig

from .test_status import BaseTestStatus
from .utils.metrics_utils import MetricsAggregator
from .utils.report_utils import ReportGenerator

# BaseMediaStrategy constants
DEVICE_LIVENESS_TEST_ALIVE = "alive"

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Task type constants for report metadata. Value is used in JSON (e.g. text_to_speech)."""

    TTS = "text_to_speech"
    # AUDIO = "audio", CNN = "cnn", etc. when those clients are migrated


# Removable after all clients use TaskType.
def _task_type_value(task_type: Union["TaskType", str]) -> str:
    """Return string value for report (enum.value or str as-is). Remove after all clients use TaskType."""
    return task_type.value if isinstance(task_type, TaskType) else task_type


class BaseMediaStrategy(ABC):
    """
    Abstract base class for media client strategies.

    Subclasses MUST define task_type class attribute (TaskType enum or str, e.g. task_type = TaskType.TTS).
    Clients that generate reports call ReportGenerator.generate_benchmark_report_for_strategy(self, ...)
    or generate_eval_report_for_strategy(self, ...) and implement _get_benchmark_report_extras /
    _create_eval_extra_data as needed.
    """

    # Subclasses override: task_type = TaskType.TTS or TASK_TYPE = "audio" (legacy)
    TASK_TYPE: str = "unknown"

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        resolved = _task_type_value(getattr(cls, "task_type", cls.TASK_TYPE))
        if resolved == "unknown":
            import warnings

            warnings.warn(
                f"{cls.__name__} should define task_type (e.g. task_type = TaskType.TTS). "
                "Using 'unknown' as default.",
                UserWarning,
                stacklevel=2,
            )

    def __init__(
        self,
        all_params,
        model_spec,
        device,
        output_path,
        service_port,
        *,
        report_generator: Optional[ReportGenerator] = None,
        aggregator: Optional[MetricsAggregator] = None,
    ):
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.service_port = service_port
        self.base_url = f"http://localhost:{service_port}"
        self.test_payloads_path = "utils/test_payloads"
        self._report_generator = report_generator or ReportGenerator()
        self._aggregator = aggregator or MetricsAggregator()
        self._task_type_str: str = _task_type_value(
            getattr(self.__class__, "task_type", self.TASK_TYPE)
        )

    def _get_aggregator(self) -> MetricsAggregator:
        """Return the shared aggregator, reset for this benchmark run."""
        self._aggregator.reset()
        return self._aggregator

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

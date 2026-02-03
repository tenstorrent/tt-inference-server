# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from tests.server_tests.test_cases.device_liveness_test import DeviceLivenessTest
from tests.server_tests.test_classes import TestConfig

from .report_utils import (
    ReportContext,
    generate_benchmark_report,
    generate_eval_report,
)
from .test_status import BaseTestStatus

# BaseMediaStrategy constants
DEVICE_LIVENESS_TEST_ALIVE = "alive"

logger = logging.getLogger(__name__)


class BaseMediaStrategy(ABC):
    """
    Abstract base class for media client strategies.

    Subclasses MUST:
    1. Define TASK_TYPE class attribute (e.g., TASK_TYPE = "tts")
    2. Override _get_benchmark_report_extras() if the client uses _generate_report()
    """

    # Subclasses should override this with their task type
    # Default is "unknown" for backward compatibility during migration
    TASK_TYPE: str = "unknown"

    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            return
        if cls.TASK_TYPE == "unknown":
            import warnings

            warnings.warn(
                f"{cls.__name__} should define TASK_TYPE class attribute "
                f"(e.g., TASK_TYPE = 'audio'). Using 'unknown' as default.",
                UserWarning,
                stacklevel=2,
            )

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

    def _get_report_context(self) -> ReportContext:
        """Build ReportContext from strategy (model_spec, device, output_path)."""
        device_name = (
            self.device.name if hasattr(self.device, "name") else str(self.device)
        )
        return ReportContext(
            model_name=self.model_spec.model_name,
            device_name=device_name,
            output_path=Path(self.output_path),
            model_id=self.model_spec.model_id,
            hf_model_repo=getattr(self.model_spec, "hf_model_repo", None),
        )

    def _generate_report(
        self,
        status_list: list[BaseTestStatus],
        pre_aggregated: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Generate benchmark report: build context, get extras from strategy, call report_utils.

        If pre_aggregated is provided (e.g. from MetricsAggregator.result() during the
        benchmark loop), aggregation is not run again over status_list (one less pass).

        Returns:
            Path to written report file, or None if status_list is empty and no pre_aggregated.
        """
        context = self._get_report_context()
        benchmark_extras, top_level_extras = self._get_benchmark_report_extras(
            status_list
        )
        return generate_benchmark_report(
            context,
            status_list,
            self.TASK_TYPE,
            extra_benchmarks=benchmark_extras,
            extra_top_level=top_level_extras,
            pre_aggregated=pre_aggregated,
        )

    def _get_benchmark_report_extras(
        self, status_list: list[BaseTestStatus]
    ) -> tuple[dict, dict]:
        """
        Return (benchmark_extras, top_level_extras) for the benchmark report.

        - benchmark_extras: merged into report["benchmarks"] (e.g. ttft_p90, accuracy_check).
        - top_level_extras: merged at report root (e.g. streaming_enabled, preprocessing_enabled).

        Override in clients that use _generate_report() (e.g. TTS).
        Default returns ({}, {}).
        """
        return ({}, {})

    def _generate_eval_report(
        self,
        status_list: Optional[list[BaseTestStatus]] = None,
        eval_result: Optional[dict] = None,
        total_time: Optional[float] = None,
        extra_data_override: Optional[dict] = None,
    ) -> Path:
        """
        Generate eval report via report_utils.

        If extra_data_override is set, use it; else call _create_eval_extra_data().
        Returns path to written eval JSON file.
        """
        context = self._get_report_context()
        if extra_data_override is not None:
            extra_data = extra_data_override
        else:
            extra_data = self._create_eval_extra_data(
                status_list=status_list,
                eval_result=eval_result,
                total_time=total_time,
            )
        return generate_eval_report(context, self.TASK_TYPE, extra_data)

    def _create_eval_extra_data(
        self,
        status_list: Optional[list[BaseTestStatus]] = None,
        eval_result: Optional[dict] = None,
        total_time: Optional[float] = None,
    ) -> dict:
        """
        Return client-specific eval payload (task_name, tolerance, score, etc.).

        Override in clients. Merged with base metadata (model, device, timestamp, task_type)
        by report_utils.generate_eval_report.

        Returns:
            Dict of eval fields; default {}.
        """
        return {}


# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from server_tests.test_cases.device_liveness_test import DeviceLivenessTest
from server_tests.test_classes import TestConfig

import workflows.model_spec  # noqa: F401
from workflows.utils_report import PerformanceTargets, get_performance_targets
from workflows.workflow_types import ReportCheckTypes


from .test_status import BaseTestStatus

# BaseMediaStrategy constants
DEVICE_LIVENESS_TEST_ALIVE = "alive"
DEFAULT_PERF_CHECK_TOLERANCE = 0.05

MIN_TAIL_LATENCY_SAMPLES = 10
TAIL_LATENCY_PERCENTILES = (50, 90, 95)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerfCheck:
    """One measured metric to compare against a performance target."""

    name: str
    measured: Optional[float]
    target: Optional[float]
    lower_is_better: bool


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
    def run_benchmark(self) -> list[BaseTestStatus]:
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

    def require_health(self) -> str:
        """Run the health check and abort the workflow if the server is not ready.

        Returns the runner-in-use string so callers can dispatch on it.
        Raises ``RuntimeError``.
        """
        health_status, runner_in_use = self.get_health()
        if not health_status:
            logger.error("Health check failed.")
            raise RuntimeError("Health check failed; aborting workflow.")
        logger.info(f"Health check passed. Runner in use: {runner_in_use}")
        return runner_in_use

    def get_performance_targets(self) -> PerformanceTargets:
        """Look up the configured perf targets for this strategy's model+device."""
        device_str = self.model_spec.cli_args.get("device")
        return get_performance_targets(
            self.model_spec.model_name,
            device_str,
            model_type=self.model_spec.model_type.name,
        )

    @staticmethod
    def _evaluate_perf_check(check: PerfCheck, tolerance: float) -> bool:
        """Evaluate a single perf check, log the outcome, and return pass/fail."""
        if check.lower_is_better:
            threshold = check.target * (1 + tolerance)
            cmp, ok = "<=", check.measured <= threshold
        else:
            threshold = check.target * (1 - tolerance)
            cmp, ok = ">=", check.measured >= threshold

        if ok:
            logger.info(
                f"✅ {check.name} PASSED: {check.measured:.4f} {cmp} {threshold:.4f}"
            )
        else:
            logger.warning(
                f"❌ {check.name} FAILED: {check.measured:.4f} not {cmp} {threshold:.4f}"
            )
        return ok

    def calculate_performance_check(
        self,
        checks: List[PerfCheck],
        tolerance: Optional[float] = None,
    ) -> ReportCheckTypes:
        """Compare measured perf metrics to configured targets with tolerance.

        - Skips checks where either ``measured`` or ``target`` is ``None``.
        - Returns NA when no check is applicable, PASS when every applicable
          check is within tolerance, FAIL otherwise.
        """
        tol = tolerance if tolerance is not None else DEFAULT_PERF_CHECK_TOLERANCE
        logger.info(f"Calculating performance check (tolerance={tol * 100:.2f}%)")

        applicable = [
            c for c in checks if c.measured is not None and c.target is not None
        ]
        if not applicable:
            logger.warning(
                "⚠️ No performance check applicable (no measured metric had a configured target)"
            )
            return ReportCheckTypes.NA

        passed = sum(self._evaluate_perf_check(c, tol) for c in applicable)
        total = len(applicable)

        if passed == total:
            logger.info(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
            return ReportCheckTypes.PASS

        logger.warning(f"⛔️ {total - passed}/{total} performance checks failed")
        return ReportCheckTypes.FAIL

    @staticmethod
    def _calculate_tail_latencies(
        values: Sequence[Optional[float]],
        min_samples: int = MIN_TAIL_LATENCY_SAMPLES,
    ) -> Dict[str, Optional[float]]:
        """Mean p50/p90/p95 over ``values``"""
        result: Dict[str, Optional[float]] = {
            f"latency_p{p}": None for p in TAIL_LATENCY_PERCENTILES
        }

        valid = [v for v in values if v is not None]
        if len(valid) < min_samples:
            logger.info(
                f"Tail latency: {len(valid)} sample(s) < min_samples={min_samples}; "
                f"emitting None for {list(result)}"
            )
            return result

        sorted_values = sorted(valid)
        n = len(sorted_values)
        for percentile in TAIL_LATENCY_PERCENTILES:
            index = min(math.ceil(n * percentile / 100.0) - 1, n - 1)
            result[f"latency_p{percentile}"] = sorted_values[index]
        logger.info(f"Tail latency over {n} samples: {result}")
        return result

    @staticmethod
    def _calculate_throughput_rps(
        num_requests: int,
        wall_clock_seconds: Optional[float],
    ) -> Optional[float]:
        """Requests-per-second over the wall-clock duration of the loop.

        Returns ``None`` when either input is missing/non-positive so the
        producer can serialise that as JSON ``null`` instead of silently
        emitting ``inf`` or ``0`` (both of which mislead downstream).
        """
        if not num_requests or not wall_clock_seconds or wall_clock_seconds <= 0:
            return None
        return num_requests / wall_clock_seconds

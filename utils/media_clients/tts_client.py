# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import asyncio
import logging
import time
from typing import Optional
from .utils.metrics_utils import MetricsAggregator


import aiohttp
from transformers import AutoTokenizer


from .base_strategy_interface import BaseMediaStrategy, TaskType
from .utils.metrics_utils import (
    aggregate_metrics_from_status_list,
    percentiles_from_metric,
)
from .utils.report_utils import ReportContext
from utils.constants import PerformanceResult
from .test_status import TtsTestStatus
from workflows.utils import get_num_calls
from workflows.utils_report import get_performance_targets

logger = logging.getLogger(__name__)


class TtsClientStrategy(BaseMediaStrategy):
    """Strategy for text-to-speech models (SpeechT5, etc.)."""

    task_type = TaskType.TTS
    DEFAULT_BENCHMARK_CALLS = 10
    DEFAULT_EVAL_CALLS = 5
    DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load tokenizer with graceful fallback."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_spec.hf_model_repo)
            logger.info(f"✅ Loaded tokenizer for {self.model_spec.hf_model_repo}")
            return tokenizer
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load tokenizer for {self.model_spec.hf_model_repo}: {e}. "
                "Using word-based counting."
            )
            return None

    def _verify_health(self) -> None:
        """Verify service health, raise on failure."""
        health_ok, runner = self.get_health()
        if not health_ok:
            raise RuntimeError("Health check failed")
        logger.info(f"Health check passed. Runner: {runner}")

    def _get_num_calls(self, is_eval: bool) -> int:
        """Get number of calls with TTS-specific defaults."""
        configured = get_num_calls(self)
        if configured != 2:
            logger.info(f"Using configured num_calls: {configured}")
            return configured
        default = self.DEFAULT_EVAL_CALLS if is_eval else self.DEFAULT_BENCHMARK_CALLS
        logger.info(f"Using TTS default: {default} calls")
        return default

    def _get_test_text(self) -> str:
        """Extract test text from params or use default."""
        if isinstance(self.all_params, (list, tuple)):
            return self.DEFAULT_TTS_TEXT
        tasks = getattr(self.all_params, "tasks", [])
        if not tasks:
            return self.DEFAULT_TTS_TEXT
        task = tasks[0]
        return getattr(task, "text", None) or getattr(
            task, "task_name", self.DEFAULT_TTS_TEXT
        )

    def run_eval(self) -> None:
        """Run TTS evaluation."""
        logger.info(
            f"Running eval for {self.model_spec.model_name} on {self.device.name}"
        )
        self._verify_health()
        num_calls = self._get_num_calls(is_eval=True)
        status_list = self._run_tts_benchmark(num_calls)
        context = ReportContext.from_strategy(self)
        extra_data = self._create_eval_extra_data(status_list=status_list)
        self._report_generator.generate_eval_report(
            context, self._task_type_str, extra_data
        )

    def _create_eval_extra_data(
        self,
        status_list=None,
        eval_result=None,
        total_time=None,
    ) -> dict:
        """TTS eval payload: task_name, tolerance, score, rtr, p90/p95, results, configs."""
        if not status_list:
            return {}
        aggregated = aggregate_metrics_from_status_list(status_list)
        ttft_value = aggregated.get("ttft_ms", 0.0)
        rtr_value = aggregated.get("rtr", 0.0)
        p90_ttft, p95_ttft = percentiles_from_metric(
            status_list, "ttft_ms", (0.90, 0.95)
        )
        logger.info(f"Extracted TTFT value: {ttft_value:.2f}ms")
        logger.info(f"Extracted RTR value: {rtr_value:.2f}")
        logger.info(f"Extracted P90 TTFT: {p90_ttft:.2f}ms, P95 TTFT: {p95_ttft:.2f}ms")

        task_name = self.all_params.tasks[0].task_name
        performance_check = self._calculate_performance_check(ttft_value, rtr_value)
        accuracy_check = self._calculate_accuracy_check()
        return {
            "task_name": task_name,
            "tolerance": self.all_params.tasks[0].score.tolerance,
            "published_score": self.all_params.tasks[0].score.published_score,
            "score": ttft_value,
            "published_score_ref": self.all_params.tasks[0].score.published_score_ref,
            "rtr": rtr_value,
            "p90_ttft": p90_ttft,
            "p95_ttft": p95_ttft,
            "performance_check": performance_check,
            "accuracy_check": accuracy_check,
            "results": {
                task_name: {
                    "score": ttft_value,
                    "rtr": rtr_value,
                    "p90_ttft": p90_ttft,
                    "p95_ttft": p95_ttft,
                }
            },
            "configs": {
                task_name: {
                    "task": task_name,
                    "dataset_path": "N/A",
                }
            },
        }

    def run_benchmark(self, attempt: int = 0) -> list[TtsTestStatus]:
        """Run TTS benchmark with streaming aggregation."""
        logger.info(
            f"Running benchmark for {self.model_spec.model_name} on {self.device.name}"
        )
        self._verify_health()
        num_calls = self._get_num_calls(is_eval=False)
        aggregator = self._create_aggregator()
        status_list = self._run_tts_benchmark(num_calls, aggregator=aggregator)

        context = ReportContext.from_strategy(self)
        aggregated = aggregator.result(len(status_list))
        ttft_ms = aggregated.get("ttft_ms", 0.0)
        p90, p95 = percentiles_from_metric(status_list, "ttft_ms", (0.90, 0.95))
        extras = {
            "ttft": ttft_ms / 1000,
            "ttft_p90": p90 / 1000,
            "ttft_p95": p95 / 1000,
        }
        self._report_generator.generate_benchmark_report(
            context,
            status_list,
            self._task_type_str,
            extra_benchmarks=extras,
            pre_aggregated=aggregated,
        )
        return status_list

    def _run_tts_benchmark(
        self,
        num_calls: int,
        aggregator: Optional["MetricsAggregator"] = None,
    ) -> list[TtsTestStatus]:
        """
        Run TTS benchmark loop.

        If aggregator is provided, metrics are aggregated during the loop.
        """
        logger.info(f"Running {num_calls} TTS calls.")
        test_text = self._get_test_text()
        status_list: list[TtsTestStatus] = []

        for i in range(num_calls):
            logger.info(f"Generating speech {i + 1}/{num_calls}...")
            status = self._generate_speech_status(test_text)
            status_list.append(status)
            if aggregator is not None:
                aggregator.add(status.get_metrics())

        return status_list

    def _generate_speech_status(self, text: str) -> TtsTestStatus:
        """Generate speech and return status object."""
        success, elapsed, ttft_ms, rtr, audio_duration = asyncio.run(
            self._generate_speech(text)
        )
        return TtsTestStatus(
            status=success,
            elapsed=elapsed,
            ttft_ms=ttft_ms,
            rtr=rtr,
            text=text,
            audio_duration=audio_duration,
            reference_text=text,
        )

    async def _generate_speech(
        self, text: str
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """
        Generate speech from text via API.

        Returns:
            (success, latency_sec, ttft_ms, rtr, audio_duration)
        """
        logger.info("🔊 Calling TTS /audio/speech endpoint")

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {"text": text, "response_format": "json"}

        url = f"{self.base_url}/audio/speech"
        start_time = time.monotonic()
        ttft_ms = None
        audio_duration = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"TTS request failed with status {response.status}: {error_text}"
                        )
                        return False, 0.0, None, None, None

                    # Check content-type to ensure we're receiving JSON, not WAV bytes
                    content_type = response.headers.get("Content-Type", "").lower()
                    logger.debug(f"Response Content-Type: {content_type}")
                    if "audio" in content_type or "wav" in content_type:
                        logger.error(
                            f"Received audio/wav response instead of JSON. "
                            f"Make sure response_format='json' is set in request. "
                            f"Content-Type: {content_type}. "
                            f"Request payload was: {payload}"
                        )
                        return False, 0.0, None, None, None

                    response_start = time.monotonic()
                    response_data = await response.json()

                    ttft_ms = (response_start - start_time) * 1000

                    audio_duration = response_data.get("duration")
                    if audio_duration is None:
                        logger.warning("Duration not found in response data")
                    else:
                        logger.info(f"Audio duration: {audio_duration}s")

                    # Verify audio data is present
                    audio_base64 = response_data.get("audio")
                    if not audio_base64:
                        logger.error("Audio data not found in response")
                        return False, 0.0, None, None, None

                    logger.info(
                        f"Received audio data (base64 length: {len(audio_base64)})"
                    )

            end_time = time.monotonic()
            total_duration = end_time - start_time

            rtr = None
            if audio_duration is not None and total_duration > 0:
                rtr = audio_duration / total_duration
                logger.info(
                    f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, processing_time={total_duration:.2f}s)"
                )
            else:
                logger.warning(
                    "Could not calculate RTR: missing duration or invalid processing time"
                )

            rtr_str = f"{rtr:.2f}" if rtr is not None else "N/A"
            logger.info(
                f"✅ Done in {total_duration:.2f}s | TTFT={ttft_ms:.2f}ms | RTR={rtr_str}"
            )

            return True, total_duration, ttft_ms, rtr, audio_duration

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return False, 0.0, None, None, None

    def _calculate_performance_check(
        self,
        ttft_value: Optional[float] = None,
        rtr_value: Optional[float] = None,
    ) -> PerformanceResult:
        """Calculate performance check based on TTFT and RTR targets.

        Uses get_performance_targets from model_performance_reference.json.

        Args:
            ttft_value: Time to first token in milliseconds
            rtr_value: Real-time ratio (audio_duration / generation_time)

        Returns:
            0 - undefined (no targets or values)
            2 - passed (all metrics within tolerance)
            3 - failed (any metric outside tolerance)
        """
        logger.info("Calculating performance check based on TTFT, RTR targets")

        # Get performance targets using the shared utility
        device_str = self.model_spec.cli_args.get("device")
        targets = get_performance_targets(
            self.model_spec.model_name,
            device_str,
            model_type=self.model_spec.model_type.name,
        )
        logger.info(f"Performance targets: {targets}")

        # TTFT is the primary metric for TTS performance - required for validation
        if not targets.ttft_ms:
            logger.warning("⚠️ No TTFT target found, skipping performance check")
            return PerformanceResult.UNDEFINED

        tolerance = targets.tolerance if targets.tolerance else 0.05
        logger.info(f"Using tolerance: {tolerance * 100:.2f}%")

        checks_passed = 0
        checks_total = 0

        if targets.ttft_ms is not None:
            checks_total += 1
            ttft_threshold = targets.ttft_ms * (1 + tolerance)
            if ttft_value <= ttft_threshold:
                logger.info(
                    f"✅ TTFT PASSED: {ttft_value:.2f}ms <= {ttft_threshold:.2f}ms"
                )
                checks_passed += 1
            else:
                logger.warning(
                    f"❌ TTFT FAILED: {ttft_value:.2f}ms > {ttft_threshold:.2f}ms"
                )

        if targets.rtr is not None:
            checks_total += 1
            rtr_threshold = targets.rtr * (1 - tolerance)
            if rtr_value >= rtr_threshold:
                logger.info(f"✅ RTR PASSED: {rtr_value:.2f} >= {rtr_threshold:.2f}")
                checks_passed += 1
            else:
                logger.warning(f"❌ RTR FAILED: {rtr_value:.2f} < {rtr_threshold:.2f}")

        if checks_total == 0:
            logger.warning("⚠️ No metrics available for validation")
            return PerformanceResult.UNDEFINED

        if checks_passed == checks_total:
            logger.info(f"✅ All {checks_total} performance checks passed")
            return PerformanceResult.PASS
        logger.warning(
            f"❌ {checks_total - checks_passed}/{checks_total} performance checks failed"
        )
        return PerformanceResult.FAIL

    def _calculate_accuracy_check(self) -> int:
        """Calculate accuracy/quality check for TTS eval/benchmark.

        Returns:
            PerformanceResult.UNDEFINED, PASS, or FAIL (0, 2, 3).
        """
        return PerformanceResult.UNDEFINED

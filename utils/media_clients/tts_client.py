# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional


import aiohttp

from .base_strategy_interface import BaseMediaStrategy
from .test_status import TtsTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import get_num_calls
from workflows.utils_report import get_performance_targets
from workflows.workflow_types import ReportCheckTypes

logger = logging.getLogger(__name__)

# Default text for TTS testing (going to be changed with dataset)
DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."


class TtsClientStrategy(BaseMediaStrategy):
    """Strategy for text-to-speech models (SpeechT5, etc.)."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)

        self.tokenizer = None
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_model_repo)
            logger.info(f"✅ Loaded tokenizer for {model_spec.hf_model_repo}")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load tokenizer for {model_spec.hf_model_repo}: {e}"
            )
            logger.info("📝 Falling back to word-based token counting")

    def run_eval(self) -> None:
        """Run evaluations for the TTS model."""
        status_list = []

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            num_calls = self._get_tts_num_calls(is_eval=True)

            status_list = self._run_tts_benchmark(num_calls)
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        benchmark_data = {}

        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value:.2f}ms")

        # Calculate RTR
        rtr_value = self._calculate_rtr_value(status_list)
        logger.info(f"Extracted RTR value: {rtr_value:.2f}")

        # Calculate tail latency (P90, P95)
        p90_ttft, p95_ttft = self._calculate_tail_latency(status_list)
        logger.info(f"Extracted P90 TTFT: {p90_ttft:.2f}ms, P95 TTFT: {p95_ttft:.2f}ms")

        # Metadata fields (excluded from numeric metrics in process_list_format_eval_files)
        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name
        benchmark_data["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        benchmark_data["task_type"] = "text_to_speech"
        # all_params is always an object with tasks attribute
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance
        benchmark_data["published_score"] = self.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[
            0
        ].score.published_score_ref
        benchmark_data["rtr"] = rtr_value
        benchmark_data["p90_ttft"] = p90_ttft
        benchmark_data["p95_ttft"] = p95_ttft

        performance_check = self._calculate_performance_check(
            ttft_value=ttft_value, rtr_value=rtr_value
        )
        benchmark_data["performance_check"] = performance_check

        benchmark_data["accuracy_check"] = self._calculate_accuracy_check()

        benchmark_data = [benchmark_data]
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )

        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmark(self, attempt=0) -> list[TtsTestStatus]:
        """Run benchmarks for the TTS model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info(f"Health check passed. Runner in use: {runner_in_use}")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            num_calls = self._get_tts_num_calls(is_eval=False)

            status_list = self._run_tts_benchmark(num_calls)

            self._generate_report(status_list)
            return status_list
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _get_tts_num_calls(self, is_eval: bool = False) -> int:
        """Get number of calls for TTS with TTS-specific defaults.

        TTS requires more samples for statistical validity:
        - BENCHMARK: 10 calls (for reliable P90/P95 tail latency)
        - EVAL: 5 calls (for consistent evaluation results)

        Args:
            is_eval: True for EVAL workflow, False for BENCHMARK workflow

        Returns:
            Number of calls to make, respecting num_eval_runs if specified
        """
        base_num_calls = get_num_calls(self)

        # If base_num_calls is not the default (2), respect the configured value
        if base_num_calls != 2:
            logger.info(f"Using configured num_eval_runs: {base_num_calls} calls")
            return base_num_calls

        # Override default (2) with TTS-specific defaults
        tts_default = 5 if is_eval else 10
        workflow_type = "eval" if is_eval else "benchmark"
        logger.info(
            f"Using TTS-specific {workflow_type} default: {tts_default} calls (was {base_num_calls})"
        )
        return tts_default

    def _generate_report(self, status_list: list[TtsTestStatus]) -> None:
        """
        Generate benchmark report for TTS model.
        """
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )

        result_filename.parent.mkdir(parents=True, exist_ok=True)

        ttft_value = self._calculate_ttft_value(status_list)

        # Calculate RTR
        rtr_value = self._calculate_rtr_value(status_list)

        # Calculate tail latency (P90, P95)
        p90_ttft, p95_ttft = self._calculate_tail_latency(status_list)

        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "ttft": ttft_value / 1000,
                "rtr": rtr_value,
                "ttft_p90": p90_ttft / 1000,  # ms to seconds; 0.0 when no data
                "ttft_p95": p95_ttft / 1000,  # ms to seconds; 0.0 when no data
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "tts",
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)

        logger.info(f"Report generated: {result_filename}")

    def _run_tts_benchmark(self, num_calls: int) -> list[TtsTestStatus]:
        """Run TTS benchmark."""
        logger.info(f"Running TTS benchmark with {num_calls} calls.")
        status_list = []

        test_text = DEFAULT_TTS_TEXT
        if (
            not isinstance(self.all_params, (list, tuple))
            and hasattr(self.all_params, "tasks")
            and len(self.all_params.tasks) > 0
        ):
            if hasattr(self.all_params.tasks[0], "text"):
                test_text = self.all_params.tasks[0].text
            elif hasattr(self.all_params.tasks[0], "task_name"):
                test_text = self.all_params.tasks[0].task_name

        for i in range(num_calls):
            logger.info(f"Generating speech {i + 1}/{num_calls}...")

            status, elapsed, ttft_ms, rtr, audio_duration = asyncio.run(
                self._generate_speech()
            )
            logger.debug(f"Generated speech in {elapsed:.2f} seconds.")

            status_list.append(
                TtsTestStatus(
                    status=status,
                    elapsed=elapsed,
                    ttft_ms=ttft_ms,
                    rtr=rtr,
                    text=test_text,
                    audio_duration=audio_duration,
                )
            )

        return status_list

    async def _generate_speech(
        self,
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """Generate speech from text.

        Returns:
            tuple: (success, latency_sec, ttft_ms, rtr, audio_duration)
                - success: True if speech generation completed successfully
                - latency_sec: Total end-to-end latency in seconds
                - ttft_ms: Time to first token/sound in milliseconds
                - rtr: Real-Time Ratio (audio_duration / processing_time)
                - audio_duration: Duration of generated audio in seconds
        """
        logger.info("🔊 Calling TTS /v1/audio/speech endpoint")

        # For eval workflow, all_params is an object with tasks attribute
        # For benchmark workflow, all_params is a list, so use default text
        text = DEFAULT_TTS_TEXT
        if (
            not isinstance(self.all_params, (list, tuple))
            and hasattr(self.all_params, "tasks")
            and len(self.all_params.tasks) > 0
        ):
            if hasattr(self.all_params.tasks[0], "text"):
                text = self.all_params.tasks[0].text
            elif hasattr(self.all_params.tasks[0], "task_name"):
                text = self.all_params.tasks[0].task_name

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {"text": text, "response_format": "json"}

        url = f"{self.base_url}/v1/audio/speech"
        start_time = time.monotonic()
        ttft_ms = None
        audio_duration = None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=120
                    ),  # cold start can exceed 90s
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
            logger.error(f"TTS generation failed: {type(e).__name__}: {e}")
            return False, 0.0, None, None, None

    def _calculate_ttft_value(self, status_list: list[TtsTestStatus]) -> float:
        """Calculate average TTFT value in milliseconds."""
        logger.info("Calculating TTFT value")

        ttft_value = 0
        if status_list:
            valid_ttft_values = [
                status.ttft_ms for status in status_list if status.ttft_ms is not None
            ]
            ttft_value = (
                sum(valid_ttft_values) / len(valid_ttft_values)
                if valid_ttft_values
                else 0
            )

        return ttft_value

    def _calculate_rtr_value(self, status_list: list[TtsTestStatus]) -> float:
        """Calculate average RTR value."""
        logger.info("Calculating RTR value")

        rtr_value = 0
        if status_list:
            valid_rtr_values = [
                status.rtr for status in status_list if status.rtr is not None
            ]
            rtr_value = (
                sum(valid_rtr_values) / len(valid_rtr_values) if valid_rtr_values else 0
            )

        return rtr_value

    def _calculate_tail_latency(
        self, status_list: list[TtsTestStatus]
    ) -> tuple[float, float]:
        """Calculate P90 and P95 tail latency for TTFT."""
        logger.info("Calculating tail latency (P90, P95)")

        if not status_list:
            return 0.0, 0.0

        valid_ttft_values = [
            status.ttft_ms for status in status_list if status.ttft_ms is not None
        ]

        if not valid_ttft_values:
            return 0.0, 0.0

        sorted_ttft = sorted(valid_ttft_values)
        n = len(sorted_ttft)
        p90_index = min(math.ceil(n * 0.9) - 1, n - 1)
        p95_index = min(math.ceil(n * 0.95) - 1, n - 1)
        p90_ttft = sorted_ttft[p90_index]
        p95_ttft = sorted_ttft[p95_index]

        return p90_ttft, p95_ttft

    def _calculate_performance_check(
        self,
        ttft_value: Optional[float] = None,
        rtr_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """Calculate performance check based on TTFT and RTR targets.

        Uses get_performance_targets from model_performance_reference.json.

        Args:
            ttft_value: Time to first token in milliseconds
            rtr_value: Real-time ratio (audio_duration / generation_time)

        Returns:
            ``ReportCheckTypes`` member. ``IntEnum`` serialises to JSON as its
            integer value, so downstream consumers still see 1/2/3 (NA/PASS/FAIL).
        """
        logger.info("Calculating performance check based on TTFT, RTR targets")

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
            return ReportCheckTypes.NA

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
            return ReportCheckTypes.NA

        if checks_passed == checks_total:
            logger.info(f"✅ All {checks_total} performance checks passed")
            return ReportCheckTypes.PASS

        logger.warning(
            f"❌ {checks_total - checks_passed}/{checks_total} performance checks failed"
        )
        return ReportCheckTypes.FAIL

    def _calculate_accuracy_check(self) -> ReportCheckTypes:
        """No quality metric implemented yet for TTS; always reports N/A."""
        return ReportCheckTypes.NA

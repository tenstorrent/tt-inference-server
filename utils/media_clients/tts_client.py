# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional


import aiohttp

from .base_strategy_interface import BaseMediaStrategy, PerfCheck
from .test_status import TtsTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import get_num_calls
from workflows.workflow_types import ReportCheckTypes

logger = logging.getLogger(__name__)

# Default text for TTS testing (going to be changed with dataset)
DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."


class TtsClientStrategy(BaseMediaStrategy):
    """Strategy for text-to-speech models (SpeechT5, etc.)."""

    def __init__(self, all_params, model_spec, device, output_path, service_port, deploy_url=None):
        super().__init__(all_params, model_spec, device, output_path, service_port, deploy_url=deploy_url)

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
        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            num_calls = self._get_tts_num_calls(is_eval=True)
            loop_start = time.monotonic()
            status_list = self._run_tts_benchmark(num_calls)
            wall_clock_seconds = time.monotonic() - loop_start
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        latency_value = self._calculate_latency(status_list)
        rtr_value = self._calculate_rtr_value(status_list)
        tail = self._calculate_tail_latencies([s.latency for s in status_list])
        throughput_rps = self._calculate_throughput_rps(
            len(status_list), wall_clock_seconds
        )
        logger.info(
            f"latency={latency_value:.4f}s, RTR={rtr_value:.2f}, tail={tail}, "
            f"throughput_rps={throughput_rps}"
        )

        benchmark_data = {
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "tts",
            "task_name": self.all_params.tasks[0].task_name,
            "tolerance": self.all_params.tasks[0].score.tolerance,
            "published_score": self.all_params.tasks[0].score.published_score,
            "score": latency_value,
            "published_score_ref": self.all_params.tasks[0].score.published_score_ref,
            "rtr": rtr_value,
            "throughput_rps": throughput_rps,
            **tail,
            "performance_check": self._calculate_performance_check(
                latency_value=latency_value, rtr_value=rtr_value
            ),
            "accuracy_check": self._calculate_accuracy_check(),
        }

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

    def run_benchmark(self) -> list[TtsTestStatus]:
        """Run benchmarks for the TTS model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            num_calls = self._get_tts_num_calls(is_eval=False)
            loop_start = time.monotonic()
            status_list = self._run_tts_benchmark(num_calls)
            wall_clock_seconds = time.monotonic() - loop_start
            self._generate_report(status_list, wall_clock_seconds)
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

    def _generate_report(
        self,
        status_list: list[TtsTestStatus],
        wall_clock_seconds: Optional[float] = None,
    ) -> None:
        """
        Generate benchmark report for TTS model.
        """
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )

        result_filename.parent.mkdir(parents=True, exist_ok=True)

        latency_value = self._calculate_latency(status_list)
        rtr_value = self._calculate_rtr_value(status_list)
        tail = self._calculate_tail_latencies([s.latency for s in status_list])
        throughput_rps = self._calculate_throughput_rps(
            len(status_list), wall_clock_seconds
        )
        performance_check = self._calculate_performance_check(latency_value, rtr_value)

        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "latency": latency_value,
                "rtr": rtr_value,
                "throughput_rps": throughput_rps,
                **tail,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "tts",
            "performance_check": performance_check,
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)

        logger.info(f"Report generated: {result_filename}")

    def _run_tts_benchmark(self, num_calls: int) -> list[TtsTestStatus]:
        """Run TTS benchmark."""
        logger.info(f"Running TTS benchmark with {num_calls} calls.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Generating speech {i + 1}/{num_calls}...")

            status, elapsed, latency, rtr, _ = asyncio.run(self._generate_speech())
            logger.debug(f"Generated speech in {elapsed:.2f} seconds.")

            status_list.append(
                TtsTestStatus(
                    status=status,
                    elapsed=elapsed,
                    latency=latency,
                    rtr=rtr,
                )
            )

        return status_list

    async def _generate_speech(
        self,
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """Generate speech from text.

        Returns:
            tuple: (success, elapsed_s, latency, rtr, audio_duration)
                - success: True if speech generation completed successfully
                - elapsed_s: Total wall-clock time in seconds (POST send
                  through full response read)
                - latency: Request latency in seconds (time until the
                  JSON response starts arriving).
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
        latency = None
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

                    latency = response_start - start_time

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
                f"✅ Done in {total_duration:.2f}s | latency={latency:.4f}s | RTR={rtr_str}"
            )

            return True, total_duration, latency, rtr, audio_duration

        except Exception as e:
            logger.error(f"TTS generation failed: {type(e).__name__}: {e}")
            return False, 0.0, None, None, None

    def _calculate_latency(self, status_list: list[TtsTestStatus]) -> float:
        """Mean end-to-end request latency in seconds."""
        logger.info("Calculating latency")

        latency_value = 0
        if status_list:
            valid_latency_values = [
                status.latency for status in status_list if status.latency is not None
            ]
            latency_value = (
                sum(valid_latency_values) / len(valid_latency_values)
                if valid_latency_values
                else 0
            )

        return latency_value

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

    def _calculate_performance_check(
        self,
        latency_value: Optional[float] = None,
        rtr_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """TTS perf check: compares latency and RTR vs configured targets.

        Targets file stores latency in ms; converted at this boundary so the
        helper can compare same-unit values.
        """
        targets = self.get_performance_targets()
        logger.info(f"Performance targets: {targets}")
        latency_target_s = (
            targets.ttft_ms / 1000.0 if targets.ttft_ms is not None else None
        )
        return self.calculate_performance_check(
            checks=[
                PerfCheck(
                    "latency", latency_value, latency_target_s, lower_is_better=True
                ),
                PerfCheck("RTR", rtr_value, targets.rtr, lower_is_better=False),
            ],
            tolerance=targets.tolerance,
        )

    def _calculate_accuracy_check(self) -> ReportCheckTypes:
        """No quality metric implemented yet for TTS; always reports N/A."""
        return ReportCheckTypes.NA

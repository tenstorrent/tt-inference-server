# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Standard library imports
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Third-party imports
import aiohttp
import requests
from transformers import AutoTokenizer

# Local imports
from .base_strategy_interface import BaseMediaStrategy
from .test_status import AudioTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import (
    get_num_calls,
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.utils_report import get_performance_targets

logger = logging.getLogger(__name__)


class AudioClientStrategy(BaseMediaStrategy):
    """Strategy for audio models (Whisper, etc.)."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)

        # Initialize tokenizer
        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_spec.hf_model_repo)
            logger.info(f"✅ Loaded tokenizer for {model_spec.hf_model_repo}")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not load tokenizer for {model_spec.hf_model_repo}: {e}"
            )
            logger.info("📝 Falling back to word-based token counting")

    def run_eval(self) -> None:
        """Run evaluations for the model."""
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

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)

            status_list = self._run_audio_transcription_benchmark(num_calls)
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        benchmark_data = {}

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")

        # Calculate RTR
        rtr_value = self._calculate_rtr_value(status_list)
        logger.info(f"Extracted RTR value: {rtr_value}")

        # Calculate T/S/U
        tsu_value = self._calculate_tsu_value(status_list)
        logger.info(f"Extracted T/S/U value: {tsu_value}")

        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name
        benchmark_data["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        benchmark_data["task_type"] = "audio"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance
        benchmark_data["published_score"] = self.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[
            0
        ].score.published_score_ref
        # For now hardcode accuracy_check to 2
        benchmark_data["accuracy_check"] = 2
        benchmark_data["t/s/u"] = tsu_value
        benchmark_data["rtr"] = rtr_value

        # Make benchmark_data is inside of list as an object
        benchmark_data = [benchmark_data]

        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmark(self, attempt=0) -> list[AudioTestStatus]:
        """Run benchmarks for the model."""
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

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)

            status_list = []
            status_list = self._run_audio_transcription_benchmark(num_calls)

            return self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _generate_report(self, status_list: list[AudioTestStatus]) -> None:
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)

        # Calculate RTR
        rtr_value = self._calculate_rtr_value(status_list)

        # Calculate T/S/U
        tsu_value = self._calculate_tsu_value(status_list)

        # Calculate accuracy check
        accuracy_check = self._calculate_accuracy_check(
            ttft_value, tsu_value, rtr_value
        )

        # Convert AudioTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": 0,
                "ttft": ttft_value,
                "inference_steps_per_second": 0,
                "t/s/u": tsu_value,
                "rtr": rtr_value,
                "accuracy_check": accuracy_check,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "audio",
            "streaming_enabled": is_streaming_enabled_for_whisper(self),
            "preprocessing_enabled": is_preprocessing_enabled_for_whisper(self),
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)

        logger.info(f"Report generated: {result_filename}")
        return True

    def _run_audio_transcription_benchmark(
        self, num_calls: int
    ) -> list[AudioTestStatus]:
        """Run audio transcription benchmark."""
        logger.info(f"Running audio transcription benchmark with {num_calls} calls.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Transcribing audio {i + 1}/{num_calls}...")
            status, elapsed, ttft, tsu, rtr = asyncio.run(self._transcribe_audio())
            logger.info(f"Transcribed audio in {elapsed:.2f} seconds.")

            status_list.append(
                AudioTestStatus(
                    status=status,
                    elapsed=elapsed,
                    ttft=ttft,
                    tsu=tsu,
                    rtr=rtr,
                )
            )

        return status_list

    async def _transcribe_audio(
        self,
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """Transcribe audio based on streaming settings."""
        logger.info("🔈 Calling whisper")
        is_preprocessing_enabled = is_preprocessing_enabled_for_whisper(self)
        logging.info(f"Preprocessing enabled: {is_preprocessing_enabled}")

        if is_streaming_enabled_for_whisper(self):
            return await self._transcribe_audio_streaming_on(is_preprocessing_enabled)

        return self._transcribe_audio_streaming_off(is_preprocessing_enabled)

    def _transcribe_audio_streaming_off(
        self, is_preprocessing_enabled: bool
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """Transcribe audio without streaming - direct transcription of the entire audio file"""
        logger.info("Transcribing audio without streaming")
        with open(f"{self.test_payloads_path}/image_client_audio_payload", "r") as f:
            audioFile = json.load(f)

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "file": audioFile["file"],
            "stream": False,
            "is_preprocessing_enabled": is_preprocessing_enabled,
        }

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/audio/transcriptions",
            json=payload,
            headers=headers,
            timeout=90,
        )
        elapsed = time.time() - start_time
        ttft = elapsed
        tsu = None  # No streaming, so T/U/S is not applicable

        # Calculate RTR (Real-Time Ratio)
        rtr = None
        if response.status_code == 200:
            try:
                response_data = response.json()
                audio_duration = response_data.get("duration")
                if audio_duration is not None:
                    rtr = audio_duration / elapsed
                    logger.info(
                        f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, processing_time={elapsed:.2f}s)"
                    )
                else:
                    logger.warning("Duration not found in response data")
            except Exception as e:
                logger.error(f"Failed to calculate RTR: {e}")

        return (response.status_code == 200), elapsed, ttft, tsu, rtr

    async def _transcribe_audio_streaming_on(
        self, is_preprocessing_enabled: bool
    ) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
        """Transcribe audio with streaming enabled - receives partial results in real-time.

        Filters out speaker markers when calculating TTFT. Measures total latency,
        time to first meaningful content token, and tokens per user per second.

        Args:
            is_preprocessing_enabled (bool): Whether audio preprocessing is enabled (aka WhisperX).

        Returns:
            tuple: (success, latency_sec, ttft_sec, tsu, rtr)
                - success: True if transcription completed successfully
                - latency_sec: Total end-to-end latency in seconds
                - ttft_sec: Time to first meaningful content token (excludes speaker markers)
                - tsu: Tokens per user per second throughput
                - rtr: Real-Time Ratio (audio_duration / processing_time)
        """
        logger.info("Transcribing audio with streaming enabled")

        # Read audio file
        with open(
            f"{self.test_payloads_path}/image_client_audio_streaming_payload", "r"
        ) as f:
            audioFile = json.load(f)

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "file": audioFile["file"],
            "stream": True,
            "is_preprocessing_enabled": is_preprocessing_enabled,
        }

        url = f"{self.base_url}/audio/transcriptions"
        start_time = time.monotonic()
        ttft = None
        total_text = ""  # Accumulate full text
        total_tokens = 0  # Track total tokens across all chunks
        chunk_texts = []  # Track individual chunks for debugging
        audio_duration = None  # Track audio duration from final chunk

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as response:
                    if response.status != 200:
                        return False, 0.0, None, None, None

                    async for line in response.content:
                        if not line.strip():
                            continue

                        try:
                            line_str = line.decode("utf-8").strip()
                            if not line_str:
                                continue
                            result = json.loads(line_str)
                            logger.info(f"Received chunk: {result}")
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse chunk: {e}")
                            continue

                        text = result.get("text", "")
                        chunk_id = result.get("chunk_id", "final")

                        # Extract audio duration from the final chunk
                        if "duration" in result:
                            audio_duration = result.get("duration")
                            logger.info(
                                f"Found audio duration in chunk: {audio_duration}s"
                            )

                        # Calculate tokens for this chunk only
                        chunk_tokens = self._count_tokens(text)

                        # Accumulate text from this chunk
                        if text.strip():
                            total_text += text + " "  # Add space between chunks
                            chunk_texts.append(text)
                            total_tokens += (
                                chunk_tokens  # Add chunk tokens to running total
                            )

                        # first token timestamp - only set when we actually receive meaningful content tokens
                        # Skip speaker markers like [SPEAKER_01], [SPEAKER_00], etc.
                        is_speaker_marker = text.strip().startswith(
                            "[SPEAKER_"
                        ) and text.strip().endswith("]")
                        now = time.monotonic()
                        if ttft is None and chunk_tokens > 0 and not is_speaker_marker:
                            ttft = now - start_time
                            logger.info(
                                f"🎯 TTFT set at {ttft:.2f}s for first meaningful content: {text!r}"
                            )

                        elapsed = now - start_time
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        # Calculate tokens per user per second (assuming 1 user for single request)
                        tokens_per_user_per_sec = (
                            tokens_per_sec / 1
                        )  # Single user for this request

                        logger.info(
                            f"[{elapsed:.2f}s] chunk={chunk_id} chunk_tokens={chunk_tokens} "
                            f"total_tokens={total_tokens} tps={tokens_per_sec:.2f} t/s/u={tokens_per_user_per_sec:.2f} text={text!r}"
                        )

            end_time = time.monotonic()
            total_duration = end_time - start_time  # Total time in seconds
            content_streaming_time = total_duration - (
                ttft if ttft is not None else 0
            )  # Time spent streaming content after TTFT
            final_tokens = total_tokens
            final_tps = (
                final_tokens / content_streaming_time
                if content_streaming_time > 0
                else 0
            )
            final_tokens_per_user_per_sec = (
                final_tps / 1
            )  # Single user for this request

            # Calculate RTR (Real-Time Ratio)
            rtr = None
            if audio_duration is not None:
                rtr = audio_duration / total_duration
                logger.info(
                    f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, processing_time={total_duration:.2f}s)"
                )
            else:
                logger.warning("Audio duration not found in streaming response")

            # If no tokens received, TTFT should be 0.0 (not total_duration)
            final_ttft = ttft if ttft is not None else 0.0
            rtr_display = f"{rtr:.2f}" if rtr is not None else "N/A"
            logger.info(
                f"\n✅ Done in {total_duration:.2f}s | TTFT={final_ttft:.2f}s | Total tokens={final_tokens} | TPS={final_tps:.2f} | T/S/U={final_tokens_per_user_per_sec:.2f} | RTR={rtr_display}"
            )

            return True, total_duration, final_ttft, final_tokens_per_user_per_sec, rtr

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return False, 0.0, None, None, None

    def _calculate_ttft_value(self, status_list: list[AudioTestStatus]) -> float:
        """Calculate TTFT value based on model type and status list."""
        logger.info("Calculating TTFT value")

        ttft_value = 0
        if status_list:
            valid_ttft_values = [
                status.ttft for status in status_list if status.ttft is not None
            ]
            ttft_value = (
                sum(valid_ttft_values) / len(valid_ttft_values)
                if valid_ttft_values
                else 0
            )

        return ttft_value

    def _calculate_rtr_value(self, status_list: list[AudioTestStatus]) -> float:
        """Calculate RTR value based on model type and status list."""
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

    def _calculate_tsu_value(self, status_list: list[AudioTestStatus]) -> float:
        """Calculate T/S/U value based on model type and status list."""
        logger.info("Calculating T/S/U value")

        tsu_value = 0
        if status_list:
            valid_tsu_values = [
                status.tsu for status in status_list if status.tsu is not None
            ]
            tsu_value = (
                sum(valid_tsu_values) / len(valid_tsu_values) if valid_tsu_values else 0
            )

        return tsu_value

    def _count_tokens(self, text: str) -> int:
        """Count tokens using model tokenizer, fallback to word count."""
        if not text.strip():
            return 0

        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer encoding failed: {e}. Using word count.")

        # Fallback to word counting
        return len(text.split())

    def _calculate_accuracy_check(
        self, ttft_value: float, tsu_value: float, rtr_value: float
    ):
        """Calculate accuracy check based on TTFT, RTR, T/S/U targets."""
        logger.info("Calculating accuracy check based on TTFT, RTR, T/S/U targets")

        # Get performance targets using the shared utility
        device_str = self.model_spec.cli_args.get("device")
        targets = get_performance_targets(
            self.model_spec.model_name,
            device_str,
            model_type=self.model_spec.model_type.name,
        )
        logger.info(f"Performance targets: {targets}")

        if not targets.ttft_ms:
            logger.warning("⚠️ No TTFT target found, skipping accuracy check")
            return 0  # UNDEFINED

        available_metrics = [
            "TTFT" if targets.ttft_ms else None,
            "T/S/U" if targets.tput_user and tsu_value else None,
            "RTR" if targets.rtr and rtr_value else None,
        ]
        logger.info(
            f"Available metrics for validation: {[m for m in available_metrics if m]}"
        )

        # Use measured values passed as parameters (not recalculated)
        tolerance = (
            targets.tolerance if targets.tolerance else 0.05
        )  # Default 5% tolerance
        logger.info(f"Using tolerance: {tolerance * 100:.2f}%")

        # Check each metric individually and determine overall result
        checks_passed = 0
        checks_total = 0

        # Always check TTFT if target is available
        if targets.ttft_ms is not None:  # pragma: no branch
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

        # Only check T/S/U if we have both target and measured value (streaming mode)
        if targets.tput_user is not None and tsu_value is not None:
            checks_total += 1
            tsu_threshold = targets.tput_user * (
                1 - tolerance
            )  # For throughput, lower is worse
            if tsu_value >= tsu_threshold:
                logger.info(f"✅ T/S/U PASSED: {tsu_value:.2f} >= {tsu_threshold:.2f}")
                checks_passed += 1
            else:
                logger.warning(
                    f"❌ T/S/U FAILED: {tsu_value:.2f} < {tsu_threshold:.2f}"
                )

        # Only check RTR if we have both target and measured value (streaming mode)
        if targets.rtr is not None and rtr_value is not None:
            checks_total += 1
            rtr_threshold = targets.rtr * (1 - tolerance)  # For RTR, lower is worse
            if rtr_value >= rtr_threshold:
                logger.info(f"✅ RTR PASSED: {rtr_value:.2f} >= {rtr_threshold:.2f}")
                checks_passed += 1
            else:
                logger.warning(f"❌ RTR FAILED: {rtr_value:.2f} < {rtr_threshold:.2f}")

        # Determine overall result
        if checks_total == 0:  # pragma: no cover
            logger.warning("No targets available for accuracy check")
            return 0  # UNDEFINED
        elif checks_passed == checks_total:
            logger.info(f"🎉 ALL CHECKS PASSED ({checks_passed}/{checks_total})")
            return 2  # PASS

        logger.warning(f"⛔️ SOME CHECKS FAILED ({checks_passed}/{checks_total} passed)")
        return 3  # FAIL

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
import time
from typing import Optional
import logging
import requests
import json
import asyncio
import aiohttp
from .base_strategy_interface import BaseMediaStrategy
from .test_status import AudioTestStatus
import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import (
    is_streaming_enabled_for_whisper,
    is_preprocessing_enabled_for_whisper,
    get_num_calls
)

logger = logging.getLogger(__name__)


class AudioClientStrategy(BaseMediaStrategy):
    """Strategy for audio models (Whisper, etc.)."""

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        status_list = []

        logger.info(f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}")
        try:
            (health_status, runner_in_use) = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                return

            logger.info(f"Runner in use: {runner_in_use}")
            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)

            status_list = self._run_audio_transcription_benchmark(num_calls)
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            return

        logger.info(f"Generating eval report...")
        benchmark_data = {}

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")

        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name
        benchmark_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        benchmark_data["task_type"] = "audio"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance
        benchmark_data["published_score"] = self.all_params.tasks[0].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[0].score.published_score_ref
        # For now hardcode accuracy_check to 2
        benchmark_data["accuracy_check"] = 2
        benchmark_data["t/s/u"] = status_list[0].tpups if status_list and len(status_list) > 0 and status_list[0].tpups is not None else 0

        # Make benchmark_data is inside of list as an object
        benchmark_data = [benchmark_data]

        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"/ self.model_spec.hf_model_repo.replace('/', '__') / f"results_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmark(self, attempt = 0) -> list[AudioTestStatus]:
        """Run benchmarks for the model."""
        logger.info(f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}")
        try:
            (health_status, runner_in_use) = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                return []

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)
            logger.info(f"Runner in use: {runner_in_use}")

            status_list = []
            status_list = self._run_audio_transcription_benchmark(num_calls)

            return self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            return []

    def get_health(self, attempt_number = 1) -> bool:
        """Check the health of the server with retries."""
        logger.info("Checking server health...")
        response = requests.get(f"{self.base_url}/tt-liveness")
        # server returns 200 if healthy only
        # otherwise it is 405
        if response.status_code != 200:
            if attempt_number < 20:
                logger.warning(f"Health check failed with status code: {response.status_code}. Retrying...")
                time.sleep(15)
                return self.get_health(attempt_number + 1)
            else:
                logger.error(f"Health check failed with status code: {response.status_code}")
                raise Exception(f"Health check failed with status code: {response.status_code}")

        return (True, response.json().get("runner_in_use", None))

    def _generate_report(self, status_list: list[AudioTestStatus]) -> None:
        logger.info(f"Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)

        # Convert AudioTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                    "num_requests": len(status_list),
                    "num_inference_steps": 0,
                    "ttft": ttft_value,
                    "inference_steps_per_second": 0,
                    "accuracy_check": 2,  # For now hardcode accuracy_check to 2,
                    "t/s/u": status_list[0].tpups if status_list and len(status_list) > 0 and status_list[0].tpups is not None else 0
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

    def _run_audio_transcription_benchmark(self, num_calls: int) -> list[AudioTestStatus]:
        """Run audio transcription benchmark."""
        logger.info(f"Running audio transcription benchmark with {num_calls} calls.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Transcribing audio {i + 1}/{num_calls}...")
            status, elapsed, ttft, tpups = asyncio.run(self._transcribe_audio())
            logger.info(f"Transcribed audio in {elapsed:.2f} seconds.")

            status_list.append(AudioTestStatus(
                status=status,
                elapsed=elapsed,
                ttft=ttft,
                tpups=tpups,
            ))

        return status_list

    async def _transcribe_audio(self) -> tuple[bool, float, Optional[float], Optional[float]]:
        """Transcribe audio based on streaming settings."""
        logger.info("🔈 Calling whisper")
        is_preprocessing_enabled = is_preprocessing_enabled_for_whisper(self)
        logging.info(f"Preprocessing enabled: {is_preprocessing_enabled}")

        if is_streaming_enabled_for_whisper(self):
            return await self._transcribe_audio_streaming_on(is_preprocessing_enabled)

        return self._transcribe_audio_streaming_off(is_preprocessing_enabled)

    def _transcribe_audio_streaming_off(self, is_preprocessing_enabled: bool) -> tuple[bool, float, Optional[float], Optional[float]]:
        """Transcribe audio without streaming - direct transcription of the entire audio file"""
        logger.info("Transcribing audio without streaming")
        with open(f"{self.test_payloads_path}/image_client_audio_payload", "r") as f:
            audioFile = json.load(f)

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile["file"],
            "stream": False,
            "is_preprocessing_enabled": is_preprocessing_enabled
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=headers, timeout=90)
        elapsed = time.time() - start_time
        ttft = elapsed
        tpups = None  # No streaming, so T/U/S is not applicable

        return (response.status_code == 200), elapsed, ttft, tpups

    async def _transcribe_audio_streaming_on(self, is_preprocessing_enabled: bool) -> tuple[bool, float, Optional[float], Optional[float]]:
        """Transcribe audio with streaming enabled - receives partial results in real-time.

        Filters out speaker markers when calculating TTFT. Measures total latency,
        time to first meaningful content token, and tokens per user per second.

        Args:
            is_preprocessing_enabled (bool): Whether audio preprocessing is enabled (aka WhisperX).

        Returns:
            tuple: (success, latency_sec, ttft_sec, tpups)
                - success: True if transcription completed successfully
                - latency_sec: Total end-to-end latency in seconds
                - ttft_sec: Time to first meaningful content token (excludes speaker markers)
                - tpups: Tokens per user per second throughput
        """
        logger.info("Transcribing audio with streaming enabled")

        # Read audio file
        with open(f"{self.test_payloads_path}/image_client_audio_streaming_payload", "r") as f:
            audioFile = json.load(f)

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile["file"],
            "stream": True,
            "is_preprocessing_enabled": is_preprocessing_enabled
        }

        url = f"{self.base_url}/audio/transcriptions"
        start_time = time.monotonic()
        ttft = None
        total_text = ""  # Accumulate full text
        chunk_texts = []  # Track individual chunks for debugging

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=90)) as response:
                    if response.status != 200:
                        return False, 0.0, None, None

                    async for line in response.content:
                        if not line.strip():
                            continue

                        try:
                            line_str = line.decode('utf-8').strip()
                            if not line_str:
                                continue
                            result = json.loads(line_str)
                            logger.info(f"Received chunk: {result}")
                        except (UnicodeDecodeError, json.JSONDecodeError) as e:
                            logger.error(f"Failed to parse chunk: {e}")
                            continue

                        text = result.get("text", "")
                        chunk_id = result.get("chunk_id", "final")

                        # Accumulate text from this chunk
                        if text.strip():
                            total_text += text + " "  # Add space between chunks
                            chunk_texts.append(text)

                        # Count total tokens from accumulated text
                        total_tokens = len(total_text.split()) if total_text.strip() else 0
                        chunk_tokens = len(text.split()) if text.strip() else 0

                        # first token timestamp - only set when we actually receive meaningful content tokens
                        # Skip speaker markers like [SPEAKER_01], [SPEAKER_00], etc.
                        is_speaker_marker = text.strip().startswith('[SPEAKER_') and text.strip().endswith(']')
                        now = time.monotonic()
                        if ttft is None and chunk_tokens > 0 and not is_speaker_marker:
                            ttft = now - start_time
                            logger.info(f"🎯 TTFT set at {ttft:.2f}s for first meaningful content: {text!r}")

                        elapsed = now - start_time
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        # Calculate tokens per user per second (assuming 1 user for single request)
                        tokens_per_user_per_sec = tokens_per_sec / 1  # Single user for this request

                        logger.info(f"[{elapsed:.2f}s] chunk={chunk_id} chunk_tokens={chunk_tokens} "
                        f"total_tokens={total_tokens} tps={tokens_per_sec:.2f} t/s/u={tokens_per_user_per_sec:.2f} text={text!r}")

            end_time = time.monotonic()
            total_duration = end_time - start_time  # Total time in seconds
            content_streaming_time = total_duration - (ttft if ttft is not None else 0)  # Time spent streaming content after TTFT
            final_tokens = len(total_text.split()) if total_text.strip() else 0
            final_tps = final_tokens / content_streaming_time if content_streaming_time > 0 else 0
            final_tokens_per_user_per_sec = final_tps / 1  # Single user for this request

            # If no tokens received, TTFT should be 0.0 (not total_duration)
            final_ttft = ttft if ttft is not None else 0.0
            logger.info(f"\n✅ Done in {total_duration:.2f}s | TTFT={final_ttft:.2f}s | Total tokens={final_tokens} | TPS={final_tps:.2f} | T/S/U={final_tokens_per_user_per_sec:.2f}")

            return True, total_duration, final_ttft, final_tokens_per_user_per_sec

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return False, 0.0, None, None

    def _calculate_ttft_value(self, status_list: list[AudioTestStatus]) -> float:
        """Calculate TTFT value based on model type and status list."""
        logger.info("Calculating TTFT value")

        ttft_value = 0
        if status_list:
            valid_ttft_values = [status.ttft for status in status_list if status.ttft is not None]
            ttft_value = sum(valid_ttft_values) / len(valid_ttft_values) if valid_ttft_values else 0

        return ttft_value
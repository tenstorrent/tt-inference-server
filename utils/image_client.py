# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from time import time
import time as time_module
from typing import Optional
import logging
import requests
import json
import asyncio
import time
import aiohttp
import glob

logger = logging.getLogger(__name__)

class SDXLTestStatus:
    status: bool
    elapsed: float
    num_inference_steps: Optional[int]
    inference_steps_per_second: Optional[float]
    ttft: Optional[float] # time to first token
    tpups: Optional[float] # tokens per user per second

    def __init__(self, status: bool, elapsed: float, num_inference_steps: int = 0, inference_steps_per_second: float = 0, ttft: Optional[float] = None, tpups: Optional[float] = None):
        self.status = status
        self.elapsed = elapsed
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.ttft = ttft
        self.tpups = tpups

class ImageClient:
    def __init__(self, all_params, model_spec, device, output_path, service_port):
        self.base_url = "http://localhost:" + str(service_port)
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.test_payloads_path = "utils/test_payloads"

    def get_health(self, attempt_number = 1) -> bool:
        """Check the health of the server with retries."""
        response = requests.get(f"{self.base_url}/tt-liveness")
        # server returns 200 if healthy only
        # otherwise it is 405
        if response.status_code != 200:
            if attempt_number < 20:
                logger.warning(f"Health check failed with status code: {response.status_code}. Retrying...")
                time_module.sleep(15)
                return self.get_health(attempt_number + 1)
            else:
                logger.error(f"Health check failed with status code: {response.status_code}")
                raise Exception(f"Health check failed with status code: {response.status_code}")

        return (True, response.json().get("runner_in_use", None))

    def _read_latest_benchmark_json(self) -> dict:
        """Read the latest benchmark JSON file based on timestamp."""
        # Pattern to match benchmark JSON files
        # Remove 'evals_output' from the path if present
        output_path_clean = str(Path(self.output_path)).replace("evals_output", "").rstrip("/")
        # Look in the benchmarks_output subdirectory
        pattern = str(Path(output_path_clean) / "benchmarks_output" / f"benchmark_{self.model_spec.model_id}_*.json")
        
        # Find all matching files
        json_files = glob.glob(pattern)
        
        if not json_files:
            raise FileNotFoundError(f"No benchmark JSON files found matching pattern: {pattern}")
        
        # Sort by modification time to get the latest
        latest_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)

    def run_evals(self) -> None:
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
        
            # Get num_calls from benchmark parameters
            num_calls = self._get_num_calls()

            is_image_generate_model = runner_in_use.startswith("tt-sd")
            is_audio_transcription_model = "whisper" in runner_in_use
            
            if runner_in_use and is_image_generate_model:
                status_list = self._run_image_generation_benchmark(num_calls)
            elif runner_in_use and is_audio_transcription_model:
                status_list = self._run_audio_transcription_benchmark(num_calls)
            elif runner_in_use and not is_image_generate_model:
                status_list = self._run_image_analysis_benchmark(num_calls)
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            return
        
        logger.info(f"Generating eval report...")
        benchmark_data = {}
        
        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")
        
        # Get streaming mode for whisper model only, default to False
        streaming_whisper = False
        if is_audio_transcription_model:
            streaming_whisper = self._get_streaming_setting_for_whisper()
        
        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name
        benchmark_data["timestamp"] = time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime())
        benchmark_data["task_type"] = "audio" if is_audio_transcription_model else "cnn"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance
        benchmark_data["published_score"] = self.all_params.tasks[0].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[0].score.published_score_ref
        # For now hardcode accuracy_check to 2
        benchmark_data["accuracy_check"] = 2
        if streaming_whisper:
            benchmark_data["t/u/s"] = status_list[0].tpups if status_list and len(status_list) > 0 and status_list[0].tpups is not None else 0
        
        # Make benchmark_data is inside of list as an object
        benchmark_data = [benchmark_data]
        
        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"/ self.model_spec.hf_model_repo.replace('/', '__') / f"results_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmarks(self, attempt = 0) -> list[SDXLTestStatus]:
        logger.info(f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}")
        try:
            (health_status, runner_in_use) = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                return []

            # Get num_calls from CNN benchmark parameters
            num_calls = self._get_num_calls()

            status_list = []
            
            is_image_generate_model = runner_in_use.startswith("tt-sd")
            is_audio_transcription_model = "whisper" in runner_in_use
            
            if runner_in_use and is_image_generate_model:
                status_list = self._run_image_generation_benchmark(num_calls)
            elif runner_in_use and is_audio_transcription_model:
                status_list = self._run_audio_transcription_benchmark(num_calls)
            elif runner_in_use and not is_image_generate_model:
                status_list = self._run_image_analysis_benchmark(num_calls)

            return self._generate_report(status_list, is_image_generate_model)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            return []
        
    def _get_num_calls(self) -> int:
        """Get number of calls from benchmark parameters."""
        logger.info("Extracting number of calls from benchmark parameters")

        # Guard clause: Handle single config object case (evals)
        if hasattr(self.all_params, 'tasks') and not isinstance(self.all_params, (list, tuple)):
            return 2 # hard coding for evals
        
        # Handle list/iterable case (benchmarks)
        cnn_params = next((param for param in self.all_params if hasattr(param, 'num_eval_runs')), None)
        return cnn_params.num_eval_runs if cnn_params and hasattr(cnn_params, 'num_eval_runs') else 2
    
    def _calculate_ttft_value(self, status_list: list[SDXLTestStatus]) -> float:
        """Calculate TTFT value based on model type and status list."""
        logger.info("Calculating TTFT value")
        ttft_value = 0
        if status_list:
            # For audio models (whisper), use average TTFT; for others, use average elapsed time
            if "whisper" in self.model_spec.model_id.lower():
                # Use average TTFT for audio models across all runs
                valid_ttft_values = [status.ttft for status in status_list if status.ttft is not None]
                ttft_value = sum(valid_ttft_values) / len(valid_ttft_values) if valid_ttft_values else 0
            else:
                # For other models, use average elapsed time
                ttft_value = sum(status.elapsed for status in status_list) / len(status_list)

        return ttft_value
    
    def _get_streaming_setting_for_whisper(self) -> bool:
        """Determine if streaming is enabled for the Whisper model based on CLI args. Default to False if not set"""
        logger.info("Checking if streaming is enabled for Whisper model")
        cli_args = getattr(self.model_spec, 'cli_args', {})
        
        # Check if streaming arg exists and has a valid value
        streaming_value = cli_args.get('streaming')
        if streaming_value is None:
            return False
        
        # Convert to string and check if it's 'true'
        streaming_enabled = str(streaming_value).lower() == 'true'
        
        return streaming_enabled
        
    def _run_image_generation_benchmark(self, num_calls: int) -> list[SDXLTestStatus]:
        logger.info(f"Running image generation benchmark.")
        status_list = []
        
        for i in range(1):
            logger.info(f"Generating image {i + 1}/{num_calls}...")
            status, elapsed = self._generate_image()
            inference_steps_per_second = 20 / elapsed if elapsed > 0 else 0
            logger.info(f"Generated image with {20} steps in {elapsed:.2f} seconds.")
            status_list.append(SDXLTestStatus(
                status=status,
                elapsed=elapsed,
                num_inference_steps=20,
                inference_steps_per_second=inference_steps_per_second
            ))

        return status_list
    
    def _run_audio_transcription_benchmark(self, num_calls: int) -> list[SDXLTestStatus]:
        logger.info(f"Running audio transcription benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Transcribing audio {i + 1}/{num_calls}...")
            status, elapsed, ttft, tpups = asyncio.run(self._transcribe_audio())
            logger.info(f"Transcribed audio in {elapsed:.2f} seconds.")
            status_list.append(SDXLTestStatus(
                status=status,
                elapsed=elapsed,
                ttft=ttft,
                tpups=tpups,
            ))

        return status_list
    
    def _run_image_analysis_benchmark(self, num_calls: int) -> list[SDXLTestStatus]:
        logger.info(f"Running image analysis benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Analyzing image {i + 1}/{num_calls}...")
            status, elapsed = self._analyze_image()
            logger.info(f"Analyzed image with {50} steps in {elapsed:.2f} seconds.")
            status_list.append(SDXLTestStatus(
                status=status,
                elapsed=elapsed,
            ))

        return status_list

    def _generate_report(self, status_list: list[SDXLTestStatus], is_image_generate_model: bool) -> None:
        logger.info(f"Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        
        # Convert SDXLTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                    "num_requests": len(status_list),
                    "num_inference_steps": status_list[0].num_inference_steps if status_list and is_image_generate_model else 0,
                    "ttft": ttft_value,
                    "inference_steps_per_second": sum(status.inference_steps_per_second for status in status_list) / len(status_list) if status_list and is_image_generate_model else 0,
                },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime()),
            "task_type": "cnn" if is_image_generate_model else "audio"
        }
        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

        return True

    def _generate_image(self, num_inference_steps: int = 20) -> tuple[bool, float]:
        """Generate image using SDXL model."""
        logger.info("Generating image 🌅")
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": "Rabbit",
            "seed": 0,
            "guidance_scale": 3.0,
            "number_of_images": 1,
            "num_inference_steps": num_inference_steps
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/image/generations", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time

        return (response.status_code == 200), elapsed
    
    def _analyze_image(self) -> tuple[bool, float]:
        """Analyze image using CNN model."""
        logger.info("Analyzing image 🔍")
        with open(f"{self.test_payloads_path}/image_client_image_payload.txt", "r") as f:
            imagePayload = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": imagePayload
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/cnn/search-image", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time

        return (response.status_code == 200), elapsed
    
    async def _transcribe_audio(self) -> tuple[bool, float, Optional[float], Optional[float]]:
        logger.info("Streaming whisper 🔈")
        if self._get_streaming_setting_for_whisper():
            return await self._transcribe_audio_streaming_on()

        return self._transcribe_audio_streaming_off()
    
    def _transcribe_audio_streaming_off(self) -> tuple[bool, float, Optional[float], Optional[float]]:
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
            "stream": False
        }
        
        start_time = time()
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        ttft = elapsed
        tpups = None  # No streaming, so T/U/S is not applicable
        
        return (response.status_code == 200), elapsed, ttft, tpups
    
    async def _transcribe_audio_streaming_on(self) -> tuple[bool, float, Optional[float], Optional[float]]:
        """Transcribe audio with streaming enabled - receives partial results
        Measures:
            - Total latency (end-to-end)
            - TTFT (time to first token)
            - Tokens per user per second (throughput)
        Returns:
            (success, latency_sec, ttft_sec, tpups)
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
            "stream": True
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

                        # first token timestamp - only set when we actually receive tokens
                        now = time.monotonic()
                        if ttft is None and chunk_tokens > 0:
                            ttft = now - start_time

                        elapsed = now - start_time
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        # Calculate tokens per user per second (assuming 1 user for single request)
                        tokens_per_user_per_sec = tokens_per_sec / 1  # Single user for this request

                        logger.info(f"[{elapsed:.2f}s] chunk={chunk_id} chunk_tokens={chunk_tokens} "
                            f"total_tokens={total_tokens} tps={tokens_per_sec:.2f} t/u/s={tokens_per_user_per_sec:.2f} text={text!r}")

            end_time = time.monotonic()
            total_time = end_time - start_time
            final_tokens = len(total_text.split()) if total_text.strip() else 0
            final_tps = final_tokens / total_time if total_time > 0 else 0
            final_tokens_per_user_per_sec = final_tps / 1  # Single user for this request
            
            # If no tokens received, TTFT should be 0.0 (not total_time)
            final_ttft = ttft if ttft is not None else 0.0
            logger.info(f"\n✅ Done in {total_time:.2f}s | TTFT={final_ttft:.2f}s | Total tokens={final_tokens} | TPS={final_tps:.2f} | T/U/S={final_tokens_per_user_per_sec:.2f}")

            return True, total_time, final_ttft, final_tokens_per_user_per_sec
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return False, 0.0, None, None
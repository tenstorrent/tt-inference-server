# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import logging
import time
from pathlib import Path

import requests

from utils.media_clients.test_status import VideoGenerationTestStatus
from workflows.utils import get_num_calls

from .base_strategy_interface import BaseMediaStrategy

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 5
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
INFERENCE_STEPS = {"mochi-1-preview": 50, "Wan2.2-T2V-A14B-Diffusers": 40}
VIDEO_JOB_STATUS_COMPLETED = "completed"
VIDEO_JOB_STATUS_FAILED = "failed"
VIDEO_JOB_STATUS_CANCELLED = "cancelled"


class VideoClientStrategy(BaseMediaStrategy):
    """Strategy for video generation models (Mochi, WAN, etc.)."""

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
            status_list = self._run_video_generation_benchmark(num_calls)

        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        benchmark_data = {}

        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name.lower()
        benchmark_data["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        benchmark_data["task_type"] = "video"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")

        benchmark_data["published_score"] = self.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[
            0
        ].score.published_score_ref

        # Make benchmark_data inside list as object
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

    def run_benchmark(self, attempt=0) -> None:
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

            # Get num_calls from video benchmark parameters
            num_calls = get_num_calls(self)

            status_list = self._run_video_generation_benchmark(num_calls)

            self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_video_generation_benchmark(
        self, num_calls: int
    ) -> list[VideoGenerationTestStatus]:
        """Run video generation benchmark."""
        logger.info("Running video generation benchmark.")
        status_list = []

        inference_steps = INFERENCE_STEPS[self.model_spec.model_name]
        logger.info(f"Inference steps: {inference_steps}")

        for i in range(num_calls):
            logger.info(f"Generating video {i + 1}/{num_calls}...")
            status, elapsed, job_id, video_path = self._generate_video(
                prompt=f"Test video generation {i + 1}",
                num_inference_steps=inference_steps,
            )
            logger.info(f"Generated video in {elapsed:.2f} seconds.")

            # Calculate inference steps per second if num_inference_steps is available
            inference_steps_per_second = inference_steps / elapsed if elapsed > 0 else 0

            status_list.append(
                VideoGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=inference_steps,
                    inference_steps_per_second=inference_steps_per_second,
                    job_id=job_id,
                    video_path=video_path,
                    prompt=f"Test video generation {i + 1}",
                )
            )

        return status_list

    def _generate_video(
        self, prompt: str, num_inference_steps: int = 20
    ) -> tuple[bool, float, str, str]:
        """Generate video using video API."""
        logger.info(f"🎬 Generating video with prompt: {prompt}")

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        logger.info(f"Payload: {payload}")

        # Submit video generation job
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/video/generations",
                json=payload,
                headers=headers,
                timeout=90,
            )

            if response.status_code != 202:
                logger.error(
                    f"Failed to submit video generation job: {response.status_code}"
                )
                return False, time.time() - start_time, "", ""

            job_data = response.json()
            job_id = job_data.get("id")
            logger.info(f"Video generation job submitted: {job_id}")

            # Poll for completion
            video_path = self._poll_video_completion(job_id, headers)
            elapsed = time.time() - start_time

            if video_path:
                logger.info(f"✅ Video generated successfully: {video_path}")
                return True, elapsed, job_id, video_path

            logger.error(f"❌ Video generation failed for job: {job_id}")
            return False, elapsed, job_id, ""
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Video generation error: {e}")
            return False, elapsed, "", ""

    def _poll_video_completion(
        self,
        job_id: str,
        headers: dict,
        polling_interval: int = DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS,
        timeout: int = DEFAULT_VIDEO_TIMEOUT_SECONDS,
    ) -> str:
        """Poll video generation job until completion or timeout."""
        logger.info(f"Polling video job: {job_id}")
        logger.info(f"Polling interval: {polling_interval} seconds")
        logger.info(f"Timeout: {timeout} seconds")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.base_url}/video/generations/{job_id}",
                    headers=headers,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.warning(f"Failed to get job status: {response.status_code}")
                    time.sleep(polling_interval)
                    continue

                job_data = response.json()
                status = job_data.get("status")
                logger.info(f"Job {job_id} status: {status}")

                if status == VIDEO_JOB_STATUS_COMPLETED:
                    video_path = self._download_video(job_id, headers)
                    return video_path
                elif status in [VIDEO_JOB_STATUS_FAILED, VIDEO_JOB_STATUS_CANCELLED]:
                    logger.error(f"Video generation {status}: {job_id}")
                    return ""

                logger.info(
                    f"Still processing, waiting {polling_interval} seconds and polling again"
                )
                time.sleep(polling_interval)

            except Exception as e:
                logger.error(f"Error polling job status: {e}")
                time.sleep(polling_interval)

        logger.error(f"Video generation timed out after {timeout}s")
        return ""

    def _download_video(self, job_id: str, headers: dict) -> str:
        """Download generated video."""
        logger.info(f"Downloading video for job: {job_id}")

        try:
            output_dir = "/tmp/videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            video_path = output_dir / f"{job_id}.mp4"

            response = requests.get(
                f"{self.base_url}/video/generations/{job_id}/download",
                headers=headers,
                timeout=300,
                stream=True,
            )

            if response.status_code != 200:
                logger.error(f"Failed to download video: {response.status_code}")
                return ""

            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Video downloaded: {video_path}")
            return str(video_path)

        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return ""

    def _generate_report(self, status_list: list[VideoGenerationTestStatus]) -> None:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)

        # Convert VideoGenerationTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": status_list[0].num_inference_steps
                if status_list
                else 0,
                "ttft": ttft_value,
                "inference_steps_per_second": sum(
                    status.inference_steps_per_second for status in status_list
                )
                / len(status_list)
                if status_list
                else 0,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "video",
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _calculate_ttft_value(
        self, status_list: list[VideoGenerationTestStatus]
    ) -> float:
        """Calculate TTFT value based on status list."""
        logger.info("Calculating TTFT value")

        return (
            sum(status.elapsed for status in status_list) / len(status_list)
            if status_list
            else 0
        )

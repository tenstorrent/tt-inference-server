# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import base64
import json
import logging
import time
from pathlib import Path

import requests

from utils.media_clients.test_status import VideoGenerationTestStatus
from workflows.utils import get_num_calls, get_repo_root_path
from workflows.workflow_types import ReportCheckTypes

from .base_strategy_interface import BaseMediaStrategy, PerfCheck
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 5
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
INFERENCE_STEPS = {
    "mochi-1-preview": 50,
    "Wan2.2-T2V-A14B-Diffusers": 40,
    "Wan2.2-I2V-A14B-Diffusers": 40,
}
# Models routed through the image-to-video endpoint instead of plain T2V.
I2V_MODEL_NAMES = frozenset({"Wan2.2-I2V-A14B-Diffusers"})

I2V_FIXTURE_IMAGE_RELPATH = (
    Path("server_tests") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
VIDEO_JOB_STATUS_COMPLETED = "completed"
VIDEO_JOB_STATUS_FAILED = "failed"
VIDEO_JOB_STATUS_CANCELLED = "cancelled"


def _load_i2v_fixture_image_base64() -> str:
    """Return the I2V conditioning frame, base64-encoded."""
    fixture_path = get_repo_root_path() / I2V_FIXTURE_IMAGE_RELPATH
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {fixture_path}. "
            "Expected a tracked sample from server_tests/datasets/imagenet_subset/."
        )
    return base64.b64encode(fixture_path.read_bytes()).decode("ascii")


class VideoClientStrategy(BaseMediaStrategy):
    """Strategy for video generation models (Mochi, WAN, etc.)."""

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            eval_result = self._run_video_generation_eval()
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

        # Extract metrics from eval result
        if eval_result:
            logger.info(
                "Adding eval results from video generation test to benchmark data"
            )
            benchmark_data["num_prompts"] = eval_result.get("num_prompts", 0)
            benchmark_data["num_inference_steps"] = eval_result.get(
                "num_inference_steps", 0
            )

            clip_results = eval_result.get("clip_results", {})
            benchmark_data["average_clip"] = clip_results.get("average_clip", 0.0)
            benchmark_data["min_clip"] = clip_results.get("min_clip", 0.0)
            benchmark_data["max_clip"] = clip_results.get("max_clip", 0.0)
            benchmark_data["clip_standard_deviation"] = clip_results.get(
                "clip_standard_deviation", 0.0
            )
            benchmark_data["accuracy_check"] = eval_result.get(
                "accuracy_check", ReportCheckTypes.NA
            )
        else:
            logger.warning("No eval results from video generation test")

        benchmark_data["fvd"] = None
        benchmark_data["fvmd"] = None
        try:
            # Run video FVD and FVMD eval
            fvd_and_fvmd_result = self._run_video_fvd_and_fvmd_eval()

            # Add FVD and FVMD results to eval data
            if fvd_and_fvmd_result:
                benchmark_data["fvd"] = fvd_and_fvmd_result.get("fvd", 0)
                benchmark_data["fvmd"] = fvd_and_fvmd_result.get("fvmd", 0)
        except Exception as e:
            logger.error(f"Error running video FVD and FVMD eval: {e}")

        benchmark_data["published_score"] = self.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["published_score_ref"] = self.all_params.tasks[
            0
        ].score.published_score_ref

        benchmark_data["performance_check"] = self._calculate_performance_check()

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

    def run_benchmark(self) -> None:
        """Run benchmarks for the model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            num_calls = get_num_calls(self)
            loop_start = time.monotonic()
            status_list = self._run_video_generation_benchmark(num_calls)
            wall_clock_seconds = time.monotonic() - loop_start
            self._generate_report(status_list, wall_clock_seconds)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _is_i2v_model(self) -> bool:
        return self.model_spec.model_name in I2V_MODEL_NAMES

    def _run_video_generation_benchmark(
        self, num_calls: int
    ) -> list[VideoGenerationTestStatus]:
        """Run video generation benchmark.

        Routes to the I2V endpoint (with a fixture conditioning image) for
        I2V models, otherwise the standard T2V endpoint.
        """
        logger.info("Running video generation benchmark.")
        status_list = []

        inference_steps = INFERENCE_STEPS[self.model_spec.model_name]
        is_i2v = self._is_i2v_model()
        logger.info(
            f"Inference steps: {inference_steps}, mode: {'i2v' if is_i2v else 't2v'}"
        )

        image_b64 = _load_i2v_fixture_image_base64() if is_i2v else None

        for i in range(num_calls):
            prompt = f"Test video generation {i + 1}"
            logger.info(f"Generating video {i + 1}/{num_calls}...")
            if is_i2v:
                status, elapsed, job_id, video_path = self._generate_video_i2v(
                    prompt=prompt,
                    image_b64=image_b64,
                    num_inference_steps=inference_steps,
                )
            else:
                status, elapsed, job_id, video_path = self._generate_video(
                    prompt=prompt,
                    num_inference_steps=inference_steps,
                )
            logger.info(f"Generated video in {elapsed:.2f} seconds.")

            inference_steps_per_second = inference_steps / elapsed if elapsed > 0 else 0

            status_list.append(
                VideoGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=inference_steps,
                    inference_steps_per_second=inference_steps_per_second,
                    job_id=job_id,
                    video_path=video_path,
                )
            )

        return status_list

    def _generate_video(
        self, prompt: str, num_inference_steps: int = 20
    ) -> tuple[bool, float, str, str]:
        """Generate video (T2V) using the video API."""
        logger.info(f"🎬 Generating video with prompt: {prompt}")
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        return self._submit_and_wait_for_video("/v1/videos/generations", payload)

    def _generate_video_i2v(
        self,
        prompt: str,
        image_b64: str,
        frame_pos: int = 0,
        num_inference_steps: int = 40,
    ) -> tuple[bool, float, str, str]:
        """Generate video (I2V) using the image-to-video API.

        The image must be base64-encoded (PNG or JPEG, no data URI prefix).
        ``frame_pos=0`` anchors the image as the first frame; supply a
        higher ``frame_pos`` to insert it mid-video. For multi-keyframe
        conditioning use :meth:`_generate_video_i2v_multi`.
        """
        logger.info(f"🎬 Generating I2V video with prompt: {prompt}")
        payload = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "image_prompts": [{"image": image_b64, "frame_pos": frame_pos}],
        }
        return self._submit_and_wait_for_video("/v1/videos/generations/i2v", payload)

    def _submit_and_wait_for_video(
        self, endpoint: str, payload: dict
    ) -> tuple[bool, float, str, str]:
        """Submit a video-generation job and poll until completion.

        Returns ``(success, elapsed_seconds, job_id, video_path)``.
        """
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        logger.info(f"POST {endpoint} payload keys: {list(payload.keys())}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers=headers,
                timeout=90,
            )

            if response.status_code != 202:
                logger.error(
                    f"Failed to submit video generation job: "
                    f"{response.status_code} {response.text[:500]}"
                )
                return False, time.time() - start_time, "", ""

            job_data = response.json()
            job_id = job_data.get("id")
            logger.info(f"Video generation job submitted: {job_id}")

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
                    f"{self.base_url}/v1/videos/generations/{job_id}",
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
            output_dir = Path("/tmp/videos")
            output_dir.mkdir(parents=True, exist_ok=True)

            video_path = output_dir / f"{job_id}.mp4"

            response = requests.get(
                f"{self.base_url}/v1/videos/generations/{job_id}/download",
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

    def _generate_report(
        self,
        status_list: list[VideoGenerationTestStatus],
        wall_clock_seconds: Optional[float] = None,
    ) -> None:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        latency_value = self._calculate_latency(status_list)
        performance_check = self._calculate_performance_check(
            latency_value=latency_value
        )
        tail = self._calculate_tail_latencies([s.elapsed for s in status_list])
        throughput_rps = self._calculate_throughput_rps(
            len(status_list), wall_clock_seconds
        )

        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": status_list[0].num_inference_steps
                if status_list
                else 0,
                "latency": latency_value,
                "inference_steps_per_second": sum(
                    status.inference_steps_per_second for status in status_list
                )
                / len(status_list)
                if status_list
                else 0,
                "throughput_rps": throughput_rps,
                **tail,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "video",
            "performance_check": performance_check,
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _calculate_latency(self, status_list: list[VideoGenerationTestStatus]) -> float:
        """Mean end-to-end request latency in seconds."""
        logger.info("Calculating latency")

        return (
            sum(status.elapsed for status in status_list) / len(status_list)
            if status_list
            else 0
        )

    def _calculate_performance_check(
        self,
        latency_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """Video perf check: compares latency vs configured target."""
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
            ],
            tolerance=targets.tolerance,
        )

    def _run_video_generation_eval(self) -> dict:
        """Run video generation eval using VideoGenerationEvalsTest.

        Returns:
            dict: eval_results with structure:
                {
                    "model": str,
                    "num_prompts": int,
                    "num_inference_steps": int,
                    "clip_results": {
                        "average_clip": float,
                        "min_clip": float,
                        "max_clip": float,
                        "clip_standard_deviation": float,
                        "num_videos": int
                    },
                    "accuracy_check": int
                }
        """
        # Lazy import to avoid loading dependencies at module import time
        from server_tests.test_cases.video_generation_eval_test import (
            VideoGenerationEvalsTest,
            VideoGenerationEvalsTestRequest,
        )
        from server_tests.test_classes import TestConfig

        logger.info("Running video generation eval.")

        # Get parameters from eval configuration with defaults
        # Use default values similar to metal team's approach
        num_prompts = 5  # Default for video evaluation
        num_inference_steps = 40  # Default for video models
        start_from = 0
        frame_sample_rate = 8  # Sample every 8th frame

        request = VideoGenerationEvalsTestRequest(
            model_name=self.model_spec.model_name,
            num_prompts=num_prompts,
            start_from=start_from,
            num_inference_steps=num_inference_steps,
            server_url=self.base_url,
            frame_sample_rate=frame_sample_rate,
        )
        logger.info(f"Running VideoGenerationEvalsTest with request: {request}")

        config = TestConfig.create_default()
        targets = {"request": request}
        test = VideoGenerationEvalsTest(config, targets)

        logger.info("Starting VideoGenerationEvalsTest")
        result = test.run_tests()

        # Extract eval_results from result
        eval_results = result.get("result", {}).get("eval_results", {})
        logger.info(f"VideoGenerationEvalsTest eval_results: {eval_results}")

        return eval_results

    def _run_video_fvd_and_fvmd_eval(self) -> dict:
        """Run video FVD and FVMD eval.

        Flow:
        1. Download reference videos from FineVideo (shared dataset dir).
        2. Run FVD test (compute_fvd) with reference vs generated videos.
        3. Run FVMD test (compute_fvmd) with same paths.
        4. Combine FVD and FVMD scores into a single dict.

        Reference videos: server_tests/datasets/video_fvd_subset/videos
        Generated videos: server_tests/datasets/videos (same as
        video_generation_eval_test) or /tmp/videos (video_client downloads).

        Returns:
            dict: Combined eval results, e.g.:
                {
                    "fvd": float,
                    "fvmd": float,
                    "reference_videos_path": str,
                    "generated_videos_path": str,
                }
        """
        from pathlib import Path

        from server_tests.test_cases.video_fvd_eval_test import (
            DATASET_DIR as FVD_DATASET_DIR,
        )
        from server_tests.test_cases.video_fvd_eval_test import (
            VideoFVDTest,
            VideoFVDTestRequest,
        )
        from server_tests.test_cases.video_fvmd_eval_test import (
            VideoFVMDTest,
            VideoFVMDTestRequest,
        )
        from server_tests.test_classes import TestConfig

        logger.info("Running video FVD and FVMD eval.")

        reference_videos_path = str(Path(FVD_DATASET_DIR) / "videos")
        # Generated videos: same dir as video_generation_eval_test, or /tmp/videos
        generated_videos_path = str(Path("server_tests/datasets/videos").resolve())
        if not Path(generated_videos_path).exists():
            generated_videos_path = "/tmp/videos"
        logger.info(
            f"Reference path: {reference_videos_path}, "
            f"generated path: {generated_videos_path}"
        )

        config = TestConfig.create_default()
        combined_results = {
            "reference_videos_path": reference_videos_path,
            "generated_videos_path": generated_videos_path,
        }

        # Step 1: Download reference videos (shared for FVD and FVMD)
        download_request = VideoFVDTestRequest(
            action="download",
            download_count=2,
            category="Sports",
        )
        download_test = VideoFVDTest(config, {"request": download_request})
        download_result = download_test.run_tests()
        if not download_result.get("success") or not download_result.get(
            "result", {}
        ).get("success"):
            logger.warning(
                "Reference video download failed: %s. "
                "FVD/FVMD may fail if reference dir is empty.",
                download_result,
            )
        else:
            logger.info("Reference videos downloaded successfully.")

        # Step 2: Run FVD (compute_fvd)
        fvd_request = VideoFVDTestRequest(
            action="compute_fvd",
            reference_videos_path=reference_videos_path,
            generated_videos_path=generated_videos_path,
        )
        fvd_test = VideoFVDTest(config, {"request": fvd_request})
        fvd_result = fvd_test.run_tests()
        fvd_ok = fvd_result.get("success") and fvd_result.get("result", {}).get(
            "success"
        )
        if fvd_ok:
            combined_results["fvd"] = fvd_result["result"].get("fvd_score")
            logger.info("FVD score: %s", combined_results["fvd"])
        else:
            combined_results["fvd"] = None
            logger.warning("FVD test failed or did not return score: %s", fvd_result)

        # Step 3: Run FVMD (compute_fvmd)
        fvmd_request = VideoFVMDTestRequest(
            action="compute_fvmd",
            reference_videos_path=reference_videos_path,
            generated_videos_path=generated_videos_path,
        )
        fvmd_test = VideoFVMDTest(config, {"request": fvmd_request})
        fvmd_result = fvmd_test.run_tests()
        fvmd_ok = fvmd_result.get("success") and fvmd_result.get("result", {}).get(
            "success"
        )
        if fvmd_ok:
            combined_results["fvmd"] = fvmd_result["result"].get("fvmd_score")
            logger.info("FVMD score: %s", combined_results["fvmd"])
        else:
            combined_results["fvmd"] = None
            logger.warning("FVMD test failed or did not return score: %s", fvmd_result)

        return combined_results

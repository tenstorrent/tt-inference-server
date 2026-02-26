# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Eval test for video generation models (Mochi, Wan2.2, etc.)."""

import json
import logging
import statistics
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Union

import imageio.v3 as iio
import requests
from PIL import Image

from tests.server_tests.base_test import BaseTest
from tests.server_tests.test_cases.server_helper import (
    DEFAULT_AUTHORIZATION,
    SERVER_BASE_URL,
)
from tests.server_tests.test_classes import TestConfig
from utils.sdxl_accuracy_utils.clip_encoder import CLIPEncoder
from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import sdxl_get_prompts

logger = logging.getLogger(__name__)


class AccuracyResult(IntEnum):
    """Accuracy check result codes."""

    UNDEFINED = 0
    BASELINE = 1
    PASS = 2
    FAIL = 3


@dataclass(frozen=True)
class VideoGenConfig:
    """Video generation eval configuration."""

    JOB_STATUS_COMPLETED: str = "completed"
    JOB_STATUS_FAILED: str = "failed"
    JOB_STATUS_CANCELLED: str = "cancelled"
    POLLING_INTERVAL_SECONDS: int = 5
    JOB_TIMEOUT_SECONDS: int = 1200
    ACCURACY_REFERENCE_PATH: str = "evals/eval_targets/model_accuracy_reference.json"
    ENDPOINT: str = "v1/videos/generations"
    DATASET_DIR: str = "tests/server_tests/datasets/videos"
    REQUEST_TIMEOUT: int = 90
    DOWNLOAD_TIMEOUT: int = 300
    POLL_STATUS_TIMEOUT: int = 30


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    MAX_ATTEMPTS: int = 230
    RETRY_DELAY: int = 10
    TIMEOUT: int = 10


CONFIG = VideoGenConfig()
HEALTH_CONFIG = HealthCheckConfig()


@dataclass
class VideoGenerationEvalsTestRequest:
    model_name: str
    num_prompts: int = 5
    start_from: int = 0
    num_inference_steps: int = 40
    server_url: str | None = None
    frame_sample_rate: int = 8  # Sample every Nth frame for CLIP scoring


class VideoGenerationEvalsTest(BaseTest):
    """Eval test for video generation models."""

    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self) -> dict:
        """Run the video generation evaluation test."""
        request = self._parse_request()
        if isinstance(request, dict):
            return request

        logger.info(
            "Running eval: model=%s, prompts=%s",
            request.model_name,
            request.num_prompts,
        )

        self.reference_data = self._load_accuracy_reference()
        effective_steps = self._get_num_inference_steps_from_reference(
            request.model_name, request.num_inference_steps
        )
        if effective_steps != request.num_inference_steps:
            logger.info(
                "Overriding num_inference_steps from %s to %s (from accuracy reference)",
                request.num_inference_steps,
                effective_steps,
            )

        logger.info(
            "Step 1: Loading %s prompts from COCO captions", request.num_prompts
        )
        prompts = sdxl_get_prompts(request.start_from, request.num_prompts)
        if len(prompts) < request.num_prompts:
            return self._error(
                f"VideoGenerationEvalTest only got {len(prompts)}/{request.num_prompts} prompts"
            )

        logger.info("Step 2: Generating videos")
        videos_info = self._generate_videos(
            prompts=prompts,
            server_url=request.server_url,
            num_inference_steps=effective_steps,
        )
        if len(videos_info) < request.num_prompts:
            return self._error(
                f"VideoGenerationEvalTest only {len(videos_info)}/{request.num_prompts} videos generated"
            )

        logger.info("Step 3: Calculating CLIP scores for video frames")
        clip_results = self._calculate_video_clip_scores(
            videos_info=videos_info,
            prompts=prompts,
            frame_sample_rate=request.frame_sample_rate,
        )

        logger.info("Step 4: Checking accuracy against reference")
        accuracy_check = self._check_accuracy(
            clip_results=clip_results,
            model_name=request.model_name,
            num_prompts=request.num_prompts,
        )

        results = {
            "model": request.model_name,
            "num_prompts": request.num_prompts,
            "num_inference_steps": effective_steps,
            "clip_results": clip_results,
            "accuracy_check": accuracy_check,
        }
        self.eval_results = results

        return {
            "success": accuracy_check == AccuracyResult.PASS,
            "eval_results": results,
        }

    def _parse_request(self) -> Union[VideoGenerationEvalsTestRequest, dict]:
        """Parse and validate request from targets."""
        raw = self.targets.get("request")
        if raw is None:
            return self._error(
                "VideoGenerationEvalTest request not provided in targets"
            )
        if isinstance(raw, VideoGenerationEvalsTestRequest):
            return raw
        if isinstance(raw, dict):
            try:
                return VideoGenerationEvalsTestRequest(**raw)
            except TypeError as e:
                return self._error(
                    f"VideoGenerationEvalTest invalid request parameters: {e}"
                )
        return self._error(
            "VideoGenerationEvalTest request must be dict or VideoGenerationEvalsTestRequest"
        )

    @staticmethod
    def _error(message: str) -> dict:
        """Create error response."""
        return {"success": False, "error": message}

    def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready."""
        health_url = f"http://localhost:{self.service_port}/tt-liveness"
        logger.info("Waiting for server: %s", health_url)
        for attempt in range(1, HEALTH_CONFIG.MAX_ATTEMPTS + 1):
            if self._check_health(health_url):
                logger.info("Server ready after %s attempt(s)", attempt)
                return True
            logger.debug(
                "Not ready (attempt %s/%s)", attempt, HEALTH_CONFIG.MAX_ATTEMPTS
            )
            time.sleep(HEALTH_CONFIG.RETRY_DELAY)
        logger.error("Server not ready after %s attempts", HEALTH_CONFIG.MAX_ATTEMPTS)
        return False

    def _check_health(self, url: str) -> bool:
        """Single health check attempt."""
        try:
            response = requests.get(url, timeout=HEALTH_CONFIG.TIMEOUT)
        except requests.RequestException:
            return False
        if response.status_code != 200:
            return False
        data = response.json()
        return data.get("status") == "alive" and data.get("model_ready", False)

    def _generate_videos(
        self,
        prompts: list[str],
        server_url: str | None,
        num_inference_steps: int,
    ) -> list[dict]:
        """Generate videos for all prompts.

        Args:
            prompts: List of text prompts
            server_url: Base server URL (e.g., http://localhost:8000)
            num_inference_steps: Number of inference steps for generation

        Returns:
            List of dicts with video information: {'prompt': str, 'video_path': str, 'job_id': str}
        """
        logger.info(f"Generating {len(prompts)} videos")

        # Wait for server to be ready
        if not self._wait_for_server_ready():
            raise RuntimeError("Server health check failed - server not ready")

        base_url = server_url or SERVER_BASE_URL
        videos_info = []

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }

        for idx, prompt in enumerate(prompts):
            logger.info(f"Generating video {idx + 1}/{len(prompts)}: {prompt[:50]}...")

            payload = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
            }

            try:
                # Submit video generation job
                response = requests.post(
                    f"{base_url}/{CONFIG.ENDPOINT}",
                    json=payload,
                    headers=headers,
                    timeout=CONFIG.REQUEST_TIMEOUT,
                )

                if response.status_code != 202:
                    logger.error(f"Failed to submit video job: {response.status_code}")
                    continue

                job_data = response.json()
                job_id = job_data.get("id")
                logger.info(f"Video job submitted: {job_id}")

                # Poll for completion
                video_path = self._poll_video_completion(
                    base_url=base_url,
                    job_id=job_id,
                    headers=headers,
                )

                if video_path:
                    videos_info.append(
                        {
                            "prompt": prompt,
                            "video_path": video_path,
                            "job_id": job_id,
                        }
                    )
                    logger.info(f"✅ Video generated successfully: {video_path}")
                else:
                    logger.error(f"❌ Video generation failed for job: {job_id}")

            except Exception as e:
                logger.error(f"Error generating video for prompt '{prompt[:50]}': {e}")
                continue

        logger.info(f"Generated {len(videos_info)}/{len(prompts)} videos successfully")
        return videos_info

    def _poll_video_completion(
        self,
        base_url: str,
        job_id: str,
        headers: dict,
        polling_interval: int = CONFIG.POLLING_INTERVAL_SECONDS,
        timeout: int = CONFIG.JOB_TIMEOUT_SECONDS,
    ) -> str:
        """Poll video generation job until completion or timeout.

        Returns:
            str: Path to downloaded video file, or empty string on failure.
        """
        logger.info(f"Polling video job: {job_id}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{base_url}/{CONFIG.ENDPOINT}/{job_id}",
                    headers=headers,
                    timeout=CONFIG.POLL_STATUS_TIMEOUT,
                )

                if response.status_code != 200:
                    logger.warning(f"Failed to get job status: {response.status_code}")
                    time.sleep(polling_interval)
                    continue

                job_data = response.json()
                status = job_data.get("status")
                logger.info(f"Job {job_id} status: {status}")

                if status == CONFIG.JOB_STATUS_COMPLETED:
                    return self._download_video(base_url, job_id, headers)
                elif status in [CONFIG.JOB_STATUS_FAILED, CONFIG.JOB_STATUS_CANCELLED]:
                    logger.error(f"Video generation {status}: {job_id}")
                    return ""

                time.sleep(polling_interval)

            except Exception as e:
                logger.error(f"Error polling job status: {e}")
                time.sleep(polling_interval)

        logger.error(f"Video generation timed out after {timeout}s")
        return ""

    def _download_video(self, base_url: str, job_id: str, headers: dict) -> str:
        """Download generated video.

        Returns:
            str: Path to downloaded video file, or empty string on failure.
        """
        logger.info(f"Downloading video for job: {job_id}")

        try:
            output_dir = Path(CONFIG.DATASET_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            video_path = output_dir / f"{job_id}.mp4"

            response = requests.get(
                f"{base_url}/{CONFIG.ENDPOINT}/{job_id}/download",
                headers=headers,
                timeout=CONFIG.DOWNLOAD_TIMEOUT,
                stream=True,
            )

            if response.status_code != 200:
                logger.error(f"Failed to download video: {response.status_code}")
                return ""

            with open(video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            logger.info(f"Video downloaded: {video_path}")
            return str(video_path)

        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return ""

    def _calculate_video_clip_scores(
        self,
        videos_info: list[dict],
        prompts: list[str],
        frame_sample_rate: int = 8,
    ) -> dict:
        """Calculate CLIP scores for video frames following tt-metal pattern.

        This method follows the tt-metal implementation pattern where:
        - Each video is represented as a list of PIL Image frames
        - CLIP scores are calculated per frame per video
        - Pattern: clip_scores = [[score for frame in video] for video in videos]

        Args:
            videos_info: List of video information dicts (contains prompt for each video)
            prompts: List of prompts (not directly used - prompts come from videos_info)
            frame_sample_rate: Sample every Nth frame (default: 8)

        Returns:
            dict: CLIP score statistics
        """
        logger.info("Calculating CLIP scores for video frames.")

        clip = CLIPEncoder()

        # Step 1: Extract all videos as list of frame lists
        # Structure matches videos_info but adds frames:
        # [{'prompt': str, 'video_path': str, 'frames': [PIL.Image, ...]}, ...]
        videos_with_frames = []
        for idx, video_info in enumerate(videos_info):
            video_path = video_info["video_path"]
            logger.info(
                f"Extracting frames from video {idx + 1}/{len(videos_info)}: {video_path}"
            )

            frames = self._extract_video_frames(video_path, frame_sample_rate)

            if not frames:
                logger.warning(f"No frames extracted from video: {video_path}")
                continue

            # Add frames to existing video_info (already has correct prompt)
            video_with_frames = {**video_info, "frames": frames}
            videos_with_frames.append(video_with_frames)
            logger.info(f"Video {idx + 1}: extracted {len(frames)} frames")

        if not videos_with_frames:
            logger.error("No videos with valid frames found")
            return {
                "average_clip": 0.0,
                "min_clip": 0.0,
                "max_clip": 0.0,
                "clip_standard_deviation": 0.0,
                "num_videos": 0,
            }

        # Step 2: Calculate CLIP scores using correct prompt from videos_info
        logger.info("Calculating CLIP scores for all frames...")
        clip_scores = [
            [
                100 * clip.get_clip_score(video["prompt"], img).item()
                for img in video["frames"]
            ]
            for video in videos_with_frames
        ]

        # Step 3: Calculate statistics per video
        video_clip_scores = []
        for idx, frame_scores in enumerate(clip_scores):
            if frame_scores:
                video_stats = {
                    "min": min(frame_scores),
                    "max": max(frame_scores),
                    "mean": statistics.mean(frame_scores),
                    "stddev": statistics.stdev(frame_scores)
                    if len(frame_scores) > 1
                    else 0.0,
                }
                video_clip_scores.append(video_stats)

                logger.info(
                    f"Video {idx + 1} CLIP stats - "
                    f"min: {video_stats['min']:.2f}, "
                    f"max: {video_stats['max']:.2f}, "
                    f"mean: {video_stats['mean']:.2f}, "
                    f"stddev: {video_stats['stddev']:.2f}"
                )

        # Step 4: Calculate overall statistics across all videos
        if video_clip_scores:
            video_mean_scores = [v["mean"] for v in video_clip_scores]
            overall_stats = {
                "average_clip": statistics.mean(video_mean_scores),
                "min_clip": min(video_mean_scores),
                "max_clip": max(video_mean_scores),
                "clip_standard_deviation": statistics.stdev(video_mean_scores)
                if len(video_mean_scores) > 1
                else 0.0,
                "num_videos": len(video_clip_scores),
            }
        else:
            overall_stats = {
                "average_clip": 0.0,
                "min_clip": 0.0,
                "max_clip": 0.0,
                "clip_standard_deviation": 0.0,
                "num_videos": 0,
            }

        logger.info(f"Overall CLIP statistics: {overall_stats}")
        return overall_stats

    def _extract_video_frames(
        self, video_path: str, frame_sample_rate: int = 8
    ) -> list[Image.Image]:
        """Extract frames from video file, matching tt-metal pipeline output format.

        In tt-metal, the video pipeline's run_single_prompt() returns frames directly
        as a list of PIL Images. This method extracts frames from the video file to
        match that format: [frame1, frame2, frame3, ...] where each frame is a PIL Image.

        Uses imageio with ffmpeg plugin instead of OpenCV to avoid
        CUDA/GPU dependencies.

        Args:
            video_path: Path to video file (.mp4)
            frame_sample_rate: Sample every Nth frame to reduce computation (default: 8)
                Set to 1 to extract all frames

        Returns:
            List of PIL Image objects representing video frames
        """
        frames = []

        try:
            # Use lazy iteration to avoid loading entire video into memory
            total_frames = 0
            for frame_idx, frame in enumerate(iio.imiter(video_path)):
                total_frames += 1

                # Sample frames at specified rate (every Nth frame)
                if frame_idx % frame_sample_rate == 0:
                    # imageio returns RGB directly (no BGR conversion needed)
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)

            logger.info(
                f"Extracted {len(frames)} frames (sampled every {frame_sample_rate} frames) "
                f"from {total_frames} total frames"
            )

        except Exception as e:
            logger.error(f"Error extracting frames from video: {e}")

        return frames

    def _check_accuracy(
        self, clip_results: dict, model_name: str, num_prompts: int
    ) -> AccuracyResult:
        """Check accuracy against reference data.

        Uses self.reference_data which is loaded once at the start of the test.

        Args:
            clip_results: CLIP score statistics
            model_name: Model name to check
            num_prompts: Number of prompts used

        Returns:
            AccuracyResult: BASELINE, PASS, or FAIL
        """
        logger.info(
            "Checking accuracy for model: %s, num_prompts: %s",
            model_name,
            num_prompts,
        )
        try:
            if model_name not in self.reference_data:
                logger.warning(
                    "Model '%s' not found in accuracy reference data.",
                    model_name,
                )
                return AccuracyResult.BASELINE

            accuracy_data = self.reference_data[model_name].get("accuracy", {})

            if str(num_prompts) not in accuracy_data:
                logger.warning(
                    "No reference data for %s prompts for model '%s'.",
                    num_prompts,
                    model_name,
                )
                return AccuracyResult.BASELINE

            reference = accuracy_data[str(num_prompts)]
            clip_range = reference.get("clip_valid_range")

            if not clip_range:
                logger.warning(
                    "No CLIP valid range found for model '%s' with %s prompts.",
                    model_name,
                    num_prompts,
                )
                return AccuracyResult.BASELINE

            average_clip = clip_results.get("average_clip", 0.0)

            if clip_range[0] <= average_clip <= clip_range[1]:
                logger.info(
                    "CLIP score %.2f is within valid range [%.2f, %.2f]",
                    average_clip,
                    clip_range[0],
                    clip_range[1],
                )
                return AccuracyResult.PASS
            logger.warning(
                "CLIP score %.2f is outside valid range [%.2f, %.2f]",
                average_clip,
                clip_range[0],
                clip_range[1],
            )
            return AccuracyResult.FAIL

        except Exception as e:
            logger.error("Error checking accuracy: %s", e)
            return AccuracyResult.BASELINE

    def _get_num_inference_steps_from_reference(
        self, model_name: str, default: int
    ) -> int:
        """Get num_inference_steps from accuracy reference.

        Uses self.reference_data which is loaded once at the start of the test.

        Args:
            model_name: Model name to look up
            default: Default value if not found in reference

        Returns:
            int: Number of inference steps from reference, or default
        """
        try:
            if model_name in self.reference_data:
                num_steps = self.reference_data[model_name].get("num_inference_steps")
                if num_steps is not None:
                    return num_steps
        except Exception as e:
            logger.warning(
                f"Could not get num_inference_steps from reference: {e}, using default"
            )
        return default

    def _load_accuracy_reference(self) -> dict:
        """Load accuracy reference data from JSON file."""
        logger.info(
            "Loading accuracy reference from: %s", CONFIG.ACCURACY_REFERENCE_PATH
        )
        try:
            with open(CONFIG.ACCURACY_REFERENCE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"VideoGenerationEvalTest accuracy reference not found: {CONFIG.ACCURACY_REFERENCE_PATH}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"VideoGenerationEvalTest invalid JSON in accuracy reference: {e}"
            )

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import requests
from PIL import Image

from tests.server_tests.base_test import BaseTest
from tests.server_tests.test_cases.server_helper import (
    DEFAULT_AUTHORIZATION,
    SERVER_DEFAULT_URL,
)
from tests.server_tests.test_classes import TestConfig
from utils.sdxl_accuracy_utils.clip_encoder import CLIPEncoder
from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import sdxl_get_prompts

logger = logging.getLogger(__name__)

# Constants
VIDEO_JOB_STATUS_COMPLETED = "completed"
VIDEO_JOB_STATUS_FAILED = "failed"
VIDEO_JOB_STATUS_CANCELLED = "cancelled"
DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 5
DEFAULT_VIDEO_TIMEOUT_SECONDS = 600
ACCURACY_REFERENCE_PATH = "evals/eval_targets/model_accuracy_reference.json"
VIDEO_GENERATION_ENDPOINT = "video/generations"
DATASET_DIR = "tests/server_tests/datasets/videos"


@dataclass
class VideoGenerationEvalsTestRequest:
    model_name: str
    num_prompts: int = 16
    start_from: int = 0
    num_inference_steps: int = 40
    server_url: str | None = None
    frame_sample_rate: int = 8  # Sample every Nth frame for CLIP scoring


class VideoGenerationEvalsTest(BaseTest):
    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self):
        request = self.targets.get("request")
        logger.info("Running VideoGenerationEvalsTest with request: %s", request)
        if not isinstance(request, VideoGenerationEvalsTestRequest):
            return {
                "success": False,
                "error": "VideoGenerationEvalsTestRequest not provided in targets",
            }

        logger.info(
            f"Measuring video generation accuracy for model: {request.model_name}"
        )

        # Load accuracy reference once for the entire test
        self.reference_data = self._load_accuracy_reference()

        # Override num_inference_steps from accuracy reference if available
        num_inference_steps = self._get_num_inference_steps_from_reference(
            request.model_name, request.num_inference_steps
        )
        if num_inference_steps != request.num_inference_steps:
            logger.info(
                f"Overriding num_inference_steps from {request.num_inference_steps} "
                f"to {num_inference_steps} (from accuracy reference)"
            )
            request.num_inference_steps = num_inference_steps

        # Step 1: Get prompts from COCO dataset
        logger.info(f"Step 1: Loading {request.num_prompts} prompts from COCO captions")
        prompts = sdxl_get_prompts(request.start_from, request.num_prompts)

        # Step 2: Generate videos
        logger.info("Step 2: Generating videos")
        videos_info = self._generate_videos(
            prompts=prompts,
            server_url=request.server_url,
            num_inference_steps=request.num_inference_steps,
        )

        # Step 3: Calculate CLIP scores
        logger.info("Step 3: Calculating CLIP scores for video frames")
        clip_results = self._calculate_video_clip_scores(
            videos_info=videos_info,
            prompts=prompts,
            frame_sample_rate=request.frame_sample_rate,
        )

        # Step 4: Calculate accuracy check
        logger.info("Step 4: Checking accuracy against reference")
        accuracy_check = self._check_accuracy(
            clip_results=clip_results,
            model_name=request.model_name,
            num_prompts=request.num_prompts,
        )

        results = {
            "model": request.model_name,
            "num_prompts": request.num_prompts,
            "num_inference_steps": request.num_inference_steps,
            "clip_results": clip_results,
            "accuracy_check": accuracy_check,
        }

        self.eval_results = results

        return {
            "success": True,
            "eval_results": results,
        }

    def _wait_for_server_ready(
        self,
        service_port: int = 8000,
        max_attempts: int = 230,
        retry_delay: int = 10,
    ) -> bool:
        """Wait for server to be ready using simple HTTP health check.

        Args:
            service_port: Port where the server is running.
            max_attempts: Maximum number of retry attempts.
            retry_delay: Seconds to wait between retries.

        Returns:
            bool: True if server is ready, False otherwise.
        """
        logger.info("Waiting for server to be ready...")
        health_url = f"http://localhost:{service_port}/tt-liveness"
        logger.info("Health URL: %s", health_url)

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "alive" and data.get("model_ready"):
                        logger.info(
                            "Server is ready after %s attempt(s)",
                            attempt,
                        )
                        return True
                logger.info(
                    "Server not ready (attempt %s/%s), retrying in %ss...",
                    attempt,
                    max_attempts,
                    retry_delay,
                )
            except requests.exceptions.RequestException as e:
                logger.info(
                    "Health check failed (attempt %s/%s): %s, retrying in %ss...",
                    attempt,
                    max_attempts,
                    e,
                    retry_delay,
                )
            time.sleep(retry_delay)

        logger.error("Server health check failed after %s attempts", max_attempts)
        return False

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

        base_url = server_url or SERVER_DEFAULT_URL
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
                    f"{base_url}/{VIDEO_GENERATION_ENDPOINT}",
                    json=payload,
                    headers=headers,
                    timeout=90,
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
        polling_interval: int = DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS,
        timeout: int = DEFAULT_VIDEO_TIMEOUT_SECONDS,
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
                    f"{base_url}/{VIDEO_GENERATION_ENDPOINT}/{job_id}",
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
                    return self._download_video(base_url, job_id, headers)
                elif status in [VIDEO_JOB_STATUS_FAILED, VIDEO_JOB_STATUS_CANCELLED]:
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
            output_dir = Path(DATASET_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            video_path = output_dir / f"{job_id}.mp4"

            response = requests.get(
                f"{base_url}/{VIDEO_GENERATION_ENDPOINT}/{job_id}/download",
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
            videos_info: List of video information dicts
            prompts: List of prompts corresponding to videos
            frame_sample_rate: Sample every Nth frame (default: 8)

        Returns:
            dict: CLIP score statistics
        """
        logger.info("Calculating CLIP scores for video frames.")

        clip = CLIPEncoder()

        # Step 1: Extract all videos as list of frame lists
        # Structure: videos = [[frame1, frame2, ...], [frame1, frame2, ...], ...]
        videos = []
        for idx, video_info in enumerate(videos_info):
            video_path = video_info["video_path"]
            logger.info(
                f"Extracting frames from video {idx + 1}/{len(videos_info)}: {video_path}"
            )

            frames = self._extract_video_frames(video_path, frame_sample_rate)

            if not frames:
                logger.warning(f"No frames extracted from video: {video_path}")
                continue

            videos.append(frames)
            logger.info(f"Video {idx + 1}: extracted {len(frames)} frames")

        if not videos:
            logger.error("No videos with valid frames found")
            return {
                "average_clip": 0.0,
                "min_clip": 0.0,
                "max_clip": 0.0,
                "clip_standard_deviation": 0.0,
                "num_videos": 0,
            }

        # Step 2: Calculate CLIP scores
        logger.info("Calculating CLIP scores for all frames...")
        clip_scores = [
            [100 * clip.get_clip_score(prompts[i], img).item() for img in video]
            for i, video in enumerate(videos)
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

        Args:
            video_path: Path to video file (.mp4)
            frame_sample_rate: Sample every Nth frame to reduce computation (default: 8)
            Set to 1 to extract all frames

        Returns:
            List of PIL Image objects representing video frames
        """
        frames = []

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return frames

            total_frames = 0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1

                # Sample frames at specified rate (every Nth frame)
                if frame_idx % frame_sample_rate == 0:
                    # Convert BGR (cv2 format) to RGB (PIL format)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image (matching tt-metal pipeline output)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)

                frame_idx += 1

            cap.release()
            logger.info(
                f"Extracted {len(frames)} frames (sampled every {frame_sample_rate} frames) "
                f"from {total_frames} total frames"
            )

        except Exception as e:
            logger.error(f"Error extracting frames from video: {e}")

        return frames

    def _check_accuracy(
        self, clip_results: dict, model_name: str, num_prompts: int
    ) -> int:
        """Check accuracy against reference data.

        Uses self.reference_data which is loaded once at the start of the test.

        Args:
            clip_results: CLIP score statistics
            model_name: Model name to check
            num_prompts: Number of prompts used

        Returns:
            int: Accuracy check status (0=unknown, 1=baseline, 2=pass, 3=fail)
        """
        logger.info(
            f"Checking accuracy for model: {model_name}, num_prompts: {num_prompts}"
        )

        try:
            if model_name not in self.reference_data:
                logger.warning(
                    f"⚠️ Model '{model_name}' not found in accuracy reference data."
                )
                return 1  # baseline

            accuracy_data = self.reference_data[model_name].get("accuracy", {})

            if str(num_prompts) not in accuracy_data:
                logger.warning(
                    f"⚠️ No reference data for {num_prompts} prompts for model '{model_name}'."
                )
                return 1  # baseline

            reference = accuracy_data[str(num_prompts)]
            clip_range = reference.get("clip_valid_range")

            if not clip_range:
                logger.warning(
                    f"⚠️ No CLIP valid range found for model '{model_name}' with {num_prompts} prompts."
                )
                return 1  # baseline

            average_clip = clip_results.get("average_clip", 0.0)

            # Check if within valid range
            if clip_range[0] <= average_clip <= clip_range[1]:
                logger.info(
                    f"✅ CLIP score {average_clip:.2f} is within valid range [{clip_range[0]:.2f}, {clip_range[1]:.2f}]"
                )
                return 2  # pass
            else:
                logger.warning(
                    f"⚠️ CLIP score {average_clip:.2f} is outside valid range [{clip_range[0]:.2f}, {clip_range[1]:.2f}]"
                )
                return 3  # fail

        except Exception as e:
            logger.error(f"Error checking accuracy: {e}")
            return 1  # baseline

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
        logger.info(f"Loading accuracy reference from: {ACCURACY_REFERENCE_PATH}")
        try:
            with open(ACCURACY_REFERENCE_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Accuracy reference file not found: {ACCURACY_REFERENCE_PATH}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in accuracy reference file: {e}")

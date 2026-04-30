# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.utils import get_num_calls

from ..context import MediaContext, common_report_metadata, require_health
from ..test_status import VideoGenerationTestStatus

logger = logging.getLogger(__name__)


DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 5
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
VIDEO_INFERENCE_STEPS = {"mochi-1-preview": 50, "Wan2.2-T2V-A14B-Diffusers": 40}
VIDEO_JOB_STATUS_COMPLETED = "completed"
VIDEO_JOB_STATUS_FAILED = "failed"
VIDEO_JOB_STATUS_CANCELLED = "cancelled"


def _download_video(ctx: MediaContext, job_id: str, headers: dict) -> str:
    logger.info(f"Downloading video for job: {job_id}")
    try:
        output_dir = Path("/tmp/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"{job_id}.mp4"

        response = requests.get(
            f"{ctx.base_url}/v1/videos/generations/{job_id}/download",
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


def _poll_video_completion(
    ctx: MediaContext,
    job_id: str,
    headers: dict,
    polling_interval: int = DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS,
    timeout: int = DEFAULT_VIDEO_TIMEOUT_SECONDS,
) -> str:
    logger.info(f"Polling video job: {job_id}")
    logger.info(f"Polling interval: {polling_interval} seconds")
    logger.info(f"Timeout: {timeout} seconds")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{ctx.base_url}/v1/videos/generations/{job_id}",
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
                return _download_video(ctx, job_id, headers)
            if status in (VIDEO_JOB_STATUS_FAILED, VIDEO_JOB_STATUS_CANCELLED):
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


def _generate_video(
    ctx: MediaContext, prompt: str, num_inference_steps: int = 20
) -> tuple[bool, float, str, str]:
    logger.info(f"🎬 Generating video with prompt: {prompt}")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "num_inference_steps": num_inference_steps}
    logger.info(f"Payload: {payload}")

    start_time = time.time()
    try:
        response = requests.post(
            f"{ctx.base_url}/v1/videos/generations",
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

        video_path = _poll_video_completion(ctx, job_id, headers)
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


def _run_video_generation_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[VideoGenerationTestStatus]:
    logger.info("Running video generation benchmark.")
    inference_steps = VIDEO_INFERENCE_STEPS[ctx.model_spec.model_name]
    logger.info(f"Inference steps: {inference_steps}")

    status_list: list[VideoGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Generating video {i + 1}/{num_calls}...")
        status, elapsed, job_id, video_path = _generate_video(
            ctx,
            prompt=f"Test video generation {i + 1}",
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
                prompt=f"Test video generation {i + 1}",
            )
        )
    return status_list


def _video_ttft(status_list: list[VideoGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def run_video_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_video_generation_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _video_ttft(status_list)
    report_data = common_report_metadata(ctx, "video")
    report_data["benchmarks"] = {
        "num_requests": len(status_list),
        "num_inference_steps": status_list[0].num_inference_steps if status_list else 0,
        "ttft": ttft_value,
        "inference_steps_per_second": (
            sum(s.inference_steps_per_second for s in status_list) / len(status_list)
            if status_list
            else 0
        ),
    }

    return report_data


__all__ = ["run_video_benchmark"]

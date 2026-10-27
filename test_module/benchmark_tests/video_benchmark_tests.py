# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflow_module.context_helpers import get_num_calls

from report_module.schema import Block
from report_module.status import TestStatus

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    SkipTest,
    VIDEO_GENERATION_ENDPOINT,
    _load_fixture_image_base64,
    block_id,
    build_video_generation_payload,
    get_video_generation_submit_endpoint,
    is_i2v_video_model,
    run_tiered_check,
)
from .._test_common.target_check import (
    performance_target_shapes,
    summary_from_tiered,
)
from .._test_common.video_output_check import (
    check_video_output,
    deployment_resolution,
    expected_video_shape,
    save_clip,
)
from ..context import MediaContext, require_health
from ..test_status import VideoGenerationTestStatus

logger = logging.getLogger(__name__)


# The clip's wall time is only known to one poll interval, so keep it well under
# the latency targets (39 s for a 480p clip).
DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 1
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
# Keyed by full HF repo id.
VIDEO_INFERENCE_STEPS = {
    "genmo/mochi-1-preview": 50,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 40,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": 40,
    # AniSora, Distill and Lightning ignore the client's step count and run a fixed
    # schedule (tt-media-server config/constants.py WAN22_*_NUM_STEPS), so these are
    # what actually runs. LoRA takes the client's count, like the base model. Prodia
    # is left out: its pipeline picks its own count and the runner does not expose it.
    "Wan-AI/Wan2.2-I2V-AniSora-V3.2": 8,
    "Wan-AI/Wan2.2-I2V-Distill-LightX2V": 4,
    "Wan-AI/Wan2.2-I2V-Lightning": 4,
    "Wan-AI/Wan2.2-I2V-LoRA": 40,
}
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
            f"{ctx.base_url}/{VIDEO_GENERATION_ENDPOINT}/{job_id}/download",
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
                f"{ctx.base_url}/{VIDEO_GENERATION_ENDPOINT}/{job_id}",
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
    ctx: MediaContext,
    prompt: str,
    num_inference_steps: int = 20,
    image_b64: str | None = None,
    shape: tuple[int, int] | None = None,
) -> tuple[bool, float, str, str]:
    logger.info(f"🎬 Generating video with prompt: {prompt}")
    model_name = ctx.model_spec.model_name
    submit_endpoint = get_video_generation_submit_endpoint(model_name)
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = build_video_generation_payload(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        model_name=model_name,
        image_b64=image_b64,
        shape=shape,
    )
    # Avoid logging the (large) base64 image prompt for I2V.
    logger.info(f"Payload keys: {sorted(payload)} -> endpoint: {submit_endpoint}")

    start_time = time.time()
    try:
        response = requests.post(
            f"{ctx.base_url}/{submit_endpoint}",
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


def _expected_video_shape(ctx: MediaContext) -> tuple[int, int, int] | None:
    return expected_video_shape(
        ctx.model_spec.hf_model_repo,
        ctx.device.name,
        deployment_resolution(ctx.model_spec),
    )


def _check_video_output(
    ctx: MediaContext, video_path: str, shape: tuple[int, int] | None = None
) -> str | None:
    return check_video_output(
        video_path,
        hf_model_repo=ctx.model_spec.hf_model_repo,
        device_name=ctx.device.name,
        resolution=deployment_resolution(ctx.model_spec),
        shape=shape,
    )


def _benchmark_shapes(ctx: MediaContext) -> list[tuple[int, int]]:
    """Output shapes with their own targets on this device; empty means one run as served."""
    return performance_target_shapes(
        ctx.model_spec.hf_model_repo, ctx.model_spec.cli_args.get("device")
    )


def _run_video_generation_benchmark(
    ctx: MediaContext, num_calls: int, shape: tuple[int, int] | None = None
) -> list[VideoGenerationTestStatus]:
    where = f" at {shape[0]}x{shape[1]}" if shape else ""
    logger.info(f"Running video generation benchmark{where}.")
    model_name = ctx.model_spec.hf_model_repo
    inference_steps = VIDEO_INFERENCE_STEPS[model_name]
    logger.info(f"Inference steps: {inference_steps}")

    # For I2V models load the conditioning image once and reuse it across calls.
    image_b64 = _load_fixture_image_base64() if is_i2v_video_model(model_name) else None

    status_list: list[VideoGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Generating video {i + 1}/{num_calls}...")
        status, elapsed, job_id, video_path = _generate_video(
            ctx,
            prompt=f"Test video generation {i + 1}",
            num_inference_steps=inference_steps,
            image_b64=image_b64,
            shape=shape,
        )
        logger.info(f"Generated video in {elapsed:.2f} seconds.")
        if status:
            problem = _check_video_output(ctx, video_path, shape)
            if problem:
                logger.error(
                    f"❌ Video output check failed for job {job_id}: {problem}"
                )
                status = False
                save_clip(
                    video_path,
                    getattr(ctx, "output_path", None),
                    "failed/benchmarks",
                )

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


def _video_target_checks(
    ctx: MediaContext,
    ttft_seconds: float,
    tput_user: float,
    shape: tuple[int, int] | None = None,
) -> tuple[dict, ReportCheckTypes]:
    # ttft is captured in seconds (time.time() deltas); targets.ttft_ms is ms.
    ttft_ms = ttft_seconds * 1000 if ttft_seconds else None
    # Per-shape fields are suffixed, so one block can grade every shape.
    suffix = f"_{shape[0]}x{shape[1]}" if shape else ""
    logger.info(f"Computing 3-tier target checks for TTFT, tput_user{suffix}")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT",
                ttft_ms,
                "ttft_ms",
                lower_is_better=True,
                field_name=f"ttft{suffix}",
            ),
            MetricSpec(
                "tput_user",
                tput_user,
                "tput_user",
                lower_is_better=False,
                field_name=f"tput_user{suffix}",
            ),
        ],
        shape,
    )


def _summarize(status_list: list[VideoGenerationTestStatus]) -> dict:
    """Counts and timings over the generations that succeeded."""
    successful = [s for s in status_list if s.status]
    return {
        "num_requests": len(status_list),
        "num_successful": len(successful),
        "ttft": _video_ttft(successful),
        "inference_steps_per_second": (
            sum(s.inference_steps_per_second for s in successful) / len(successful)
            if successful
            else 0
        ),
    }


def _run_video_benchmark_by_shape(
    ctx: MediaContext, num_calls: int, shapes: list[tuple[int, int]]
) -> Block:
    """Benchmark each output shape against its own targets, in one block.

    Every shape runs even when the deployment cannot serve it yet: a clip of
    the wrong shape fails that shape's generations, which fails the block.
    """
    records = []
    target_checks: dict[str, dict] = {}
    for shape in shapes:
        status_list = _run_video_generation_benchmark(ctx, num_calls, shape)
        summary = _summarize(status_list)
        checks, check = _video_target_checks(
            ctx, summary["ttft"], summary["inference_steps_per_second"], shape
        )
        for tier, fields in checks.items():
            target_checks.setdefault(tier, {}).update(fields)
        records.append(
            {
                "shape": f"{shape[0]}x{shape[1]}",
                **summary,
                "tput_user": summary["inference_steps_per_second"],
                "target_check": check,
            }
        )

    num_total = sum(r["num_requests"] for r in records)
    num_successful = sum(r["num_successful"] for r in records)
    inference_steps = VIDEO_INFERENCE_STEPS[ctx.model_spec.hf_model_repo]
    block_data: dict = {
        "records": records,
        "num_requests": num_total,
        "num_successful": num_successful,
        "num_inference_steps": inference_steps,
        "target_check": summary_from_tiered(target_checks),
        "target_checks": target_checks,
    }
    if num_successful != num_total:
        logger.error(
            f"Video benchmark: only {num_successful}/{num_total} generations "
            f"succeeded across {len(shapes)} shapes; marking benchmark as FAILED."
        )
        block_data["status"] = TestStatus.FAIL.value
    return Block(
        kind="benchmarks",
        task_type="video",
        title="Video Benchmark",
        id=block_id(ctx) or None,
        targets={
            "num_prompts": num_total,
            "num_inference_steps": inference_steps,
            "shapes": [r["shape"] for r in records],
        },
        data=block_data,
    )


def run_video_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    model_name = ctx.model_spec.hf_model_repo
    if model_name not in VIDEO_INFERENCE_STEPS:
        raise SkipTest(
            f"video benchmark not implemented for model {model_name!r}; "
            f"supported: {sorted(VIDEO_INFERENCE_STEPS)}"
        )
    require_health(ctx)

    shapes = _benchmark_shapes(ctx)
    if shapes:
        return _run_video_benchmark_by_shape(ctx, get_num_calls(ctx), shapes)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_video_generation_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    # Only successful generations produce meaningful timings. A fast HTTP
    # rejection (bad route/auth, server error) returns near-instantly, which
    # would otherwise inflate throughput and deflate TTFT
    successful = [s for s in status_list if s.status]
    num_total = len(status_list)
    num_successful = len(successful)
    all_succeeded = num_total > 0 and num_successful == num_total

    ttft_value = _video_ttft(successful)
    inference_steps_per_second = (
        sum(s.inference_steps_per_second for s in successful) / num_successful
        if successful
        else 0
    )
    # Sequential single-user benchmark, so tput_user = total throughput.
    target_checks, target_check = _video_target_checks(
        ctx, ttft_value, inference_steps_per_second
    )

    block_data: dict = {
        "Benchmarks": {
            "num_requests": num_total,
            "num_successful": num_successful,
            "num_inference_steps": (
                status_list[0].num_inference_steps if status_list else 0
            ),
            "ttft": ttft_value,
            "inference_steps_per_second": inference_steps_per_second,
            "tput_user": inference_steps_per_second,
            "target_check": target_check,
            "target_checks": target_checks,
        },
    }

    # A benchmark where any generation failed is not a valid PASS regardless of
    # the timing of the survivors: surface it as a blocking failure so a broken
    # endpoint cannot be reported as passing.
    if not all_succeeded:
        logger.error(
            f"Video benchmark: only {num_successful}/{num_total} generations "
            f"succeeded; marking benchmark as FAILED."
        )
        block_data["status"] = TestStatus.FAIL.value

    return Block(
        kind="benchmarks",
        task_type="video",
        title="Video Benchmark",
        id=block_id(ctx) or None,
        targets={
            "num_prompts": num_total,
            "num_inference_steps": (
                status_list[0].num_inference_steps if status_list else 0
            ),
        },
        data=block_data,
    )


__all__ = ["run_video_benchmark"]

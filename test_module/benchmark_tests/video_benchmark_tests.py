# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    _load_fixture_image_base64,
    block_id,
    build_video_generation_payload,
    get_video_download_url,
    get_video_generation_submit_endpoint,
    get_video_job_endpoint,
    is_i2v_video_model,
    is_minimax_h3_model,
    run_tiered_check,
)
from .._test_common.server_metrics import (
    scrape_video_metrics,
    server_side_video_metrics,
)
from ..context import MediaContext, require_health
from ..test_status import VideoGenerationTestStatus

logger = logging.getLogger(__name__)


# One second: a 5 s interval adds up to 5 s (about 6 % of a Wan2.2 generation) of
# measurement error, which hides the difference between two servers.
DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS = 1
DEFAULT_VIDEO_TIMEOUT_SECONDS = 1200
# Keyed by full HF repo id.
VIDEO_INFERENCE_STEPS = {
    "genmo/mochi-1-preview": 50,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 40,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": 40,
    # Fixed by the runner; labels the rows only.
    "MiniMaxAI/MiniMax-H3-FL2VA-LightX2V": 4,
    "MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V": 4,
}
# The same request every time, so the runs are comparable between servers and
# between runs. Models whose payload fixes the seed already keep their own.
VIDEO_BENCHMARK_PROMPT = "a red ball rolling on grass"
VIDEO_BENCHMARK_SEED = 42
# The phases of one request, as the client sees them (the same for every server):
#   submit_s      the POST until the 202 answer
#   queue_wait_s  until a poll first sees the job in progress (the polling interval
#                 is the resolution; the server_side block has the exact wait)
#   processing_s  in progress until completed: the device run and the mp4 encode
#   download_s    completed until the mp4 is downloaded
VIDEO_PHASES = ("submit_s", "queue_wait_s", "processing_s", "download_s")
# Videos submitted at once by the burst phase; 0 (the default) turns it off.
# Keep it below the server's queue limit, a full queue answers 429.
VIDEO_BENCHMARK_BURST_JOBS_ENV = "VIDEO_BENCHMARK_BURST_JOBS"
# A shape the boot warmup did not cover compiles inside its first request.
MINIMAX_H3_VIDEO_TIMEOUT_SECONDS = 1800


VIDEO_JOB_STATUS_IN_PROGRESS = "in_progress"
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
            get_video_download_url(
                ctx.base_url, ctx.model_spec.model_name, job_id, headers
            ),
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
    polling_interval: float = DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS,
    timeout: int = DEFAULT_VIDEO_TIMEOUT_SECONDS,
    timings: dict | None = None,
) -> str:
    """Wait for the job and download its video; "" when it failed or timed out.
    ``timings["first_in_progress_at"]`` is set when a poll first sees the job
    `in_progress` and ``timings["completed_at"]`` when it reports `completed`
    (before the download), so the phases of a request can be told apart; both are
    as exact as the polling interval."""
    logger.info(f"Polling video job: {job_id}")
    logger.info(f"Polling interval: {polling_interval} seconds")
    logger.info(f"Timeout: {timeout} seconds")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{ctx.base_url}/"
                f"{get_video_job_endpoint(ctx.model_spec.model_name, job_id)}",
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

            if (
                status == VIDEO_JOB_STATUS_IN_PROGRESS
                and timings is not None
                and "first_in_progress_at" not in timings
            ):
                timings["first_in_progress_at"] = time.time()
            if status == VIDEO_JOB_STATUS_COMPLETED:
                if timings is not None:
                    timings["completed_at"] = time.time()
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


def _request_phases(
    start_time: float, submitted_at: float, poll_times: dict, finished_at: float
) -> dict:
    """The VIDEO_PHASES of one request from its timestamps."""
    completed_at = poll_times["completed_at"]
    started_at = min(poll_times.get("first_in_progress_at", submitted_at), completed_at)
    started_at = max(started_at, submitted_at)
    return {
        "submit_s": submitted_at - start_time,
        "queue_wait_s": started_at - submitted_at,
        "processing_s": completed_at - started_at,
        "download_s": finished_at - completed_at,
    }


def _generate_video(
    ctx: MediaContext,
    prompt: str,
    num_inference_steps: int = 20,
    image_b64: str | None = None,
    timings: dict | None = None,
) -> tuple[bool, float, str, str]:
    """Submit one job and wait for its video. Returns (ok, seconds from the
    submit to the end of the download, job id, video path). When ``timings`` is
    given, ``timings["generation_seconds"]`` is the time until the job reported
    `completed` (without the download)."""
    logger.info(f"🎬 Generating video with prompt: {prompt}")
    model_name = ctx.model_spec.model_name
    h3 = is_minimax_h3_model(model_name)
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
    )
    payload.setdefault("seed", VIDEO_BENCHMARK_SEED)
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

        submitted_at = time.time()
        job_data = response.json()
        job_id = job_data.get("id")
        logger.info(f"Video generation job submitted: {job_id}")

        poll_times: dict = {}
        video_path = _poll_video_completion(
            ctx,
            job_id,
            headers,
            timeout=(
                MINIMAX_H3_VIDEO_TIMEOUT_SECONDS
                if h3
                else DEFAULT_VIDEO_TIMEOUT_SECONDS
            ),
            timings=poll_times,
        )
        finished_at = time.time()
        elapsed = finished_at - start_time
        if timings is not None and "completed_at" in poll_times:
            timings["generation_seconds"] = poll_times["completed_at"] - start_time
            timings.update(
                _request_phases(start_time, submitted_at, poll_times, finished_at)
            )

        if video_path:
            if timings is not None:
                try:
                    timings["video_bytes"] = os.path.getsize(video_path)
                except OSError:
                    pass
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
    model_name = ctx.model_spec.hf_model_repo
    inference_steps = VIDEO_INFERENCE_STEPS[model_name]
    logger.info(f"Inference steps: {inference_steps}")

    # For I2V models load the conditioning image once and reuse it across calls.
    needs_image = is_i2v_video_model(model_name) or is_minimax_h3_model(
        ctx.model_spec.model_name
    )
    image_b64 = _load_fixture_image_base64() if needs_image else None

    status_list: list[VideoGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Generating video {i + 1}/{num_calls}...")
        timings: dict = {}
        status, elapsed, job_id, video_path = _generate_video(
            ctx,
            prompt=VIDEO_BENCHMARK_PROMPT,
            num_inference_steps=inference_steps,
            image_b64=image_b64,
            timings=timings,
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
                prompt=VIDEO_BENCHMARK_PROMPT,
                generation_seconds=timings.get("generation_seconds"),
                phases={k: timings[k] for k in VIDEO_PHASES if k in timings} or None,
                video_bytes=timings.get("video_bytes"),
            )
        )
    return status_list


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _run_video_burst_benchmark(ctx: MediaContext, num_jobs: int) -> dict:
    """Submit ``num_jobs`` videos at once and report how the server queues them:
    the videos per minute over the whole burst and the latency of each video
    (submit to download), which includes the wait in the queue."""
    logger.info(f"Running video burst benchmark with {num_jobs} jobs.")
    model_name = ctx.model_spec.hf_model_repo
    inference_steps = VIDEO_INFERENCE_STEPS[model_name]
    needs_image = is_i2v_video_model(model_name) or is_minimax_h3_model(
        ctx.model_spec.model_name
    )
    image_b64 = _load_fixture_image_base64() if needs_image else None

    start = time.time()
    with ThreadPoolExecutor(max_workers=num_jobs) as pool:
        futures = [
            pool.submit(
                _generate_video,
                ctx,
                VIDEO_BENCHMARK_PROMPT,
                inference_steps,
                image_b64,
            )
            for _ in range(num_jobs)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.time() - start

    latencies = [elapsed for ok, elapsed, _job, _path in results if ok]
    summary = {
        "burst_jobs": num_jobs,
        "burst_succeeded": len(latencies),
        "burst_wall_seconds": wall_seconds,
        "burst_videos_per_minute": (
            60 * len(latencies) / wall_seconds if wall_seconds > 0 else 0
        ),
    }
    if latencies:
        summary["burst_latency_p50"] = _percentile(latencies, 0.5)
        summary["burst_latency_p95"] = _percentile(latencies, 0.95)
    return summary


def _burst_jobs_from_env() -> int:
    raw = os.environ.get(VIDEO_BENCHMARK_BURST_JOBS_ENV, "")
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        logger.warning(f"Ignoring {VIDEO_BENCHMARK_BURST_JOBS_ENV}={raw!r}")
        return 0


def _video_end_to_end_latency(status_list: list[VideoGenerationTestStatus]) -> float:
    """Mean seconds of a whole request: from the POST until the mp4 is downloaded."""
    logger.info("Calculating the end to end latency")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def _video_target_checks(
    ctx: MediaContext, latency_seconds: float, tput_user: float
) -> tuple[dict, ReportCheckTypes]:
    # The latency is captured in seconds (time.time() deltas); the target is in ms
    # (end_to_end_latency_ms of the reference file).
    latency_ms = latency_seconds * 1000 if latency_seconds else None
    logger.info("Computing 3-tier target checks for end_to_end_latency, tput_user")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "end_to_end_latency",
                latency_ms,
                "e2el_ms",
                lower_is_better=True,
                field_name="end_to_end_latency",
            ),
            MetricSpec(
                "tput_user",
                tput_user,
                "tput_user",
                lower_is_better=False,
                field_name="tput_user",
            ),
        ],
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

    burst_jobs = _burst_jobs_from_env()
    try:
        num_calls = get_num_calls(ctx)
        # The server's own view of each phase (empty for a server without /metrics).
        metrics_start = scrape_video_metrics(ctx.base_url)
        status_list = _run_video_generation_benchmark(ctx, num_calls)
        metrics_sequential = scrape_video_metrics(ctx.base_url)
        burst = _run_video_burst_benchmark(ctx, burst_jobs) if burst_jobs else {}
        metrics_end = scrape_video_metrics(ctx.base_url) if burst_jobs else {}
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

    latency_value = _video_end_to_end_latency(successful)
    inference_steps_per_second = (
        sum(s.inference_steps_per_second for s in successful) / num_successful
        if successful
        else 0
    )
    # Sequential single-user benchmark, so tput_user = total throughput.
    target_checks, target_check = _video_target_checks(
        ctx, latency_value, inference_steps_per_second
    )

    generation_times = [
        s.generation_seconds for s in successful if s.generation_seconds is not None
    ]
    block_data: dict = {
        "Benchmarks": {
            "num_requests": num_total,
            "num_successful": num_successful,
            "num_inference_steps": (
                status_list[0].num_inference_steps if status_list else 0
            ),
            "end_to_end_latency_s": latency_value,
            "inference_steps_per_second": inference_steps_per_second,
            "tput_user": inference_steps_per_second,
            "target_check": target_check,
            "target_checks": target_checks,
        },
    }
    phase_values = [s.phases for s in successful if s.phases]
    if generation_times:
        # Until the job reported `completed`: the latency above also holds the download.
        block_data["Benchmarks"]["generation_seconds"] = sum(generation_times) / len(
            generation_times
        )
    video_sizes = [s.video_bytes for s in successful if s.video_bytes is not None]
    if video_sizes:
        block_data["Benchmarks"]["video_bytes"] = sum(video_sizes) / len(video_sizes)
    if phase_values:
        # The same keys for every server: the mean over the requests, and each request.
        block_data["Benchmarks"]["phases"] = {
            key: sum(p[key] for p in phase_values) / len(phase_values)
            for key in VIDEO_PHASES
        }
        block_data["Benchmarks"]["phases_per_request"] = phase_values
    if burst:
        block_data["Benchmarks"].update(burst)
    server_side = server_side_video_metrics(metrics_start, metrics_sequential)
    if server_side:
        block_data["Benchmarks"]["server_side"] = server_side
    burst_server_side = server_side_video_metrics(metrics_sequential, metrics_end)
    if burst_server_side:
        block_data["Benchmarks"]["burst_server_side"] = burst_server_side

    # A benchmark where any generation failed is not a valid PASS regardless of
    # the timing of the survivors: surface it as a blocking failure so a broken
    # endpoint cannot be reported as passing.
    if not all_succeeded:
        logger.error(
            f"Video benchmark: only {num_successful}/{num_total} generations "
            f"succeeded; marking benchmark as FAILED."
        )
        block_data["status"] = TestStatus.FAIL.value
    if burst and burst["burst_succeeded"] != burst["burst_jobs"]:
        logger.error(
            f"Video burst: only {burst['burst_succeeded']}/{burst['burst_jobs']} "
            f"generations succeeded; marking benchmark as FAILED."
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

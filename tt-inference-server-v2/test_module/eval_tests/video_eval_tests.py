# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.workflow_types import ReportCheckTypes

from ..context import MediaContext, common_eval_metadata, require_health

logger = logging.getLogger(__name__)


def _run_video_generation_eval(ctx: MediaContext) -> dict:
    """Delegate to VideoGenerationEvalsTest."""
    from server_tests.test_classes import TestConfig

    from .video_generation_eval_test import (
        VideoGenerationEvalsTest,
        VideoGenerationEvalsTestRequest,
    )

    logger.info("Running video generation eval.")

    num_prompts = 5
    num_inference_steps = 40
    start_from = 0
    frame_sample_rate = 8

    request = VideoGenerationEvalsTestRequest(
        model_name=ctx.model_spec.model_name,
        num_prompts=num_prompts,
        start_from=start_from,
        num_inference_steps=num_inference_steps,
        server_url=ctx.base_url,
        frame_sample_rate=frame_sample_rate,
    )
    logger.info(f"Running VideoGenerationEvalsTest with request: {request}")

    config = TestConfig.create_default()
    test = VideoGenerationEvalsTest(config, {"request": request})

    logger.info("Starting VideoGenerationEvalsTest")
    result = test.run_tests()

    eval_results = result.get("result", {}).get("eval_results", {})
    logger.info(f"VideoGenerationEvalsTest eval_results: {eval_results}")
    return eval_results


def _run_video_fvd_and_fvmd_eval() -> dict:
    """Run FVD + FVMD eval against reference and generated video directories."""
    from server_tests.test_classes import TestConfig

    from .video_fvd_eval_test import DATASET_DIR as FVD_DATASET_DIR
    from .video_fvd_eval_test import VideoFVDTest, VideoFVDTestRequest
    from .video_fvmd_eval_test import VideoFVMDTest, VideoFVMDTestRequest

    logger.info("Running video FVD and FVMD eval.")

    reference_videos_path = str(Path(FVD_DATASET_DIR) / "videos")
    generated_videos_path = str(Path("server_tests/datasets/videos").resolve())
    if not Path(generated_videos_path).exists():
        generated_videos_path = "/tmp/videos"
    logger.info(
        f"Reference path: {reference_videos_path}, generated path: {generated_videos_path}"
    )

    config = TestConfig.create_default()
    combined_results: dict = {
        "reference_videos_path": reference_videos_path,
        "generated_videos_path": generated_videos_path,
    }

    download_request = VideoFVDTestRequest(
        action="download", download_count=2, category="Sports"
    )
    download_test = VideoFVDTest(config, {"request": download_request})
    download_result = download_test.run_tests()
    if not download_result.get("success") or not download_result.get("result", {}).get(
        "success"
    ):
        logger.warning(
            "Reference video download failed: %s. "
            "FVD/FVMD may fail if reference dir is empty.",
            download_result,
        )
    else:
        logger.info("Reference videos downloaded successfully.")

    fvd_request = VideoFVDTestRequest(
        action="compute_fvd",
        reference_videos_path=reference_videos_path,
        generated_videos_path=generated_videos_path,
    )
    fvd_test = VideoFVDTest(config, {"request": fvd_request})
    fvd_result = fvd_test.run_tests()
    fvd_ok = fvd_result.get("success") and fvd_result.get("result", {}).get("success")
    if fvd_ok:
        combined_results["fvd"] = fvd_result["result"].get("fvd_score")
        logger.info("FVD score: %s", combined_results["fvd"])
    else:
        combined_results["fvd"] = None
        logger.warning("FVD test failed or did not return score: %s", fvd_result)

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


def run_video_eval(ctx: MediaContext) -> dict:
    """Run evaluations for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        eval_result = _run_video_generation_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = common_eval_metadata(ctx, "video")

    if eval_result:
        logger.info("Adding eval results from video generation test to benchmark data")
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
        fvd_and_fvmd_result = _run_video_fvd_and_fvmd_eval()
        if fvd_and_fvmd_result:
            benchmark_data["fvd"] = fvd_and_fvmd_result.get("fvd", 0)
            benchmark_data["fvmd"] = fvd_and_fvmd_result.get("fvmd", 0)
    except Exception as e:
        logger.error(f"Error running video FVD and FVMD eval: {e}")

    benchmark_data["published_score"] = ctx.all_params.tasks[0].score.published_score
    benchmark_data["published_score_ref"] = ctx.all_params.tasks[
        0
    ].score.published_score_ref

    return benchmark_data


__all__ = ["run_video_eval"]

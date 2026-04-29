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
from workflows.workflow_types import ReportCheckTypes

from ..context import MediaContext, common_eval_metadata, require_health
from ..test_status import CnnGenerationTestStatus

logger = logging.getLogger(__name__)


CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"


def _cnn_ttft(status_list: list[CnnGenerationTestStatus]) -> float:
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def _analyze_image(ctx: MediaContext) -> tuple[bool, float]:
    logger.info("🔍 Analyzing image")
    with open(f"{ctx.test_payloads_path}/image_client_image_payload", "r") as f:
        image_payload = f.read()

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {"prompt": image_payload}
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/cnn/search-image",
        json=payload,
        headers=headers,
        timeout=90,
    )
    elapsed = time.time() - start_time
    return (response.status_code == 200), elapsed


def _run_image_analysis_benchmark(
    ctx: MediaContext, num_calls: int
) -> list[CnnGenerationTestStatus]:
    logger.info("Running image analysis benchmark.")
    status_list: list[CnnGenerationTestStatus] = []
    for i in range(num_calls):
        logger.info(f"Analyzing image {i + 1}/{num_calls}...")
        status, elapsed = _analyze_image(ctx)
        logger.info(f"Analyzed image with 50 steps in {elapsed:.2f} seconds.")
        status_list.append(CnnGenerationTestStatus(status=status, elapsed=elapsed))
    return status_list


def _run_mobilenetv2_eval(ctx: MediaContext) -> dict:
    """Delegate MobileNetV2 accuracy eval to VisionEvalsTest."""
    from server_tests.test_cases.vision_evals_test import (
        VisionEvalsTest,
        VisionEvalsTestRequest,
    )
    from server_tests.test_classes import TestConfig

    logger.info("Running mobilenetv2 eval.")
    request = VisionEvalsTestRequest(
        action="measure_accuracy",
        mode="device",
        models=[CNN_MOBILENETV2_RUNNER],
        server_url=f"{ctx.base_url}/v1/cnn/search-image",
    )
    logger.info(f"Running VisionEvalsTest with request: {request}")

    config = TestConfig.create_default()
    test = VisionEvalsTest(config, {"request": request})

    logger.info("Starting VisionEvalsTest")
    result = test.run_tests()

    eval_results = result.get("result", {}).get("eval_results", {})
    model_results = eval_results.get(CNN_MOBILENETV2_RUNNER, {})
    logger.info(f"VisionEvalsTest model results: {model_results}")

    device_result = model_results.get("device", {})
    device_result["accuracy_status"] = model_results.get(
        "accuracy_status", ReportCheckTypes.NA
    )
    logger.info(f"VisionEvalsTest device eval_results: {device_result}")
    return device_result


def run_cnn_eval(ctx: MediaContext) -> dict:
    """Run evaluations for a CNN model (MobileNetV2, ResNet, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = require_health(ctx)

    try:
        eval_result = None
        status_list: list[CnnGenerationTestStatus] = []
        if runner_in_use == CNN_MOBILENETV2_RUNNER:
            eval_result = _run_mobilenetv2_eval(ctx)
        else:
            num_calls = get_num_calls(ctx)
            status_list = _run_image_analysis_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    benchmark_data = common_eval_metadata(ctx, "cnn")

    if runner_in_use == CNN_MOBILENETV2_RUNNER and eval_result:
        logger.info("Adding eval results from eval spec test to benchmark data")
        benchmark_data["accuracy_check"] = eval_result.get(
            "accuracy_status", ReportCheckTypes.NA
        )
        benchmark_data["correct"] = eval_result["correct"]
        benchmark_data["total"] = eval_result["total"]
        benchmark_data["mismatches_count"] = eval_result["mismatches_count"]
    else:
        logger.info("No eval results from eval spec test to add to benchmark data")
        ttft_value = _cnn_ttft(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")
        benchmark_data["published_score"] = ctx.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = ctx.all_params.tasks[
            0
        ].score.published_score_ref

    return benchmark_data


__all__ = ["run_cnn_eval"]

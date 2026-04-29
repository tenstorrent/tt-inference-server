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
from ..test_status import CnnGenerationTestStatus

logger = logging.getLogger(__name__)


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


def _cnn_ttft(status_list: list[CnnGenerationTestStatus]) -> float:
    logger.info("Calculating TTFT value")
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def run_cnn_benchmark(ctx: MediaContext) -> dict:
    """Run benchmarks for a CNN model (MobileNetV2, ResNet, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    try:
        num_calls = get_num_calls(ctx)
        status_list = _run_image_analysis_benchmark(ctx, num_calls)
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _cnn_ttft(status_list)
    report_data = common_report_metadata(ctx, "cnn")
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


__all__ = ["run_cnn_benchmark"]

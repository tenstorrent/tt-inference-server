# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import base64
import json
import logging
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflows.workflow_types import ReportCheckTypes

from ..context import MediaContext, common_eval_metadata, require_health
from ..test_status import CnnGenerationTestStatus

logger = logging.getLogger(__name__)


CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"
# Reuse the ImageNet subset prepared by VisionEvalsTest so benchmarks and
# accuracy evals exercise the model with the exact same inputs.
IMAGENET_DATASET_DIR = "server_tests/datasets/imagenet_subset"
IMAGENET_METADATA_FILE = "metadata.json"
# Number of images to fetch when the ImageNet subset is missing on disk. Once
# downloaded, the benchmark sends one request per image found in the dataset
# (so the request count equals len(metadata), not this constant).
DEFAULT_DATASET_DOWNLOAD_COUNT = 20


def _ensure_imagenet_dataset() -> tuple[Path, list[dict]]:
    """Ensure the ImageNet subset is available locally and return its path
    plus the loaded metadata.

    If the dataset directory or metadata is missing this triggers a fresh
    download via VisionEvalsTest so we share a single source of truth with
    the benchmark flow.
    """
    dataset_path = Path(IMAGENET_DATASET_DIR)
    metadata_path = dataset_path / IMAGENET_METADATA_FILE

    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata:
                return dataset_path, metadata
            logger.warning(
                "ImageNet metadata at %s is empty; re-downloading.", metadata_path
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to read existing ImageNet metadata at %s: %s; will re-download.",
                metadata_path,
                e,
            )

    logger.info(
        "ImageNet subset missing at %s; downloading %s samples.",
        dataset_path,
        DEFAULT_DATASET_DOWNLOAD_COUNT,
    )

    # Lazy import to avoid loading 'datasets' library at module import time
    from server_tests.test_classes import TestConfig

    from .vision_evals_test import VisionEvalsTest, VisionEvalsTestRequest

    config = TestConfig.create_default()
    request = VisionEvalsTestRequest(
        action="download", download_count=DEFAULT_DATASET_DOWNLOAD_COUNT
    )
    download_test = VisionEvalsTest(config, {"request": request})
    download_result = download_test.run_tests()
    if not download_result.get("success"):
        raise RuntimeError(f"Failed to download ImageNet samples: {download_result}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not metadata:
        raise RuntimeError(
            f"ImageNet metadata at {metadata_path} is empty after download."
        )
    return dataset_path, metadata


def _analyze_image(ctx: MediaContext, image_path: Path) -> tuple[bool, float]:
    """Analyze a single image using the CNN model and return (ok, elapsed)."""
    logger.info("🔍 Analyzing image: %s", image_path)
    with image_path.open("rb") as img_fp:
        encoded = base64.b64encode(img_fp.read()).decode("ascii")
    image_payload = f"data:image/jpeg;base64,{encoded}"

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


def _run_image_analysis_benchmark(ctx: MediaContext) -> list[CnnGenerationTestStatus]:
    """Run image analysis over the ImageNet subset (one request per image)."""
    dataset_path, metadata = _ensure_imagenet_dataset()
    total_requests = len(metadata)
    logger.info(
        "Running image analysis benchmark over ImageNet subset at %s "
        "(%d images -> %d requests).",
        dataset_path,
        total_requests,
        total_requests,
    )

    status_list: list[CnnGenerationTestStatus] = []
    for i, sample in enumerate(metadata, start=1):
        image_file = dataset_path / sample["filename"]
        logger.info(f"Analyzing image {i}/{total_requests}: {sample['filename']}")
        status, elapsed = _analyze_image(ctx, image_file)
        logger.info(f"Analyzed image in {elapsed:.2f} seconds.")
        status_list.append(CnnGenerationTestStatus(status=status, elapsed=elapsed))

    logger.info(
        "Completed image analysis benchmark: %d requests sent.", len(status_list)
    )
    return status_list


def _cnn_ttft(status_list: list[CnnGenerationTestStatus]) -> float:
    return sum(s.elapsed for s in status_list) / len(status_list) if status_list else 0


def _run_mobilenetv2_eval(ctx: MediaContext) -> dict:
    """Delegate MobileNetV2 accuracy eval to VisionEvalsTest."""
    from server_tests.test_classes import TestConfig

    from .vision_evals_test import VisionEvalsTest, VisionEvalsTestRequest

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
            status_list = _run_image_analysis_benchmark(ctx)
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

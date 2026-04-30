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

from ..context import MediaContext, common_report_metadata, require_health
from ..test_status import CnnGenerationTestStatus

logger = logging.getLogger(__name__)


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
    the eval flow. Once the dataset exists we use whatever images are
    present - the number of benchmark requests is implied by
    ``len(metadata)``.
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

    from ..eval_tests.vision_evals_test import (
        VisionEvalsTest,
        VisionEvalsTestRequest,
    )

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
    """Run image analysis benchmark using the ImageNet subset dataset.

    Sends one request per image present in the dataset (so the number of
    requests is determined by the dataset itself, not by a benchmark
    parameter). This reuses the same dataset that VisionEvalsTest uses
    for accuracy measurements; here we only measure inference timing per
    request - no accuracy comparison. Aggregate metrics such as TTFT are
    computed downstream from ``len(status_list)``.
    """
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
        "Completed image analysis benchmark: %d requests sent.",
        len(status_list),
    )
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
        status_list = _run_image_analysis_benchmark(ctx)
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

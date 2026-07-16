# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from .._test_common import ReportCheckTypes, block_id
from ..context import HardwareRequirement, MediaContext, require_health
from ..test_status import CnnGenerationTestStatus

logger = logging.getLogger(__name__)


CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"
CNN_YOLOX_NANO_RUNNER = "tt-xla-yolox-nano"

_VISION_STATUS_TO_CHECK: dict[int, ReportCheckTypes] = {
    0: ReportCheckTypes.NA,
    2: ReportCheckTypes.PASS,
    3: ReportCheckTypes.FAIL,
}
# Reuse the ImageNet subset prepared by VisionEvalsTest so benchmarks and
# accuracy evals exercise the model with the exact same inputs.
IMAGENET_DATASET_DIR = "server_tests/datasets/imagenet_subset"
IMAGENET_METADATA_FILE = "metadata.json"
# Number of images to fetch when the ImageNet subset is missing on disk. Once
# downloaded, the benchmark sends one request per image found in the dataset
# (so the request count equals len(metadata), not this constant).
DEFAULT_DATASET_DOWNLOAD_COUNT = 20

# --- YOLOX COCO mAP eval ----------------------------------------------------
COCO_HF_DATASET = "detection-datasets/coco"
COCO_HF_SPLIT = "val"
# Number of COCO val images to evaluate.
DEFAULT_COCO_NUM_IMAGES = 100
# Detection thresholds requested from the server for mAP
# (standard COCO eval uses ~0.01).
COCO_EVAL_MIN_CONFIDENCE = 1.0  # percent -> server score_thr ~0.01
COCO_EVAL_TOP_K = 300


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
    from .._test_common import TestConfig

    from .vision_evals_test import VisionEvalsTest, VisionEvalsTestRequest

    config = TestConfig.create_default()
    request = VisionEvalsTestRequest(
        action="download", download_count=DEFAULT_DATASET_DOWNLOAD_COUNT
    )
    download_test = VisionEvalsTest(config, {"request": request})
    download_result = download_test.run_tests()
    if not download_result.data.get("success"):
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
    from .._test_common import TestConfig

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

    eval_results = result.data.get("eval_results", {})
    model_results = eval_results.get(CNN_MOBILENETV2_RUNNER, {})
    logger.info(f"VisionEvalsTest model results: {model_results}")

    device_result = model_results.get("device", {})
    raw_status = model_results.get("accuracy_status")
    device_result["accuracy_status"] = _VISION_STATUS_TO_CHECK.get(
        raw_status, ReportCheckTypes.NA
    )
    logger.info(f"VisionEvalsTest device eval_results: {device_result}")
    return device_result


def run_cnn_eval(ctx: MediaContext) -> Block:
    """Run evaluations for a CNN model (MobileNetV2, ResNet, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )

    runner_in_use = require_health(ctx, HardwareRequirement.ANY_CHIP)

    try:
        eval_result = None
        status_list: list[CnnGenerationTestStatus] = []
        if runner_in_use == CNN_MOBILENETV2_RUNNER:
            eval_result = _run_mobilenetv2_eval(ctx)
        elif runner_in_use == CNN_YOLOX_NANO_RUNNER:
            eval_result = _run_yolox_coco_eval(ctx)
        else:
            status_list = _run_image_analysis_benchmark(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    logger.info("Generating eval report...")
    task = ctx.all_params.tasks[0]
    data: dict = {
        "task_name": task.task_name,
        "tolerance": task.score.tolerance,
    }

    if runner_in_use == CNN_MOBILENETV2_RUNNER and eval_result:
        logger.info("Adding eval results from eval spec test to benchmark data")
        data["accuracy_check"] = eval_result.get("accuracy_status", ReportCheckTypes.NA)
        data["correct"] = eval_result["correct"]
        data["total"] = eval_result["total"]
        data["mismatches_count"] = eval_result["mismatches_count"]
    elif runner_in_use == CNN_YOLOX_NANO_RUNNER and eval_result:
        logger.info("Adding YOLOX COCO mAP eval results")
        data["accuracy_check"] = eval_result.get("accuracy_status", ReportCheckTypes.NA)
        # Report mAP (as a percentage) as the eval score against the published
        # COCO mAP (25.8 for yolox_nano).
        data["score"] = eval_result["mAP_percent"]
        data["mAP"] = eval_result["mAP_percent"]
        data["num_images"] = eval_result["num_images"]
        data["published_score"] = task.score.published_score
        data["published_score_ref"] = task.score.published_score_ref
        # For single-concurrency CNN inference, tput_user (images/sec/user) =
        # 1/latency; emitted so the enforced tput_check reads real throughput
        # instead of defaulting to 0.
        latency_for_perf_check = eval_result.get("mean_latency")
        if latency_for_perf_check:
            data["tput_user"] = 1.0 / latency_for_perf_check
    else:
        logger.info("No eval results from eval spec test to add to benchmark data")
        ttft_value = _cnn_ttft(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")
        data["published_score"] = task.score.published_score
        data["score"] = ttft_value
        data["published_score_ref"] = task.score.published_score_ref
        # Non-MobileNet CNNs only get a timing benchmark here, no accuracy grade,
        # so accuracy is Not Applicable (non-blocking).
        data["accuracy_check"] = ReportCheckTypes.NA

    return Block(
        kind="evals",
        task_type="cnn",
        title="CNN Eval",
        id=block_id(ctx) or None,
        targets={
            "task_name": ctx.all_params.tasks[0].task_name,
            "tolerance": ctx.all_params.tasks[0].score.tolerance,
            "published_score": ctx.all_params.tasks[0].score.published_score,
            "published_score_ref": ctx.all_params.tasks[0].score.published_score_ref,
        },
        data=data,
    )


# --- YOLOX COCO mAP eval ----------------------------------------------------


def _yolox_detect(ctx: MediaContext, image_jpeg: bytes) -> tuple[list, float]:
    """POST one JPEG image and return (detections, elapsed_seconds).

    Each detection is ``(box_xyxy, score, cls_ind)``. Requests a low confidence
    threshold + high top_k so low-confidence detections are retained for mAP.
    """
    encoded = base64.b64encode(image_jpeg).decode("ascii")
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer your-secret-key",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": f"data:image/jpeg;base64,{encoded}",
        "top_k": COCO_EVAL_TOP_K,
        "min_confidence": COCO_EVAL_MIN_CONFIDENCE,
        "response_format": "json",
    }
    start_time = time.time()
    response = requests.post(
        f"{ctx.base_url}/v1/cnn/search-image",
        json=payload,
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - start_time

    detections: list = []
    if response.status_code != 200:
        logger.warning("YOLOX COCO eval: server returned HTTP %s", response.status_code)
        return detections, elapsed
    try:
        body = response.json()
    except ValueError:
        return detections, elapsed

    image_data = body.get("image_data") if isinstance(body, dict) else None
    first = image_data[0] if isinstance(image_data, list) and image_data else image_data
    if not isinstance(first, dict):
        return detections, elapsed
    output = first.get("output") or {}
    boxes = output.get("boxes") or []
    scores = output.get("scores") or []
    indices = output.get("indices") or []
    for box, score, cls_ind in zip(boxes, scores, indices):
        detections.append((box, float(score), int(cls_ind)))
    return detections, elapsed


def _pil_to_jpeg_bytes(image) -> bytes:
    """Encode a PIL image to JPEG bytes (RGB)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _corner_to_height_width_format(bbox) -> Optional[list]:
    """Convert an ``[x1,y1,x2,y2]`` box to ``[x,y,w,h]`` floats.

    Returns None for a malformed/None box so the caller can skip it (the source
    data or server response occasionally has missing coordinates).
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(c is None for c in bbox):
        return None
    try:
        x1, y1, x2, y2 = (float(c) for c in bbox)
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _run_yolox_coco_eval(ctx: MediaContext) -> dict:
    """Evaluate the served YOLOX model with COCO mAP over a val2017 subset.

    Orchestrates the two steps: collect GT + server predictions
    (:func:`_collect_yolox_coco_predictions`), then score them
    (:func:`_score_yolox_coco_predictions`) and derive an accuracy status.

    Robust-but-strict: a bad per-sample response is logged and skipped rather
    than crashing the evals workflow, but if nothing could be scored at all
    (server returned no boxes, empty GT, or COCOeval error) the result is FAIL
    — a serving/plumbing regression must not pass silently.

    Returns ``accuracy_status`` / ``mAP_percent`` / ``num_images`` /
    ``mean_latency``.
    """
    task = ctx.all_params.tasks[0]
    published = task.score.published_score
    tolerance = task.score.tolerance
    num_images = int(os.environ.get("YOLOX_COCO_NUM_IMAGES", DEFAULT_COCO_NUM_IMAGES))

    images_meta, annotations, predictions, mean_latency = (
        _collect_yolox_coco_predictions(ctx, num_images)
    )
    mAP, scored = _score_yolox_coco_predictions(images_meta, annotations, predictions)

    mAP_percent = mAP * 100.0
    if scored and published:
        accuracy_status = ReportCheckTypes.from_result(
            mAP_percent >= published * (1.0 - tolerance)
        )
    elif scored and not published:
        # No reference score configured — cannot judge the mAP.
        accuracy_status = ReportCheckTypes.NA
    else:
        # Could not compute a real mAP: the server returned no detection boxes,
        # ground truth was empty, or COCOeval errored. For a release gate this
        # is a FAIL, not NA — a serving/plumbing regression must not pass
        # silently.
        logger.error(
            "YOLOX COCO eval: could not score mAP (preds=%d, anns=%d); failing "
            "accuracy_check (is the server returning boxes?)",
            len(predictions),
            len(annotations),
        )
        accuracy_status = ReportCheckTypes.FAIL

    logger.info(
        "YOLOX COCO eval: mAP=%.4f (%.2f%%) images=%d preds=%d anns=%d "
        "published=%s -> accuracy_status=%s",
        mAP,
        mAP_percent,
        len(images_meta),
        len(predictions),
        len(annotations),
        published,
        accuracy_status,
    )
    return {
        "accuracy_status": accuracy_status,
        "mAP_percent": mAP_percent,
        "num_images": len(images_meta),
        "mean_latency": mean_latency,
    }


def _collect_yolox_coco_predictions(
    ctx: MediaContext, num_images: int
) -> tuple[list, list, list, float]:
    """Stream a COCO val subset, run inference per image, and collect GT +
    server predictions in COCO format.

    Streams ``detection-datasets/coco`` (val; ``objects`` is a dict-of-lists
    with xyxy ``bbox`` and 0..79 ``category``). Category ids 0..79 are shared
    between GT and predictions (canonical COCO order). A bad sample is logged
    and skipped rather than aborting the run.

    Returns ``(images_meta, annotations, predictions, mean_latency)`` where the
    first three are COCO-format lists and ``mean_latency`` is the mean per-image
    request latency in seconds.
    """
    # Lazy import: heavy dep only needed for this eval.
    from datasets import load_dataset

    images_meta: list = []
    annotations: list = []
    predictions: list = []
    latencies: list = []
    ann_id = 1
    logged_pred = False

    try:
        logger.info(
            "YOLOX COCO eval: streaming %s[%s], %d images",
            COCO_HF_DATASET,
            COCO_HF_SPLIT,
            num_images,
        )
        ds = load_dataset(COCO_HF_DATASET, split=COCO_HF_SPLIT, streaming=True)
        for i, sample in enumerate(itertools.islice(ds, num_images), start=1):
            try:
                image_id = int(sample["image_id"])
                image = sample["image"]
                if not hasattr(image, "width"):
                    # Undecoded {bytes, path} dict — decode to PIL.
                    from PIL import Image as _PILImage

                    image = _PILImage.open(io.BytesIO(image["bytes"]))
                width = int(sample.get("width") or image.width)
                height = int(sample.get("height") or image.height)
                images_meta.append({"id": image_id, "width": width, "height": height})

                objects = sample.get("objects") or {}
                categories = objects.get("category") or []
                bboxes = objects.get("bbox") or []
                for cat, bbox in zip(categories, bboxes):
                    xywh = _corner_to_height_width_format(bbox)
                    if xywh is None or cat is None:
                        continue
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(cat),
                            "bbox": xywh,
                            "area": float(xywh[2] * xywh[3]),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

                dets, elapsed = _yolox_detect(ctx, _pil_to_jpeg_bytes(image))
                latencies.append(elapsed)
                if not logged_pred:
                    logger.info(
                        "YOLOX COCO eval: first image returned %d detections "
                        "(sample=%s)",
                        len(dets),
                        dets[0] if dets else None,
                    )
                    logged_pred = True
                for box, score, cls_ind in dets:
                    xywh = _corner_to_height_width_format(box)
                    if xywh is None:
                        continue
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(cls_ind),
                            "bbox": xywh,
                            "score": float(score),
                        }
                    )
            except Exception as e:  # noqa: BLE001 - skip a bad sample, keep going
                logger.warning("YOLOX COCO eval: skipping sample %d: %s", i, e)
            if i % 10 == 0:
                logger.info("YOLOX COCO eval: %d images processed", i)
    except Exception:
        logger.exception("YOLOX COCO eval: dataset/inference loop failed")

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return images_meta, annotations, predictions, mean_latency


def _score_yolox_coco_predictions(
    images_meta: list, annotations: list, predictions: list
) -> tuple[float, bool]:
    """Score predictions against ground truth with ``pycocotools`` COCOeval.

    Returns ``(mAP, scored)`` where ``mAP`` is mAP@[.5:.95] (>= 0) and
    ``scored`` is False when there was nothing to score (no predictions or no
    GT) or COCOeval errored.
    """
    # Lazy imports: heavy deps only needed for this eval.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not (predictions and annotations):
        logger.warning(
            "YOLOX COCO eval: nothing to score (predictions=%d, annotations=%d) "
            "- is the server returning detection boxes?",
            len(predictions),
            len(annotations),
        )
        return 0.0, False

    try:
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images_meta,
            "annotations": annotations,
            "categories": [{"id": c, "name": str(c)} for c in range(80)],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = [m["id"] for m in images_meta]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return max(float(coco_eval.stats[0]), 0.0), True
    except Exception:
        logger.exception("YOLOX COCO eval: COCOeval failed; mAP=0.0")
        return 0.0, False


__all__ = ["run_cnn_eval"]

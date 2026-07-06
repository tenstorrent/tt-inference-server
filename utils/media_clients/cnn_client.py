# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import io
import itertools
import json
import logging
import os
import time
from pathlib import Path

import requests

from utils.media_clients.test_status import CnnGenerationTestStatus
from workflows.workflow_types import ReportCheckTypes

from .base_strategy_interface import BaseMediaStrategy, PerfCheck
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"
CNN_YOLOX_NANO_RUNNER = "tt-xla-yolox-nano"
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
# Benchmark (latency/throughput timing only) reuses COCO images for YOLOX so it
# doesn't also pull the ImageNet subset. Materialized once from the same HF
# dataset the eval streams (parquet already cached).
COCO_BENCHMARK_DATASET_DIR = "server_tests/datasets/coco_bench"


class CnnClientStrategy(BaseMediaStrategy):
    """Strategy for cnn models (RESNET, etc)."""

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        status_list = []

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            runner_in_use = self.require_health()
            eval_result = None
            if runner_in_use == CNN_MOBILENETV2_RUNNER:
                eval_result = self._run_mobilenetv2_eval()
            elif runner_in_use == CNN_YOLOX_NANO_RUNNER:
                eval_result = self._run_yolox_coco_eval()
            else:
                status_list = self._run_image_analysis_benchmark()
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        benchmark_data = {}

        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name.lower()
        benchmark_data["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        benchmark_data["task_type"] = "cnn"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance

        latency_for_perf_check: Optional[float] = None

        if runner_in_use == CNN_MOBILENETV2_RUNNER and eval_result:
            logger.info("Adding eval results from eval spec test to benchmark data")
            benchmark_data["accuracy_check"] = eval_result.get(
                "accuracy_status", ReportCheckTypes.NA
            )
            benchmark_data["correct"] = eval_result["correct"]
            benchmark_data["total"] = eval_result["total"]
            benchmark_data["mismatches_count"] = eval_result["mismatches_count"]
        elif runner_in_use == CNN_YOLOX_NANO_RUNNER and eval_result:
            logger.info("Adding YOLOX COCO mAP eval results")
            benchmark_data["accuracy_check"] = eval_result.get(
                "accuracy_status", ReportCheckTypes.NA
            )
            # Report mAP (as a percentage) as the eval score against the
            # published COCO mAP (25.8 for yolox_nano).
            benchmark_data["score"] = eval_result["mAP_percent"]
            benchmark_data["mAP"] = eval_result["mAP_percent"]
            benchmark_data["num_images"] = eval_result["num_images"]
            benchmark_data["published_score"] = self.all_params.tasks[
                0
            ].score.published_score
            benchmark_data["published_score_ref"] = self.all_params.tasks[
                0
            ].score.published_score_ref
            latency_for_perf_check = eval_result.get("mean_latency")

            # For single-concurrency CNN inference, tput_user (images/sec/user) = 1/latency.
            if latency_for_perf_check:
                benchmark_data["tput_user"] = 1.0 / latency_for_perf_check

        else:
            logger.info("No eval results from eval spec test to add to benchmark data")
            latency_value = self._calculate_latency(status_list)
            logger.info(f"Extracted latency value (s): {latency_value}")
            latency_for_perf_check = latency_value

            benchmark_data["published_score"] = self.all_params.tasks[
                0
            ].score.published_score
            benchmark_data["score"] = latency_value
            benchmark_data["published_score_ref"] = self.all_params.tasks[
                0
            ].score.published_score_ref

        benchmark_data["performance_check"] = self._calculate_performance_check(
            latency_value=latency_for_perf_check,
        )

        # Make benchmark_data is inside of list as an object
        benchmark_data = [benchmark_data]

        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmark(self) -> None:
        """Run benchmarks for the model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            self.require_health()
            if self._is_yolox():
                self._ensure_coco_benchmark_images()
            else:
                self._ensure_imagenet_dataset()
            loop_start = time.monotonic()
            status_list = self._run_image_analysis_benchmark()
            wall_clock_seconds = time.monotonic() - loop_start

            self._generate_report(status_list, wall_clock_seconds)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _is_yolox(self) -> bool:
        """True for YOLOX detection models (they eval/benchmark on COCO)."""
        return "yolox" in (self.model_spec.hf_model_repo or "").lower()

    def _ensure_coco_benchmark_images(
        self, count: int = DEFAULT_DATASET_DOWNLOAD_COUNT
    ) -> tuple[Path, list[dict]]:
        """Materialize a small COCO image set for benchmark timing.

        The benchmark only needs images to POST for latency/throughput timing
        (it ignores labels), so YOLOX reuses COCO here instead of pulling the
        ImageNet subset. Streamed from the same HF dataset the eval uses (so the
        parquet is already cached); images are saved once and reused. Returns
        ``(dataset_path, metadata)`` matching ``_ensure_imagenet_dataset``.
        """
        dataset_path = Path(COCO_BENCHMARK_DATASET_DIR)
        metadata_path = dataset_path / IMAGENET_METADATA_FILE
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if metadata:
                    return dataset_path, metadata
            except (json.JSONDecodeError, OSError):
                pass

        logger.info(
            "COCO benchmark images missing at %s; materializing %d images.",
            dataset_path,
            count,
        )
        # Lazy import to avoid loading 'datasets' at module import time.
        from datasets import load_dataset

        ds = load_dataset(COCO_HF_DATASET, split=COCO_HF_SPLIT, streaming=True)
        dataset_path.mkdir(parents=True, exist_ok=True)
        metadata: list[dict] = []
        for i, sample in enumerate(itertools.islice(ds, count)):
            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            filename = f"coco_{i:03d}.jpg"
            image.save(dataset_path / filename)
            metadata.append({"filename": filename})

        if not metadata:
            raise RuntimeError("No COCO benchmark images could be materialized.")
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f)
        return dataset_path, metadata

    def _run_image_analysis_benchmark(self) -> list[CnnGenerationTestStatus]:
        """Run image analysis benchmark using an image subset dataset.

        Sends one request per image present in the dataset (so the number of
        requests is determined by the dataset itself, not by a benchmark
        parameter). This reuses the same dataset that VisionEvalsTest uses
        for accuracy measurements; here we only measure inference timing per
        request - no accuracy comparison. Aggregate metrics such as TTFT are
        computed downstream from `len(status_list)`.
        """
        # The benchmark only needs images to time inference (accuracy-agnostic).
        # YOLOX reuses COCO so it doesn't also download the ImageNet subset;
        # other CNNs keep using the shared ImageNet subset.
        if self._is_yolox():
            dataset_path, metadata = self._ensure_coco_benchmark_images()
        else:
            dataset_path, metadata = self._ensure_imagenet_dataset()
        total_requests = len(metadata)
        logger.info(
            "Running image analysis benchmark over %s (%d images -> %d requests).",
            dataset_path,
            total_requests,
            total_requests,
        )

        status_list: list[CnnGenerationTestStatus] = []
        for i, sample in enumerate(metadata, start=1):
            image_file = dataset_path / sample["filename"]
            logger.info(f"Analyzing image {i}/{total_requests}: {sample['filename']}")
            status, elapsed = self._analyze_image(image_file)
            logger.info(f"Analyzed image in {elapsed:.2f} seconds.")
            status_list.append(
                CnnGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                )
            )

        logger.info(
            "Completed image analysis benchmark: %d requests sent.",
            len(status_list),
        )
        return status_list

    def _ensure_imagenet_dataset(self) -> tuple[Path, list[dict]]:
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
                    "ImageNet metadata at %s is empty; re-downloading.",
                    metadata_path,
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to read existing ImageNet metadata at %s: %s; "
                    "will re-download.",
                    metadata_path,
                    e,
                )

        logger.info(
            "ImageNet subset missing at %s; downloading %s samples.",
            dataset_path,
            DEFAULT_DATASET_DOWNLOAD_COUNT,
        )

        # Lazy import to avoid loading 'datasets' library at module import time
        from server_tests.test_cases.vision_evals_test import (
            VisionEvalsTest,
            VisionEvalsTestRequest,
        )
        from server_tests.test_classes import TestConfig

        config = TestConfig.create_default()
        request = VisionEvalsTestRequest(
            action="download", download_count=DEFAULT_DATASET_DOWNLOAD_COUNT
        )
        download_test = VisionEvalsTest(config, {"request": request})
        download_result = download_test.run_tests()
        if not download_result.get("success"):
            raise RuntimeError(
                f"Failed to download ImageNet samples: {download_result}"
            )

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not metadata:
            raise RuntimeError(
                f"ImageNet metadata at {metadata_path} is empty after download."
            )
        return dataset_path, metadata

    def _analyze_image(self, image_path: Path) -> tuple[bool, float]:
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
            f"{self.base_url}/v1/cnn/search-image",
            json=payload,
            headers=headers,
            timeout=90,
        )
        elapsed = time.time() - start_time

        return (response.status_code == 200), elapsed

    def _generate_report(
        self,
        status_list: list[CnnGenerationTestStatus],
        wall_clock_seconds: Optional[float] = None,
    ) -> None:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        latency_value = self._calculate_latency(status_list)
        performance_check = self._calculate_performance_check(
            latency_value=latency_value
        )
        tail = self._calculate_tail_latencies([s.elapsed for s in status_list])
        throughput_rps = self._calculate_throughput_rps(
            len(status_list), wall_clock_seconds
        )

        # CNN inference is single-shot, not iterative, so step-based fields
        # (``num_inference_steps`` / ``inference_steps_per_second``) do not
        # apply and are intentionally omitted from the CNN benchmark JSON.
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "latency": latency_value,
                "throughput_rps": throughput_rps,
                **tail,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name.lower(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "cnn",
            "performance_check": performance_check,
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _calculate_latency(self, status_list: list[CnnGenerationTestStatus]) -> float:
        """Mean end-to-end request latency in seconds."""
        logger.info("Calculating latency")

        return (
            sum(status.elapsed for status in status_list) / len(status_list)
            if status_list
            else 0
        )

    def _calculate_performance_check(
        self,
        latency_value: Optional[float] = None,
    ) -> ReportCheckTypes:
        """CNN perf check: compares latency vs configured target.

        Targets file stores latency in ms; converted at this boundary so the
        helper can compare same-unit values.
        """
        targets = self.get_performance_targets()
        logger.info(f"Performance targets: {targets}")
        latency_target_s = (
            targets.ttft_ms / 1000.0 if targets.ttft_ms is not None else None
        )
        return self.calculate_performance_check(
            checks=[
                PerfCheck(
                    "latency", latency_value, latency_target_s, lower_is_better=True
                ),
            ],
            tolerance=targets.tolerance,
        )

    def _run_mobilenetv2_eval(self) -> dict:
        """Run mobilenetv2 eval.

        Returns:
            dict: eval_results with structure:
                {
                    "tt-xla-mobilenetv2": {
                        "accuracy": 0.36,
                        "correct": 36,
                        "total": 100,
                        "mismatches_count": 64
                    }
                }
        """
        # Lazy import to avoid loading 'datasets' library at module import time
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
            server_url=f"{self.base_url}/v1/cnn/search-image",
        )
        logger.info(f"Running VisionEvalsTest with request: {request}")

        config = TestConfig.create_default()
        targets = {"request": request}
        test = VisionEvalsTest(config, targets)

        logger.info("Starting VisionEvalsTest")
        result = test.run_tests()

        # Extract eval_results from nested structure: {model: {cpu: {...}, device: {...}, accuracy_status: int}}
        eval_results = result.get("result", {}).get("eval_results", {})
        model_results = eval_results.get(CNN_MOBILENETV2_RUNNER, {})
        logger.info(f"VisionEvalsTest model results: {model_results}")

        # Get device mode results for benchmark comparison
        device_result = model_results.get("device", {})
        device_result["accuracy_status"] = model_results.get(
            "accuracy_status", ReportCheckTypes.NA
        )
        logger.info(f"VisionEvalsTest device eval_results: {device_result}")

        return device_result

    # --- YOLOX COCO mAP eval ------------------------------------------------

    def _yolox_detect(self, image_jpeg: bytes) -> tuple[list, float]:
        """POST one JPEG image and return (detections, elapsed_seconds).

        Each detection is ``(box_xyxy, score, cls_ind)``. Requests a low
        confidence threshold + high top_k so low-confidence detections are
        retained for mAP.
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
            f"{self.base_url}/v1/cnn/search-image",
            json=payload,
            headers=headers,
            timeout=120,
        )
        elapsed = time.time() - start_time

        detections: list = []
        if response.status_code != 200:
            logger.warning(
                "YOLOX COCO eval: server returned HTTP %s", response.status_code
            )
            return detections, elapsed
        try:
            body = response.json()
        except ValueError:
            return detections, elapsed

        image_data = body.get("image_data") if isinstance(body, dict) else None
        first = (
            image_data[0] if isinstance(image_data, list) and image_data else image_data
        )
        if not isinstance(first, dict):
            return detections, elapsed
        output = first.get("output") or {}
        boxes = output.get("boxes") or []
        scores = output.get("scores") or []
        indices = output.get("indices") or []
        for box, score, cls_ind in zip(boxes, scores, indices):
            detections.append((box, float(score), int(cls_ind)))
        return detections, elapsed

    @staticmethod
    def _pil_to_jpeg_bytes(image) -> bytes:
        """Encode a PIL image to JPEG bytes (RGB)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return buf.getvalue()

    @staticmethod
    def _corner_to_height_width_format(bbox) -> Optional[list]:
        """Convert an ``[x1,y1,x2,y2]`` box to ``[x,y,w,h]`` floats.

        Returns None for a malformed/None box so the caller can skip it (the
        source data or server response occasionally has missing coordinates).
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

    def _run_yolox_coco_eval(self) -> dict:
        """Evaluate the served YOLOX model with COCO mAP over a val2017 subset.

        Orchestrates the two steps: collect GT + server predictions
        (:meth:`_collect_yolox_coco_predictions`), then score them
        (:meth:`_score_yolox_coco_predictions`) and derive an accuracy status.

        Robust-but-strict: a bad per-sample response is logged and skipped
        rather than crashing the evals workflow, but if nothing could be scored
        at all (server returned no boxes, empty GT, or COCOeval error) the
        result is FAIL — a serving/plumbing regression must not pass silently.

        Returns ``accuracy_status`` / ``mAP_percent`` / ``num_images`` /
        ``mean_latency``.
        """
        task = self.all_params.tasks[0]
        published = task.score.published_score
        tolerance = task.score.tolerance
        num_images = int(
            os.environ.get("YOLOX_COCO_NUM_IMAGES", DEFAULT_COCO_NUM_IMAGES)
        )

        images_meta, annotations, predictions, mean_latency = (
            self._collect_yolox_coco_predictions(num_images)
        )
        mAP, scored = self._score_yolox_coco_predictions(
            images_meta, annotations, predictions
        )

        mAP_percent = mAP * 100.0
        if scored and published:
            accuracy_status = ReportCheckTypes.from_result(
                mAP_percent >= published * (1.0 - tolerance)
            )
        elif scored and not published:
            # No reference score configured — cannot judge the mAP.
            accuracy_status = ReportCheckTypes.NA
        else:
            # Could not compute a real mAP: the server returned no detection
            # boxes, ground truth was empty, or COCOeval errored. For a release
            # gate this is a FAIL, not NA — a serving/plumbing regression must
            # not pass silently.
            logger.error(
                "YOLOX COCO eval: could not score mAP (preds=%d, anns=%d); "
                "failing accuracy_check (is the server returning boxes?)",
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
        self, num_images: int
    ) -> tuple[list, list, list, float]:
        """Stream a COCO val subset, run inference per image, and collect GT +
        server predictions in COCO format.

        Streams ``detection-datasets/coco`` (val; ``objects`` is a dict-of-lists
        with xyxy ``bbox`` and 0..79 ``category``). Category ids 0..79 are shared
        between GT and predictions (canonical COCO order). A bad sample is logged
        and skipped rather than aborting the run.

        Returns ``(images_meta, annotations, predictions, mean_latency)`` where
        the first three are COCO-format lists and ``mean_latency`` is the mean
        per-image request latency in seconds.
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
                    images_meta.append(
                        {"id": image_id, "width": width, "height": height}
                    )

                    objects = sample.get("objects") or {}
                    categories = objects.get("category") or []
                    bboxes = objects.get("bbox") or []
                    for cat, bbox in zip(categories, bboxes):
                        xywh = self._corner_to_height_width_format(bbox)
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

                    dets, elapsed = self._yolox_detect(self._pil_to_jpeg_bytes(image))
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
                        xywh = self._corner_to_height_width_format(box)
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
        self, images_meta: list, annotations: list, predictions: list
    ) -> tuple[float, bool]:
        """Score predictions against ground truth with ``pycocotools`` COCOeval.

        Returns ``(mAP, scored)`` where ``mAP`` is mAP@[.5:.95] (>= 0) and
        ``scored`` is False when there was nothing to score (no predictions or
        no GT) or COCOeval errored.
        """
        # Lazy imports: heavy deps only needed for this eval.
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        if not (predictions and annotations):
            logger.warning(
                "YOLOX COCO eval: nothing to score (predictions=%d, "
                "annotations=%d) - is the server returning detection boxes?",
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

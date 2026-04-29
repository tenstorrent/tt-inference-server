# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import json
import logging
import time
from pathlib import Path

import requests

from utils.media_clients.test_status import CnnGenerationTestStatus
from workflows.workflow_types import ReportCheckTypes

from .base_strategy_interface import BaseMediaStrategy

logger = logging.getLogger(__name__)

# Constants
CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"
# Reuse the ImageNet subset prepared by VisionEvalsTest so benchmarks and
# accuracy evals exercise the model with the exact same inputs.
IMAGENET_DATASET_DIR = "server_tests/datasets/imagenet_subset"
IMAGENET_METADATA_FILE = "metadata.json"
# Number of images to fetch when the ImageNet subset is missing on disk. Once
# downloaded, the benchmark sends one request per image found in the dataset
# (so the request count equals len(metadata), not this constant).
DEFAULT_DATASET_DOWNLOAD_COUNT = 20


class CnnClientStrategy(BaseMediaStrategy):
    """Strategy for cnn models (RESNET, etc)."""

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        status_list = []

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")
            # 2026-01-11 11:05:48,031 - utils.media_clients.cnn_client - INFO - Runner in use: tt-xla-mobilenetv2

            eval_result = None
            if runner_in_use == CNN_MOBILENETV2_RUNNER:
                eval_result = self._run_mobilenetv2_eval()
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
            # Calculate TTFT
            ttft_value = self._calculate_ttft_value(status_list)
            logger.info(f"Extracted TTFT value: {ttft_value}")

            benchmark_data["published_score"] = self.all_params.tasks[
                0
            ].score.published_score
            benchmark_data["score"] = ttft_value
            benchmark_data["published_score_ref"] = self.all_params.tasks[
                0
            ].score.published_score_ref

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

    def run_benchmark(self, attempt=0) -> None:
        """Run benchmarks for the model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info(f"Health check passed. Runner in use: {runner_in_use}")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            status_list = self._run_image_analysis_benchmark()

            self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_image_analysis_benchmark(self) -> list[CnnGenerationTestStatus]:
        """Run image analysis benchmark using the ImageNet subset dataset.

        Sends one request per image present in the dataset (so the number of
        requests is determined by the dataset itself, not by a benchmark
        parameter). This reuses the same dataset that VisionEvalsTest uses
        for accuracy measurements; here we only measure inference timing per
        request - no accuracy comparison. Aggregate metrics such as TTFT are
        computed downstream from `len(status_list)`.
        """
        dataset_path, metadata = self._ensure_imagenet_dataset()
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
        from test_module.eval_tests.vision_evals_test import (
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

    def _generate_report(self, status_list: list[CnnGenerationTestStatus]) -> None:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)

        # Convert ImageGenerationTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": status_list[0].num_inference_steps
                if status_list
                else 0,
                "ttft": ttft_value,
                "inference_steps_per_second": sum(
                    status.inference_steps_per_second for status in status_list
                )
                / len(status_list)
                if status_list
                else 0,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "cnn",
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    def _calculate_ttft_value(
        self, status_list: list[CnnGenerationTestStatus]
    ) -> float:
        """Calculate TTFT value based on status list."""
        logger.info("Calculating TTFT value")

        return (
            sum(status.elapsed for status in status_list) / len(status_list)
            if status_list
            else 0
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
        from test_module.eval_tests.vision_evals_test import (
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

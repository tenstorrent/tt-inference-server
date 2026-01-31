# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
import time
from pathlib import Path

import requests

from utils.media_clients.test_status import CnnGenerationTestStatus
from workflows.utils import get_num_calls

from .base_strategy_interface import BaseMediaStrategy

logger = logging.getLogger(__name__)

# Constants
CNN_MOBILENETV2_RUNNER = "tt-xla-mobilenetv2"


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

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)
            eval_result = None
            if runner_in_use == CNN_MOBILENETV2_RUNNER:
                eval_result = self._run_mobilenetv2_eval()
            else:
                status_list = self._run_image_analysis_benchmark(num_calls)
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
            benchmark_data["accuracy"] = eval_result.get("accuracy_status", 0)
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

            # Get num_calls from CNN benchmark parameters
            num_calls = get_num_calls(self)

            status_list = []
            status_list = self._run_image_analysis_benchmark(num_calls)

            self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _run_image_analysis_benchmark(
        self, num_calls: int
    ) -> list[CnnGenerationTestStatus]:
        """Run image analysis benchmark."""
        logger.info("Running image analysis benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Analyzing image {i + 1}/{num_calls}...")
            status, elapsed = self._analyze_image()
            logger.info(f"Analyzed image with {50} steps in {elapsed:.2f} seconds.")
            status_list.append(
                CnnGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                )
            )

        return status_list

    def _analyze_image(self) -> tuple[bool, float]:
        """Analyze image using CNN model."""
        logger.info("🔍 Analyzing image")
        with open(f"{self.test_payloads_path}/image_client_image_payload", "r") as f:
            imagePayload = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {"prompt": imagePayload}
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/cnn/search-image",
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
        from tests.server_tests.test_cases.vision_evals_test import (
            VisionEvalsTest,
            VisionEvalsTestRequest,
        )
        from tests.server_tests.test_classes import TestConfig

        logger.info("Running mobilenetv2 eval.")

        request = VisionEvalsTestRequest(
            action="measure_accuracy",
            mode="device",
            models=[CNN_MOBILENETV2_RUNNER],
            server_url=f"{self.base_url}/cnn/search-image",
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
        device_result["accuracy_status"] = model_results.get("accuracy_status", 0)
        logger.info(f"VisionEvalsTest device eval_results: {device_result}")

        return device_result

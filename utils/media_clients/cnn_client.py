# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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

# Runners that VisionEvalsTest can run a real CPU-vs-device accuracy eval against
# (ImageNet subset). Kept in sync with server_tests/test_cases/vision_evals_test.py
# MODELS list. Routing these runners to that eval is what produces the per-model
# accuracy_status used as accuracy_check in the dashboard.
VISION_EVAL_SUPPORTED_RUNNERS = frozenset(
    {
        "tt-xla-resnet",
        "tt-xla-vovnet",
        "tt-xla-mobilenetv2",
        "tt-xla-efficientnet",
        "tt-xla-segformer",
        "tt-xla-vit",
    }
)


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
            if runner_in_use in VISION_EVAL_SUPPORTED_RUNNERS:
                eval_result = self._run_vision_eval(runner_in_use)
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

        if runner_in_use in VISION_EVAL_SUPPORTED_RUNNERS and eval_result:
            logger.info("Adding eval results from eval spec test to benchmark data")
            benchmark_data["accuracy_check"] = eval_result.get("accuracy_status", 0)
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

            # CNN classifiers without a labeled-dataset eval pathway derive
            # accuracy_check from API success rate so acceptance_criteria has
            # a pass/fail signal. Values match ReportCheckTypes (PASS=2, FAIL=3).
            all_ok = bool(status_list) and all(s.status for s in status_list)
            benchmark_data["accuracy_check"] = 2 if all_ok else 3

            # CNN classifiers run a single forward pass, so the LLM-style
            # inference_steps_per_second is always 0. Report throughput as
            # images-per-second derived from the mean per-request latency.
            benchmark_data["tput_user"] = (1.0 / ttft_value) if ttft_value > 0 else 0

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

        # Warmup request: the first call after server startup pays the XLA
        # compilation cost, which inflates TTFT averages well above the
        # steady-state value. Run one request to prime the cache and exclude
        # it from the measurements.
        logger.info("Warmup request to prime XLA cache (excluded from metrics)...")
        warmup_status, warmup_elapsed = self._analyze_image()
        logger.info(
            f"Warmup completed in {warmup_elapsed:.2f}s (status={warmup_status})"
        )

        for i in range(num_calls):
            logger.info(f"Analyzing image {i + 1}/{num_calls}...")
            status, elapsed = self._analyze_image()
            logger.info(f"Analyzed image in {elapsed:.2f} seconds.")
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

        # CNN classifiers report throughput as images/sec derived from TTFT;
        # populate tput_user (and the legacy inference_steps_per_second alias
        # consumed by acceptance_criteria) so downstream reporting is non-zero.
        images_per_second = (1.0 / ttft_value) if ttft_value > 0 else 0

        # Convert ImageGenerationTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": status_list[0].num_inference_steps
                if status_list
                else 0,
                "ttft": ttft_value,
                "tput_user": images_per_second,
                "inference_steps_per_second": images_per_second,
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

    def _run_vision_eval(self, runner_name: str) -> dict:
        """Run the CPU-vs-device accuracy eval (VisionEvalsTest) for a CNN runner.

        Returns:
            dict: device-mode eval results for ``runner_name`` with structure:
                {
                    "accuracy": 0.36,
                    "correct": 36,
                    "total": 100,
                    "mismatches_count": 64,
                    "accuracy_status": <PASS/FAIL int from ReportCheckTypes>,
                }
        """
        # Lazy import to avoid loading 'datasets' library at module import time
        from server_tests.test_cases.vision_evals_test import (
            VisionEvalsTest,
            VisionEvalsTestRequest,
        )
        from server_tests.test_classes import TestConfig

        logger.info(f"Running vision eval for runner: {runner_name}")

        request = VisionEvalsTestRequest(
            action="measure_accuracy",
            mode="device",
            models=[runner_name],
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
        model_results = eval_results.get(runner_name, {})
        logger.info(f"VisionEvalsTest model results: {model_results}")

        # Get device mode results for benchmark comparison
        device_result = model_results.get("device", {})
        device_result["accuracy_status"] = model_results.get("accuracy_status", 0)
        logger.info(f"VisionEvalsTest device eval_results: {device_result}")

        return device_result

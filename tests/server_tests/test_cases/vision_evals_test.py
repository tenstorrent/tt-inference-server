# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
"""Eval test for vision/CNN models (ResNet, MobileNetV2, EfficientNet, etc.)."""

import base64
import itertools
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional, Union

import requests
from datasets import DownloadConfig, Image, load_dataset

from tests.server_tests.base_test import BaseTest
from tests.server_tests.test_cases.server_helper import (
    DEFAULT_AUTHORIZATION,
    SERVER_DEFAULT_URL,
)
from tests.server_tests.test_classes import TestConfig

logger = logging.getLogger(__name__)


class AccuracyResult(IntEnum):
    """Accuracy check result codes."""

    UNDEFINED = 0
    BASELINE = 1
    PASS = 2
    FAIL = 3


@dataclass(frozen=True)
class VisionEvalsConfig:
    """Vision evals configuration."""

    DATASET_DIR: str = "tests/server_tests/datasets/imagenet_subset"
    REQUEST_TIMEOUT_SECONDS: float = 60.0
    ACCURACY_THRESHOLD: float = 0.05
    CPU_ACCURACY_FILE: str = "cpu_accuracy.json"
    DEVICE_ACCURACY_FILE: str = "device_accuracy.json"
    DEFAULT_DOWNLOAD_COUNT: int = 20
    MODELS: tuple = (
        "tt-xla-resnet",
        "tt-xla-vovnet",
        "tt-xla-mobilenetv2",
        "tt-xla-efficientnet",
        "tt-xla-segformer",
        "tt-xla-vit",
    )


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration."""

    MAX_ATTEMPTS: int = 230
    RETRY_DELAY: int = 10
    TIMEOUT: int = 10


CONFIG = VisionEvalsConfig()
HEALTH_CONFIG = HealthCheckConfig()

ACCURACY_FILE_BY_MODE = {
    "cpu": CONFIG.CPU_ACCURACY_FILE,
    "device": CONFIG.DEVICE_ACCURACY_FILE,
}

DOWNLOAD_CLEANUP_DELAY = 5


@dataclass
class VisionEvalsTestRequest:
    action: Literal["download", "measure_accuracy", "compare"]
    mode: Literal["cpu", "device"] = "device"
    models: Optional[list[str]] = None
    server_url: Optional[str] = None
    download_count: int = CONFIG.DEFAULT_DOWNLOAD_COUNT


class VisionEvalsTest(BaseTest):
    """Eval test for vision/CNN models."""

    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self) -> dict:
        """Run the vision evaluation test."""
        request = self._parse_request()
        if isinstance(request, dict):
            return request

        if request.server_url and not request.models:
            return self._error(
                "VisionEvalsTest server_url requires models to be specified"
            )

        target_models = request.models or list(CONFIG.MODELS)

        action_handlers = {
            "download": lambda: self._handle_download(request),
            "measure_accuracy": lambda: self._handle_measure_accuracy(request, target_models),
            "compare": lambda: self._handle_compare(),
        }
        handler = action_handlers.get(request.action)
        if handler is None:
            return self._error(f"Unknown action: {request.action}")
        return handler()

    def _handle_download(self, request: VisionEvalsTestRequest) -> dict:
        """Handle the download action."""
        logger.info(f"Downloading samples: {request.download_count}")
        self._download_samples(count=request.download_count)
        return {
            "success": True,
            "action": "download",
            "count": request.download_count,
        }

    def _handle_measure_accuracy(
        self,
        request: VisionEvalsTestRequest,
        target_models: list[str],
    ) -> dict:
        """Handle the measure_accuracy action."""
        logger.info(f"Measuring accuracy for models: {target_models}")

        logger.info("Step 1: Downloading samples")
        self._download_samples(count=request.download_count)

        logger.info("Step 2: Measuring CPU accuracy")
        self._measure_accuracy(
            models=target_models,
            server_url=request.server_url,
            mode="cpu",
        )

        logger.info("Step 3: Measuring device accuracy")
        self._measure_accuracy(
            models=target_models,
            server_url=request.server_url,
            mode="device",
        )

        logger.info("Step 4: Comparing CPU vs device results")
        accuracy_status = self._compare_results()

        for model, status in accuracy_status.items():
            if model in self.eval_results:
                self.eval_results[model]["accuracy_status"] = status

        return {
            "success": True,
            "action": "measure_accuracy",
            "mode": "full",
            "eval_results": self.eval_results,
        }

    def _handle_compare(self) -> dict:
        """Handle the compare action."""
        self._compare_results()
        return {"success": True, "action": "compare"}

    def _parse_request(self) -> Union[VisionEvalsTestRequest, dict]:
        """Parse and validate request from targets."""
        raw = self.targets.get("request")
        if raw is None:
            return self._error("VisionEvalsTest request not provided in targets")
        if isinstance(raw, VisionEvalsTestRequest):
            return raw
        if isinstance(raw, dict):
            try:
                return VisionEvalsTestRequest(**raw)
            except TypeError as e:
                return self._error(f"VisionEvalsTest invalid request parameters: {e}")
        return self._error(
            "VisionEvalsTest request must be dict or VisionEvalsTestRequest"
        )

    @staticmethod
    def _error(message: str) -> dict:
        """Create error response."""
        return {"success": False, "error": message}

    def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready."""
        health_url = f"http://localhost:{self.service_port}/tt-liveness"
        logger.info(f"Waiting for server: {health_url}")
        for attempt in range(1, HEALTH_CONFIG.MAX_ATTEMPTS + 1):
            if self._check_health(health_url):
                logger.info(f"Server ready after {attempt} attempt(s)")
                return True
            logger.debug(f"Not ready (attempt {attempt}/{HEALTH_CONFIG.MAX_ATTEMPTS})")
            time.sleep(HEALTH_CONFIG.RETRY_DELAY)
        logger.error(f"Server not ready after {HEALTH_CONFIG.MAX_ATTEMPTS} attempts")
        return False

    def _check_health(self, url: str) -> bool:
        """Single health check attempt."""
        try:
            response = requests.get(url, timeout=HEALTH_CONFIG.TIMEOUT)
        except requests.RequestException:
            return False
        if response.status_code != 200:
            return False
        data = response.json()
        return data.get("status") == "alive" and data.get("model_ready", False)

    def _load_metadata(self, dataset_path: Path) -> list[dict]:
        logger.info(f"Loading metadata from {dataset_path}")
        metadata_path = dataset_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            raise ValueError("Metadata must be a list of sample descriptors.")

        return metadata

    def _replay_samples(
        self,
        metadata: list[dict],
        dataset_path: Path,
        server_url: str,
        authorization: Optional[str],
        timeout: float,
    ) -> list[dict]:
        """Replay dataset samples against the server and collect responses."""
        logger.info(f"Replaying samples from {dataset_path} to {server_url}")
        headers = self._build_headers(authorization)
        results: list[dict] = []

        with requests.Session() as session:
            for entry in metadata:
                result = self._process_single_sample(
                    session, entry, dataset_path, server_url, headers, timeout,
                )
                results.append(result)

        return results

    @staticmethod
    def _process_single_sample(
        session: requests.Session,
        entry: dict,
        dataset_path: Path,
        server_url: str,
        headers: dict,
        timeout: float,
    ) -> dict:
        """Send a single image sample to the server and return the result."""
        image_file = dataset_path / entry["filename"]
        if not image_file.exists():
            raise FileNotFoundError(f"Missing image file: {image_file}")

        with image_file.open("rb") as img_fp:
            encoded = base64.b64encode(img_fp.read()).decode("ascii")

        payload = {"prompt": f"data:image/jpeg;base64,{encoded}"}
        response = session.post(
            server_url, headers=headers, json=payload, timeout=timeout,
        )
        response.raise_for_status()
        return {"sample": entry, "response": response.json()}

    @staticmethod
    def _build_headers(authorization: Optional[str] = None) -> dict[str, str]:
        """Build request headers."""
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {authorization or DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_label(raw: Union[str, int, None]) -> str:
        if raw is None:
            return ""
        if isinstance(raw, int):
            return str(raw)

        normalized = str(raw).strip().lower()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        cleaned = [
            char if char.isalnum() or char == "_" else "_" for char in normalized
        ]
        collapsed = "".join(cleaned)
        while "__" in collapsed:
            collapsed = collapsed.replace("__", "_")
        return collapsed.strip("_")

    def _extract_prediction(self, payload: dict) -> tuple[Optional[str], Optional[str]]:
        image_data = payload.get("image_data") if isinstance(payload, dict) else None

        if isinstance(image_data, list) and image_data:
            image_data = image_data[0]

        if not isinstance(image_data, dict):
            return None, None

        label = image_data.get("top1_class_label")
        probability = image_data.get("top1_class_probability")

        if label is None:
            output = image_data.get("output")
            if isinstance(output, dict):
                labels = output.get("labels")
                if isinstance(labels, list) and labels:
                    label = labels[0]

        return label, probability

    def _analyze_results(self, entries: list[dict]) -> tuple[int, int, list[dict]]:
        """Analyze inference results against expected labels.

        Returns:
            (correct_count, total_count, mismatches)
        """
        total = len(entries)
        correct = 0
        mismatches: list[dict] = []

        for entry in entries:
            sample = entry.get("sample", {})
            response_payload = entry.get("response", {})
            predicted_label, probability = self._extract_prediction(response_payload)

            expected_label = sample.get("label") or sample.get("label_id")
            expected_key = self._normalize_label(expected_label)
            predicted_key = self._normalize_label(predicted_label)

            if expected_key and expected_key == predicted_key:
                correct += 1
            else:
                mismatches.append(
                    {
                        "sample": sample,
                        "expected_label": expected_label,
                        "predicted_label": predicted_label,
                        "probability": probability,
                        "normalized_expected": expected_key,
                        "normalized_predicted": predicted_key,
                        "response": response_payload.get("image_data"),
                    }
                )

        return correct, total, mismatches

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _download_samples(self, count: int = CONFIG.DEFAULT_DOWNLOAD_COUNT) -> None:
        """Stream a small ImageNet subset and materialize images plus metadata."""
        logger.info(f"Downloading {count} samples.")
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
            download_config=DownloadConfig(num_proc=1),
        )
        dataset = ds.cast_column("image", Image(decode=True))
        samples = itertools.islice(dataset, count)

        output_path = Path(CONFIG.DATASET_DIR)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        label_feature = dataset.features.get("label")
        label_names = label_feature.names if hasattr(label_feature, "names") else None

        metadata = []
        for idx, sample in enumerate(samples):
            image = sample["image"]
            label_id = sample.get("label")
            label_name = (
                label_names[label_id]
                if label_names and label_id is not None
                else str(label_id)
            )

            safe_label = (label_name or "unknown").replace(" ", "_")
            filename = f"imagenet_{idx:03d}_{safe_label}.jpg"

            image_path = output_path / filename
            image.save(image_path)

            metadata.append(
                {
                    "index": idx,
                    "label_id": label_id,
                    "label": label_name,
                    "filename": filename,
                }
            )

        metadata_path = output_path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved {len(metadata)} ImageNet samples to {output_path} (metadata: {metadata_path})"
        )
        time.sleep(
            DOWNLOAD_CLEANUP_DELAY
        )  # Workaround for streaming dataset cleanup issues

    def _get_accuracy_status(
        self,
        cpu_accuracy: Optional[float],
        device_accuracy: Optional[float],
        threshold: float = CONFIG.ACCURACY_THRESHOLD,
    ) -> tuple[AccuracyResult, Optional[float]]:
        """Evaluate device accuracy against CPU accuracy.

        Returns:
            (AccuracyResult, min_acceptable):
                AccuracyResult: UNDEFINED, PASS, or FAIL
                min_acceptable: CPU * (1 - threshold), or None when undefined
        """
        if cpu_accuracy is None or device_accuracy is None:
            return (AccuracyResult.UNDEFINED, None)

        min_acceptable = cpu_accuracy * (1 - threshold)
        if device_accuracy < min_acceptable:
            return (AccuracyResult.FAIL, min_acceptable)

        return (AccuracyResult.PASS, min_acceptable)

    def _compare_results(self) -> dict[str, AccuracyResult]:
        """Compare CPU and device accuracy results and print a summary.

        Returns:
            Dictionary mapping model names to AccuracyResult values.
        """
        logger.info("Comparing CPU and device accuracy results.")
        cpu_accuracy, device_accuracy = self._load_accuracy_files()
        if cpu_accuracy is None or device_accuracy is None:
            return {}

        accuracy_status, unacceptable = self._evaluate_models(cpu_accuracy, device_accuracy)
        self._log_comparison_table(cpu_accuracy, device_accuracy, accuracy_status)

        if unacceptable:
            formatted = ", ".join(unacceptable)
            logger.error(
                f"Device accuracy is not acceptable: {formatted} fell more than 5% below the CPU baseline."
            )
            sys.exit(1)

        return accuracy_status

    def _load_accuracy_files(self) -> tuple[Optional[dict], Optional[dict]]:
        """Load CPU and device accuracy JSON files.

        Returns:
            (cpu_accuracy, device_accuracy) dicts, or (None, None) if missing.
        """
        dataset_path = Path(CONFIG.DATASET_DIR)
        cpu_path = dataset_path / ACCURACY_FILE_BY_MODE["cpu"]
        device_path = dataset_path / ACCURACY_FILE_BY_MODE["device"]

        if not cpu_path.exists():
            logger.warning(f"CPU accuracy file not found: {cpu_path}")
            return None, None
        if not device_path.exists():
            logger.warning(f"Device accuracy file not found: {device_path}")
            return None, None

        with cpu_path.open("r", encoding="utf-8") as f:
            cpu_accuracy = json.load(f)
        with device_path.open("r", encoding="utf-8") as f:
            device_accuracy = json.load(f)

        return cpu_accuracy, device_accuracy

    def _evaluate_models(
        self,
        cpu_accuracy: dict[str, float],
        device_accuracy: dict[str, float],
    ) -> tuple[dict[str, AccuracyResult], list[str]]:
        """Evaluate each model's device accuracy against CPU baseline.

        Returns:
            (accuracy_status, unacceptable_models)
        """
        accuracy_status: dict[str, AccuracyResult] = {}
        unacceptable: list[str] = []

        for model in sorted(set(cpu_accuracy.keys()).union(device_accuracy.keys())):
            cpu_value = cpu_accuracy.get(model)
            device_value = device_accuracy.get(model)
            status, _ = self._get_accuracy_status(cpu_value, device_value)
            accuracy_status[model] = status

            if status == AccuracyResult.FAIL:
                unacceptable.append(model)

        return accuracy_status, unacceptable

    def _log_comparison_table(
        self,
        cpu_accuracy: dict[str, float],
        device_accuracy: dict[str, float],
        accuracy_status: dict[str, AccuracyResult],
    ) -> None:
        """Log a formatted comparison table of CPU vs device accuracy."""
        logger.info("Accuracy Comparison (CPU vs Device):")
        logger.info(
            f"{'Model':<30} {'CPU Accuracy (%)':>18} {'Device Accuracy (%)':>20} {'Difference (%)':>18} {'Status':>8}"
        )
        logger.info("-" * 100)

        for model in sorted(accuracy_status.keys()):
            cpu_value = cpu_accuracy.get(model)
            device_value = device_accuracy.get(model)
            status = accuracy_status[model]
            _, min_acceptable = self._get_accuracy_status(cpu_value, device_value)

            cpu_pct = cpu_value * 100.0 if cpu_value is not None else None
            device_pct = device_value * 100.0 if device_value is not None else None

            if cpu_pct is not None and device_pct is not None:
                diff_display = f"{device_pct - cpu_pct:18.2f}"
            else:
                diff_display = f"{'N/A':>18}"

            cpu_display = f"{cpu_pct:18.2f}" if cpu_pct is not None else f"{'N/A':>18}"
            device_display = f"{device_pct:20.2f}" if device_pct is not None else f"{'N/A':>20}"
            status_label = "PASS" if status == AccuracyResult.PASS else "FAIL" if status == AccuracyResult.FAIL else "N/A"

            logger.info(f"{model:<30} {cpu_display} {device_display} {diff_display} {status_label:>8}")

            if min_acceptable is not None and status == AccuracyResult.PASS:
                logger.info(f"Device accuracy {device_value:.4f} is above minimum acceptable {min_acceptable:.4f}")
            elif min_acceptable is not None and status == AccuracyResult.FAIL:
                logger.info(f"Device accuracy {device_value:.4f} is below minimum acceptable {min_acceptable:.4f}")

    def _measure_accuracy(
        self,
        models: Optional[list[str]] = None,
        server_url: Optional[str] = None,
        mode: Literal["cpu", "device"] = "cpu",
        authorization: Optional[str] = None,
        timeout: float = CONFIG.REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Measure accuracy for models against an already-running server."""
        logger.info(f"Measuring accuracy for models: {models} in {mode} mode")
        dataset_path = Path(CONFIG.DATASET_DIR)
        metadata = self._load_metadata(dataset_path)
        summary_path = dataset_path / ACCURACY_FILE_BY_MODE[mode]
        accuracy_summary: dict[str, float] = {}

        if not self._wait_for_server_ready():
            raise RuntimeError("Server health check failed - server not ready")

        for model in models:
            logger.info(f"Measuring accuracy for model: {model}")

            results = self._replay_samples(
                metadata=metadata,
                dataset_path=dataset_path,
                server_url=server_url or SERVER_DEFAULT_URL,
                authorization=authorization,
                timeout=timeout,
            )

            correct, total, mismatches = self._analyze_results(results)
            logger.info(
                f"Correct: {correct}, Total: {total}, Mismatches: {len(mismatches)}"
            )

            accuracy = (correct / total) if total else 0.0
            logger.info(f"Accuracy for model {model}: {accuracy * 100:.2f}%")
            accuracy_summary[model] = accuracy

            if model not in self.eval_results:
                self.eval_results[model] = {}

            self.eval_results[model][mode] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "mismatches_count": len(mismatches),
            }

            logger.info(
                f"[{mode.upper()}] {model}: {accuracy * 100:.2f}% accuracy ({correct}/{total} correct)"
            )

            if mismatches:
                mismatch_path = dataset_path / f"{model}_{mode}_mismatches.json"
                self._write_json(mismatch_path, mismatches)
                logger.info(f"Saved {len(mismatches)} mismatches to {mismatch_path}")

        self._write_json(summary_path, accuracy_summary)
        logger.info(f"Saved {mode} accuracy summary to {summary_path}")

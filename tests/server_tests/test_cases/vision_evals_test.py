# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import base64
import itertools
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

import requests
from datasets import DownloadConfig, Image, load_dataset

from tests.server_tests.base_test import BaseTest
from tests.server_tests.test_cases.server_helper import (
    DEFAULT_AUTHORIZATION,
    SERVER_DEFAULT_URL,
    launch_cpu_server,
    launch_device_server,
    stop_server,
    wait_for_server_ready,
)
from tests.server_tests.test_classes import TestConfig

DATASET_DIR = "tests/server_tests/datasets/imagenet_subset"
MODELS = [
    "tt-xla-resnet",
    "tt-xla-vovnet",
    "tt-xla-mobilenetv2",
    "tt-xla-efficientnet",
    "tt-xla-segformer",
    # "tt-xla-unet",
    "tt-xla-vit",
]
REQUEST_TIMEOUT_SECONDS = 60.0
ACCURACY_FILE_BY_MODE = {
    "cpu": "cpu_accuracy.json",
    "device": "device_accuracy.json",
}

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class VisionEvalsTestRequest:
    action: Literal["download", "measure_accuracy", "compare"]
    mode: Literal["cpu", "device"] = "device"  # for measure_accuracy action
    models: list[str] | None = None  # None means all MODELS
    server_url: str | None = None  # optional, for CI
    download_count: int = 20  # for download action


class VisionEvalsTest(BaseTest):
    def __init__(self, config: TestConfig, targets: dict):
        super().__init__(config, targets)
        self.eval_results: dict = {}

    async def _run_specific_test_async(self):
        request = self.targets.get("request")
        logger.info("Running VisionEvalsTest with request: %s", request)
        if not isinstance(request, VisionEvalsTestRequest):
            return {
                "success": False,
                "error": "VisionEvalsTestRequest not provided in targets",
            }

        if request.server_url and not request.models:
            return {
                "success": False,
                "error": "server_url requires models to be specified",
            }

        target_models = request.models or MODELS

        if request.action == "download":
            logger.info("Downloading samples: %s", request.download_count)
            self._download_samples(count=request.download_count)
            return {
                "success": True,
                "action": "download",
                "count": request.download_count,
            }

        elif request.action == "measure_accuracy":
            logger.info("Measuring accuracy for models: %s", target_models)

            # Step 1: Download samples
            logger.info("Step 1: Downloading samples")
            self._download_samples(count=request.download_count)

            # Step 2: Measure CPU accuracy (always launches its own CPU server)
            logger.info("Step 2: Measuring CPU accuracy")
            self._measure_accuracy(
                models=target_models,
                server_url=None,  # CPU mode always launches own server
                mode="cpu",
            )

            # Step 3: Measure device accuracy (uses existing server if provided)
            logger.info("Step 3: Measuring device accuracy")
            self._measure_accuracy(
                models=target_models,
                server_url=request.server_url,  # Use existing server if provided
                mode="device",
            )

            # Step 4: Compare results
            logger.info("Step 4: Comparing CPU vs device results")
            self._compare_results()

            return {
                "success": True,
                "action": "measure_accuracy",
                "mode": "full",
                "eval_results": self.eval_results,
            }

        elif request.action == "compare":
            self._compare_results()
            return {"success": True, "action": "compare"}

        raise ValueError(f"Unknown action: {request.action}")

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
        authorization: str | None,
        timeout: float,
    ) -> list[dict]:
        logger.info(f"Replaying samples from {dataset_path} to {server_url}")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {authorization or DEFAULT_AUTHORIZATION}",
            "Content-Type": "application/json",
        }

        session = requests.Session()
        results: list[dict] = []

        try:
            for entry in metadata:
                image_file = dataset_path / entry["filename"]
                if not image_file.exists():
                    raise FileNotFoundError(f"Missing image file: {image_file}")

                with image_file.open("rb") as img_fp:
                    encoded = base64.b64encode(img_fp.read()).decode("ascii")

                payload = {"prompt": f"data:image/jpeg;base64,{encoded}"}
                response = session.post(
                    server_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                results.append({"sample": entry, "response": body})
        finally:
            session.close()

        return results

    def _normalize_label(self, raw: str | int | None) -> str:
        logger.info("Normalizing label.")
        if raw is None:
            return ""
        if isinstance(raw, int):
            return str(raw)

        normalized = str(raw).strip().lower()
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace(" ", "_")
        cleaned = []
        for char in normalized:
            cleaned.append(char if char.isalnum() or char == "_" else "_")
        collapsed = "".join(cleaned)
        while "__" in collapsed:
            collapsed = collapsed.replace("__", "_")
        return collapsed.strip("_")

    def _extract_prediction(self, payload: dict) -> tuple[str | None, str | None]:
        logger.info("Extracting prediction.")
        image_data = payload.get("image_data") if isinstance(payload, dict) else None

        # Handle list response (image_data is List[ImageClassificationResult])
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
        logger.info("Analyzing results.")
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

    def _write_json(self, path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _download_samples(self, count: int = 20) -> None:
        """Stream a small ImageNet subset and materialize images plus metadata."""
        logger.info(f"Downloading {count} samples.")
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        ds = load_dataset(
            "imagenet-1k",
            split="validation",
            streaming=True,
            download_config=DownloadConfig(num_proc=1),
        )
        dataset = ds.cast_column("image", Image(decode=True))
        samples = itertools.islice(dataset, count)

        output_path = Path(DATASET_DIR)
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
            "Saved %s ImageNet samples to %s (metadata: %s)",
            len(metadata),
            output_path,
            metadata_path,
        )
        time.sleep(5)  # Workaround for streaming dataset cleanup issues

    def _compare_results(self) -> None:
        """Compare CPU and device accuracy results and print a summary."""
        logger.info("Comparing results CPU and device accuracy.")
        dataset_path = Path(DATASET_DIR)
        cpu_accuracy_path = dataset_path / ACCURACY_FILE_BY_MODE["cpu"]
        device_accuracy_path = dataset_path / ACCURACY_FILE_BY_MODE["device"]

        if not cpu_accuracy_path.exists():
            logger.warning("CPU accuracy file not found: %s", cpu_accuracy_path)
            logger.info("Run CPU accuracy measurement with --measure_cpu_accuracy")
            return
        if not device_accuracy_path.exists():
            logger.warning("Device accuracy file not found: %s", device_accuracy_path)
            logger.info(
                "Run device accuracy measurement with --measure_device_accuracy"
            )
            return

        with cpu_accuracy_path.open("r", encoding="utf-8") as f:
            cpu_accuracy = json.load(f)
        with device_accuracy_path.open("r", encoding="utf-8") as f:
            device_accuracy = json.load(f)

        logger.info("Accuracy Comparison (CPU vs Device):")
        logger.info(
            f"{'Model':30} {'CPU Accuracy (%)':>18} {'Device Accuracy (%)':>20} {'Difference (%)':>18}"
        )
        logger.info("-" * 90)

        unacceptable: list[str] = []

        for model in sorted(set(cpu_accuracy.keys()).union(device_accuracy.keys())):
            cpu_value = cpu_accuracy.get(model)
            device_value = device_accuracy.get(model)

            cpu_pct = cpu_value * 100.0 if cpu_value is not None else None
            device_pct = device_value * 100.0 if device_value is not None else None

            if cpu_pct is not None and device_pct is not None:
                diff = device_pct - cpu_pct
                if diff < -5.0:
                    unacceptable.append(model)
                diff_display = f"{diff:18.2f}"
            else:
                diff_display = f"{'N/A':>18}"

            cpu_display = f"{cpu_pct:18.2f}" if cpu_pct is not None else f"{'N/A':>18}"
            device_display = (
                f"{device_pct:20.2f}" if device_pct is not None else f"{'N/A':>20}"
            )

            logger.info(
                "%s %s %s %s", f"{model:30}", cpu_display, device_display, diff_display
            )

        if unacceptable:
            formatted = ", ".join(unacceptable)
            logger.error(
                "Device accuracy is not acceptable: %s fell more than 5%% below the CPU baseline.",
                formatted,
            )
            sys.exit(1)

    def _measure_accuracy(
        self,
        models: list[str] | None = None,
        server_url: str = None,
        mode: Literal["cpu", "device"] = "cpu",
        authorization: str | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        logger.info(f"Measuring accuracy for models: {models} in {mode} mode")
        dataset_path = Path(DATASET_DIR)
        metadata = self._load_metadata(dataset_path)
        summary_path = dataset_path / ACCURACY_FILE_BY_MODE[mode]
        accuracy_summary: dict[str, float] = {}

        for model in models:
            process = None
            log_path = None

            try:
                if mode == "cpu":
                    logger.info("Starting CPU server for model: %s", model)
                    process, log_path = launch_cpu_server(model)
                    wait_for_server_ready(process, log_path=log_path)
                    logger.info("CPU server is ready for model: %s", model)
                elif mode == "device":
                    if server_url:
                        logger.info(
                            "Using existing server at %s for model: %s",
                            server_url,
                            model,
                        )
                    else:
                        logger.info("Starting device server for model: %s", model)
                        process, log_path = launch_device_server(model)
                        wait_for_server_ready(process, log_path=log_path)
                        logger.info("Device server is ready for model: %s", model)

                results = self._replay_samples(
                    metadata=metadata,
                    dataset_path=dataset_path,
                    server_url=server_url or SERVER_DEFAULT_URL,
                    authorization=authorization,
                    timeout=timeout,
                )
            finally:
                # Clean up the server process if we started one
                if process is not None:
                    stop_server(process)

            correct, total, mismatches = self._analyze_results(results)

            accuracy = (correct / total) if total else 0.0
            accuracy_summary[model] = accuracy

            # Store detailed results for this model and mode
            if model not in self.eval_results:
                self.eval_results[model] = {}
            self.eval_results[model][mode] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "mismatches_count": len(mismatches),
            }

            logger.info(
                "[%s] %s: %.2f%% accuracy (%s/%s correct)",
                mode.upper(),
                model,
                accuracy * 100,
                correct,
                total,
            )

            if mismatches:
                mismatch_path = dataset_path / f"{model}_{mode}_mismatches.json"
                self._write_json(mismatch_path, mismatches)
                logger.info(
                    "Saved %s mismatches to %s",
                    len(mismatches),
                    mismatch_path,
                )

        self._write_json(summary_path, accuracy_summary)
        logger.info("Saved %s accuracy summary to %s", mode, summary_path)


"""
Usage:
1. Download the dataset (n images).
2. Measure CPU accuracy for the dataset -> output JSON
    For each model start the server in CPU mode
    Run inference for all images
    Compare the returned label with the one from metadata
    Calculate the percentage of correct labels
    Save to the JSON file cpu_accuracy.json { model: accuracy }
3. Measure device accuracy for the dataset -> output JSON
    There are two invocation variants
        - CI: the server is already running for a given model, we need the server URL and the name of the model under test
        - Local: start the server in device mode for each model

Example commands:
    python tests/server_tests/test_cases/vision_evals_test.py --download 20
    python tests/server_tests/test_cases/vision_evals_test.py --measure_cpu_accuracy
    python tests/server_tests/test_cases/vision_evals_test.py --measure_cpu_accuracy --model tt-xla-resnet
    python tests/server_tests/test_cases/vision_evals_test.py --measure_device_accuracy
    python tests/server_tests/test_cases/vision_evals_test.py --measure_device_accuracy --model tt-xla-resnet --server_url http://127.0.0.1:8000/cnn/search-image
    python tests/server_tests/test_cases/vision_evals_test.py --compare_results
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision evaluation test utility")
    parser.add_argument(
        "--download",
        nargs="?",
        const=20,
        type=int,
        metavar="COUNT",
        help="Download and prepare the dataset (defaults to 20 samples).",
    )
    parser.add_argument(
        "--measure_cpu_accuracy",
        action="store_true",
        help="Measure CPU model accuracy",
    )
    parser.add_argument(
        "--measure_device_accuracy",
        action="store_true",
        help="Measure device model accuracy",
    )
    parser.add_argument(
        "--compare_results",
        action="store_true",
        help="Compare CPU and device accuracy results",
    )
    parser.add_argument(
        "--model",
        help="Specific model runner to compare; defaults to all configured models.",
    )
    parser.add_argument(
        "--server_url", help="Server URL to use for TT device comparisons"
    )
    args = parser.parse_args()

    if args.server_url and not args.model:
        logger.error("When providing a server URL model must be specified (--model)")
        exit(1)

    target_models = [args.model] if args.model else None
    config = TestConfig.create_default()
    request = None

    if args.download is not None:
        request = VisionEvalsTestRequest(
            action="download", download_count=args.download
        )
    elif args.measure_cpu_accuracy or args.measure_device_accuracy:
        request = VisionEvalsTestRequest(
            action="measure_accuracy",
            mode="cpu" if args.measure_cpu_accuracy else "device",
            models=target_models,
            server_url=args.server_url,
        )
    elif args.compare_results:
        request = VisionEvalsTestRequest(action="compare")

    if request:
        test = VisionEvalsTest(config, {"request": request})
        result = test.run_tests()
        if not result.get("success"):
            logger.error("Test failed: %s", result)
            exit(1)
    else:
        parser.print_help()
        exit(1)

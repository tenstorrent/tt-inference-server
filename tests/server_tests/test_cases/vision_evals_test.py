# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import base64
import itertools
import logging
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

from datasets import DownloadConfig, Image, load_dataset
import requests

from server_helper import (
    DEFAULT_AUTHORIZATION,
    SERVER_DEFAULT_URL,
    launch_cpu_server,
    launch_device_server,
    stop_server,
    wait_for_server_ready,
)

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


def _load_metadata(dataset_path: Path) -> list[dict]:
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("Metadata must be a list of sample descriptors.")

    return metadata


def _replay_samples(
    metadata: list[dict],
    dataset_path: Path,
    server_url: str,
    authorization: str | None,
    timeout: float,
) -> list[dict]:
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


def _normalize_label(raw: str | int | None) -> str:
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


def _extract_prediction(payload: dict) -> tuple[str | None, str | None]:
    image_data = payload.get("image_data") if isinstance(payload, dict) else None
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


def _analyze_results(entries: list[dict]) -> tuple[int, int, list[dict]]:
    total = len(entries)
    correct = 0
    mismatches: list[dict] = []

    for entry in entries:
        sample = entry.get("sample", {})
        response_payload = entry.get("response", {})
        predicted_label, probability = _extract_prediction(response_payload)

        expected_label = sample.get("label") or sample.get("label_id")
        expected_key = _normalize_label(expected_label)
        predicted_key = _normalize_label(predicted_label)

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


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def download_samples(count: int = 20) -> None:
    """Stream a small ImageNet subset and materialize images plus metadata."""

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


def compare_results() -> None:
    """Compare CPU and device accuracy results and print a summary."""

    dataset_path = Path(DATASET_DIR)
    cpu_accuracy_path = dataset_path / ACCURACY_FILE_BY_MODE["cpu"]
    device_accuracy_path = dataset_path / ACCURACY_FILE_BY_MODE["device"]

    if not cpu_accuracy_path.exists():
        logger.warning("CPU accuracy file not found: %s", cpu_accuracy_path)
        logger.info("Run CPU accuracy measurement with --measure_cpu_accuracy")
        return
    if not device_accuracy_path.exists():
        logger.warning("Device accuracy file not found: %s", device_accuracy_path)
        logger.info("Run device accuracy measurement with --measure_device_accuracy")
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


def measure_accuracy(
    models: list[str] | None = None,
    server_url: str = None,
    mode: Literal["cpu", "device"] = "cpu",
    authorization: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> None:
    dataset_path = Path(DATASET_DIR)
    metadata = _load_metadata(dataset_path)
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

            results = _replay_samples(
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

        correct, total, mismatches = _analyze_results(results)

        accuracy = (correct / total) if total else 0.0
        accuracy_summary[model] = accuracy

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
            _write_json(mismatch_path, mismatches)
            logger.info(
                "Saved %s mismatches to %s",
                len(mismatches),
                mismatch_path,
            )

    _write_json(summary_path, accuracy_summary)
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
        "--measure_cpu_accuracy", action="store_true", help="Measure CPU model accuracy"
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

    target_models = [args.model] if args.model else MODELS

    if args.download is not None:
        download_samples(count=args.download)
    elif args.measure_cpu_accuracy or args.measure_device_accuracy:
        measure_accuracy(
            models=target_models,
            server_url=args.server_url,
            mode="cpu" if args.measure_cpu_accuracy else "device",
        )
        compare_results()
    elif args.compare_results:
        compare_results()

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import itertools
import json
from pathlib import Path

import argparse
from datasets import DownloadConfig, Image, load_dataset
import requests

from server_helper import (
    DEFAULT_AUTHORIZATION,
    launch_cpu_server,
    sanitize_model_name,
    stop_server,
    wait_for_server_ready,
    SERVER_DEFAULT_URL,
)

DATASET_DIR = "tests/server_tests/datasets/imagenet_subset"
MODELS = [
        'tt-xla-resnet', 
        'tt-xla-vovnet', 
        'tt-xla-mobilenetv2',
        'tt-xla-efficientnet',
        'tt-xla-segformer',
        # 'tt-xla-unet',
        'tt-xla-vit'
]
_MIN_RELATIVE_REFERENCE = 1e-8


def _coerce_probability(value: float | str) -> float:
    if isinstance(value, str):
        stripped = value.strip()
        is_percent = stripped.endswith("%")
        if is_percent:
            stripped = stripped[:-1]
        numeric = float(stripped)
        return numeric / 100.0 if is_percent else numeric
    return float(value)



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

    for entry in metadata:
        image_file = dataset_path / entry["filename"]
        if not image_file.exists():
            raise FileNotFoundError(f"Missing image file: {image_file}")

        with image_file.open("rb") as img_fp:
            encoded = base64.b64encode(img_fp.read()).decode("ascii")

        payload = {"prompt": f"data:image/jpeg;base64,{encoded}"}

        try:
            response = session.post(
                server_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except requests.RequestException as exc:
            body = {
                "error": str(exc),
                "status_code": getattr(exc.response, "status_code", None),
            }
        except json.JSONDecodeError:
            body = {"error": "Response is not valid JSON", "raw_text": response.text}

        results.append({"sample": entry, "response": body})

    session.close()
    return results


def _numeric_close(cpu_value: float, device_value: float, tolerance: float) -> tuple[bool, dict | None]:
    abs_diff = abs(cpu_value - device_value)
    reference = max(abs(cpu_value), abs(device_value), _MIN_RELATIVE_REFERENCE)
    rel_diff = abs_diff / reference
    if rel_diff <= tolerance:
        return True, None
    return False, {
        "cpu": cpu_value,
        "device": device_value,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
    }


def _compare_structures(
    cpu_value,
    device_value,
    tolerance: float,
    path: str = "",
) -> list[dict]:
    differences: list[dict] = []

    cpu_label = cpu_value["image_data"]["top1_class_label"]
    cpu_prob = _coerce_probability(cpu_value["image_data"]["top1_class_probability"])
    device_label = device_value["image_data"]["top1_class_label"]
    device_prob = _coerce_probability(device_value["image_data"]["top1_class_probability"])

    if cpu_label != device_label:
        differences.append(
            {
                "path": f"{path}.image_data.top1_class_label",
                "cpu": cpu_label,
                "device": device_label,
            }
        )
        return differences

    close, details = _numeric_close(cpu_prob, device_prob, tolerance)
    if not close:
        diff = {"path": f"{path}.image_data.top1_class_probability"}
        if details:
            diff.update(details)
        differences.append(diff)

    return differences


def prepare_vision_eval_test(
    count: int = 20,
):
    """Stream a small ImageNet subset and materialize images plus metadata."""

    dataset = load_dataset(
        "imagenet-1k",
        split="validation",
        streaming=True,
        download_config=DownloadConfig(num_proc=1),
    )
    dataset = dataset.cast_column("image", Image(decode=True))
    samples = itertools.islice(dataset, count)
    
    output_path = Path(DATASET_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    label_feature = dataset.features.get("label")
    label_names = label_feature.names if hasattr(label_feature, "names") else None

    metadata = []
    for idx, sample in enumerate(samples):
        image = sample["image"]
        label_id = sample.get("label")
        label_name = (
            label_names[label_id] if label_names and label_id is not None else str(label_id)
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

    print(f"Saved {len(metadata)} ImageNet samples to {output_path} (metadata: {metadata_path})")


def record_cpu_vision_eval_results(
    server_url: str = SERVER_DEFAULT_URL,
    output_filename: str = "responses_cpu.json",
    authorization: str | None = None,
    timeout: float = 60.0,
    models: list[str] | None = None,
):
    """Replay image samples against a local CPU server and persist responses."""

    dataset_path = Path(DATASET_DIR)
    metadata = _load_metadata(dataset_path)
    base_path = Path(output_filename)
    base_name = base_path.stem or "responses_cpu"
    suffix = base_path.suffix or ".json"

    models = models or list(MODELS)
    aggregated_results = {}

    for model in models:
        print(f"\n=== Recording CPU outputs for model: {model} ===")
        process, log_path = launch_cpu_server(model)
        try:
            wait_for_server_ready(process, log_path=log_path)
            model_results = _replay_samples(
                metadata,
                dataset_path,
                server_url,
                authorization,
                timeout,
            )
            aggregated_results[model] = model_results

            safe_model = sanitize_model_name(model)
            model_output = dataset_path / f"{base_name}_{safe_model}{suffix}"
            with model_output.open("w", encoding="utf-8") as f:
                json.dump(model_results, f, indent=2)
            print(f"Recorded {len(model_results)} responses to {model_output}")
        finally:
            stop_server(process)
            if log_path.exists():
                try:
                    log_content = log_path.read_text(encoding="utf-8", errors="replace")
                    tail = log_content[-4000:]
                    print(f"--- Server log for {model} ({log_path}) ---\n{tail}\n--- End log ---")
                except OSError as exc:
                    print(f"Could not read server log {log_path}: {exc}")

    if aggregated_results:
        aggregated_path = dataset_path / f"{base_name}_all{suffix}"
        with aggregated_path.open("w", encoding="utf-8") as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"Saved aggregated results to {aggregated_path}")

    return aggregated_results
    

def _resolve_cpu_results(
    dataset_path: Path,
    cpu_filename: str,
    model: str | None,
) -> tuple[list[dict], Path]:
    base_path = Path(cpu_filename)
    suffix = base_path.suffix or ".json"
    stem = base_path.stem or "responses_cpu"

    candidates = []
    if model:
        safe_model = sanitize_model_name(model)
        candidates.append(dataset_path / f"{stem}_{safe_model}{suffix}")
    candidates.append(dataset_path / cpu_filename)

    cpu_path = None
    data = None
    for candidate in candidates:
        if candidate.exists():
            cpu_path = candidate
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
            break

    if cpu_path is None or data is None:
        paths = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Missing CPU response file for model '{model}': checked {paths}. "
            "Please run recording first with --record_cpu and server in CPU mode (RUNS_ON_CPU=true)."
        )

    if isinstance(data, dict):
        if not model:
            raise ValueError(
                f"Aggregated CPU results in {cpu_path} require specifying a model for comparison."
            )
        if model not in data:
            available = ", ".join(sorted(data.keys()))
            raise KeyError(f"Model '{model}' not found in aggregated CPU results ({available}).")
        cpu_results = data[model]
        if not isinstance(cpu_results, list):
            raise ValueError(f"CPU results for model '{model}' must be a list of samples.")
        return cpu_results, cpu_path

    if not isinstance(data, list):
        raise ValueError(f"CPU results in {cpu_path} must be a list of samples.")

    return data, cpu_path


def compare_with_cpu(
    server_url: str = SERVER_DEFAULT_URL,
    cpu_filename: str = "responses_cpu.json",
    output_filename: str = "responses_tt.json",
    authorization: str | None = None,
    timeout: float = 60.0,
    tolerance: float = 0.05,
    model: str | None = None,
):
    """Compare TT device responses with recorded CPU baselines."""

    if model is None:
        raise ValueError("Model name required when launching the comparison server.")

    dataset_path = Path(DATASET_DIR)
    cpu_results, resolved_path = _resolve_cpu_results(dataset_path, cpu_filename, model)

    process = None
    log_path = None
    try:
        process, log_path = launch_cpu_server(model)
        wait_for_server_ready(process, log_path=log_path)

        metadata = [entry["sample"] for entry in cpu_results]
        device_results = _replay_samples(metadata, dataset_path, server_url, authorization, timeout)
    finally:
        if process is not None:
            stop_server(process)
        if log_path is not None and log_path.exists():
            try:
                log_content = log_path.read_text(encoding="utf-8", errors="replace")
                tail = log_content[-4000:]
                print(f"--- Server log for {model} ({log_path}) ---\n{tail}\n--- End log ---")
            except OSError as exc:
                print(f"Could not read server log {log_path}: {exc}")

    if len(device_results) != len(cpu_results):
        raise ValueError("CPU and device result counts do not match.")

    if output_filename:
        output_path = Path(output_filename)
        suffix = output_path.suffix or ".json"
        stem = output_path.stem or "responses_tt"
        if model:
            safe_model = sanitize_model_name(model)
            output_path = dataset_path / f"{stem}_{safe_model}{suffix}"
        else:
            output_path = dataset_path / output_path
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(device_results, f, indent=2)
        print(f"Saved TT responses to {output_path}")

    mismatches = []
    for cpu_entry, device_entry in zip(cpu_results, device_results):
        if cpu_entry["sample"] != device_entry["sample"]:
            raise ValueError("Sample ordering mismatch between CPU and device runs.")

        differences = _compare_structures(cpu_entry["response"], device_entry["response"], tolerance)
        if differences:
            mismatches.append(
                {
                    "sample": cpu_entry["sample"],
                    "cpu_response": cpu_entry["response"],
                    "device_response": device_entry["response"],
                    "differences": differences,
                }
            )

    total = len(cpu_results)
    match_count = total - len(mismatches)
    label = f" for model '{model}'" if model else ""
    print(
        f"Compared {total} samples{label} using CPU baseline {resolved_path}: {match_count} match, {len(mismatches)} differ"
    )

    if mismatches:
        mismatch_path = Path("mismatches.json")
        suffix = mismatch_path.suffix or ".json"
        stem = mismatch_path.stem or "mismatches"
        if model:
            safe_model = sanitize_model_name(model)
            mismatch_path = dataset_path / f"{stem}_{safe_model}{suffix}"
        else:
            mismatch_path = dataset_path / mismatch_path
        with mismatch_path.open("w", encoding="utf-8") as f:
            json.dump(mismatches, f, indent=2)
        print(f"Wrote mismatch details to {mismatch_path}")
        print(
            f"❌ Vision evaluation failed for {model}: {len(mismatches)} mismatches (details: {mismatch_path})"
        )
    else:
        print(
            f"✅ Vision evaluation passed for {model}: all {total} samples match the CPU baseline"
        )

    return mismatches


def compare_models_with_cpu(
    server_url: str = SERVER_DEFAULT_URL,
    cpu_filename: str = "responses_cpu.json",
    output_filename: str = "responses_tt.json",
    authorization: str | None = None,
    timeout: float = 60.0,
    tolerance: float = 0.05,
    models: list[str] | None = None,
):
    """Iterate over the configured models and compare TT results with CPU baselines."""
    
    summary: dict[str, list[dict]] = {}

    target_models = models or list(MODELS)
    passed = 0
    failed = 0

    for model in target_models:
        print(f"\n=== Comparing TT outputs for model: {model} ===")
        try:
            mismatches = compare_with_cpu(
                server_url=server_url,
                cpu_filename=cpu_filename,
                output_filename=output_filename,
                authorization=authorization,
                timeout=timeout,
                tolerance=tolerance,
                model=model,
            )
            summary[model] = mismatches
            if mismatches:
                failed += 1
            else:
                passed += 1
        except Exception as exc:  # pragma: no cover - surface errors to caller
            failed += 1
            summary[model] = [{"error": str(exc)}]
            print(f"❌ Comparison failed for {model}: {exc}")

    total = len(target_models)
    if total:
        if failed:
            print(
                f"\n❌ Vision evaluation comparisons completed with {failed} failure(s) out of {total} model(s)"
            )
        else:
            print(
                f"\n✅ Vision evaluation comparisons passed for all {total} model(s)"
            )

    return summary


"""

Prepare for test
================

1. Install datasets library
2. Export HF_TOKEN
3. Download images
    Download 20 images from ImageNet validation set
    Save images and metadata to folder "tests/server_tests/datasets/imagenet_subset"
4. Record CPU results
    For each vison model    
    Start server in "cpu" mode
    Run test to record results
    Save to folder "tests/server_tests/datasets/imagenet_subset"
    
Test
====

1. For each model run using TT device
2. Compare results with expected values
3. Report differences


Example usage
=============

Prepare dataset:
python tests/server_tests/test_cases/vision_evals_test.py --download

Record CPU results for all models:
python tests/server_tests/test_cases/vision.py --record_cpu

Record CPU results for specific model:
python tests/server_tests/test_cases/vision_evals_test.py --record_cpu --model tt-xla-resnet

Compare TT device results with CPU baselines for all models:
python tests/server_tests/test_cases/vision_evals_test.py --compare

Compare TT device results with CPU baselines for specific model:
python tests/server_tests/test_cases/vision_evals_test.py --compare --model tt-xla-resnet

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision evaluation test utility")
    parser.add_argument("--download", action="store_true", help="Download and prepare the dataset")
    parser.add_argument("--record_cpu", action="store_true", help="Record CPU server responses")
    parser.add_argument("--compare", action="store_true", help="Compare TT device responses with CPU baselines")
    parser.add_argument("--model", help="Specific model runner to compare; defaults to all configured models.")
    parser.add_argument("--server_url", help="Server URL to use for TT device comparisons; defaults to SERVER_DEFAULT_URL.")
    args = parser.parse_args()
    target_models = [args.model] if args.model else MODELS
    if args.download:
        prepare_vision_eval_test()
    if args.record_cpu:
        record_cpu_vision_eval_results(models=target_models)
    if args.compare:        
        compare_models_with_cpu(models=target_models, server_url=args.server_url or SERVER_DEFAULT_URL)

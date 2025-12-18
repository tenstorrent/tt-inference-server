#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Script to automatically add markers to tests in server_tests_config.json.

This script analyzes test configurations and suggests/applies appropriate markers
based on test name, model, device, and other characteristics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List


def infer_markers_from_test(test: Dict, model: str, device: str) -> List[str]:
    """
    Infer appropriate markers for a test based on its characteristics.

    Args:
        test: Test configuration dictionary
        model: Model name
        device: Device name

    Returns:
        List of suggested markers
    """
    markers = []

    # Device marker
    device_lower = device.lower()
    if device_lower in ["n150", "n300", "t3k", "galaxy"]:
        markers.append(device_lower)

    # Model category markers
    image_models = [
        "stable-diffusion-xl-base-1.0",
        "stable-diffusion-xl-base-1.0-img-2-img",
        "stable-diffusion-xl-1.0-inpainting-0.1",
        "stable-diffusion-3.5-large",
    ]

    audio_models = [
        "distil-whisper/distil-large-v3",
        "openai/whisper-large-v3",
        "distil-large-v3",
        "whisper-large-v3",
    ]

    cnn_models = [
        "resnet-50",
        "vovnet",
        "mobilenetv2",
        "Qwen3-Embedding-4B",
    ]

    if model in image_models:
        markers.append("image")
    elif model in audio_models:
        markers.append("audio")
    elif model in cnn_models:
        markers.append("cnn")

    # Specific model markers
    if "stable-diffusion-xl" in model or "sdxl" in model.lower():
        markers.append("sdxl")
    elif (
        "stable-diffusion-3.5" in model
        or "sd3.5" in model.lower()
        or "sd35" in model.lower()
    ):
        markers.append("sd35")
    elif "whisper" in model.lower():
        markers.append("whisper")
    elif "resnet" in model.lower():
        markers.append("resnet")
    elif "vovnet" in model.lower():
        markers.append("vovnet")
    elif "mobilenet" in model.lower():
        markers.append("mobilenet")
    elif "qwen" in model.lower():
        markers.append("qwen")

    # Test type markers based on test name
    test_name = test.get("name", "")

    if "Load" in test_name:
        markers.extend(["load", "e2e", "slow", "heavy"])
    elif "Param" in test_name:
        markers.extend(["param", "e2e", "slow"])
    elif "Liveness" in test_name:
        markers.extend(["smoke", "functional", "fast"])
    elif "Eval" in test_name:
        markers.extend(["eval", "e2e", "slow", "heavy"])
    elif "Stability" in test_name:
        markers.extend(["stability", "e2e", "slow", "heavy"])

    # Remove duplicates while preserving order
    seen = set()
    unique_markers = []
    for marker in markers:
        if marker not in seen:
            seen.add(marker)
            unique_markers.append(marker)

    return unique_markers


def add_markers_to_config(config_path: Path, dry_run: bool = True) -> None:
    """
    Add markers to all tests in the configuration file.

    Args:
        config_path: Path to server_tests_config.json
        dry_run: If True, only print changes without modifying the file
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    test_cases = config.get("test_cases", [])
    modified_count = 0
    total_count = 0

    for test_group in test_cases:
        models = test_group.get("weights", [])
        device = test_group.get("device", "")

        for test in test_group.get("test_cases", []):
            total_count += 1
            test_name = test.get("name", "")

            # Check if test already has markers
            existing_markers = test.get("markers", [])

            if not existing_markers:
                # Infer markers for the first model in the list
                model = models[0] if models else ""
                suggested_markers = infer_markers_from_test(test, model, device)

                if dry_run:
                    print(f"\nTest: {test_name}")
                    print(f"  Model: {model}")
                    print(f"  Device: {device}")
                    print(f"  Suggested markers: {suggested_markers}")
                else:
                    test["markers"] = suggested_markers

                modified_count += 1
            else:
                if dry_run:
                    print(
                        f"\nTest: {test_name} - Already has markers: {existing_markers}"
                    )

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Total tests: {total_count}")
    print(f"  Tests needing markers: {modified_count}")
    print(f"  Tests with markers: {total_count - modified_count}")

    if not dry_run:
        # Write back to file with proper formatting
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print(f"\n✅ Configuration file updated: {config_path}")
    else:
        print("\n⚠️  Dry run mode - no changes written to file")
        print("   Run with --apply to apply changes")


def validate_markers(config_path: Path) -> None:
    """
    Validate that all tests have appropriate markers.

    Args:
        config_path: Path to server_tests_config.json
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    available_markers = config.get("available_markers", {})
    all_valid_markers = set()

    # Collect all valid markers from available_markers
    for category, markers in available_markers.items():
        if isinstance(markers, dict):
            all_valid_markers.update(markers.keys())

    test_cases = config.get("test_cases", [])
    issues = []

    for test_group in test_cases:
        models = test_group.get("weights", [])
        device = test_group.get("device", "")

        for test in test_group.get("test_cases", []):
            test_name = test.get("name", "")
            markers = test.get("markers", [])

            if not markers:
                issues.append(f"❌ {test_name}: No markers defined")
                continue

            # Check for invalid markers
            invalid = [m for m in markers if m not in all_valid_markers]
            if invalid:
                issues.append(f"⚠️  {test_name}: Invalid markers: {invalid}")

            # Check for required marker categories
            has_model_category = any(
                m in ["image", "audio", "cnn", "llm"] for m in markers
            )
            has_test_type = any(
                m
                in ["load", "param", "functional", "smoke", "e2e", "stability", "eval"]
                for m in markers
            )
            has_device = device.lower() in markers

            if not has_model_category:
                issues.append(
                    f"⚠️  {test_name}: Missing model category marker (image/audio/cnn/llm)"
                )

            if not has_test_type:
                issues.append(f"⚠️  {test_name}: Missing test type marker")

            if not has_device:
                issues.append(f"⚠️  {test_name}: Missing device marker ({device})")

    print(f"\n{'=' * 70}")
    print("Marker Validation Results")
    print(f"{'=' * 70}")

    if issues:
        print(f"\nFound {len(issues)} issues:\n")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All tests have valid markers!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Add or validate markers in server_tests_config.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "server_tests_config.json",
        help="Path to server_tests_config.json",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to the file (default is dry-run)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing markers instead of adding new ones",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.validate:
        validate_markers(args.config)
    else:
        add_markers_to_config(args.config, dry_run=not args.apply)


if __name__ == "__main__":
    main()

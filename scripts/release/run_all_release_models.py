#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from workflows.model_spec import MODEL_SPECS
from evals.eval_config import EVAL_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.utils import get_repo_root_path


def parse_arguments():
    """Parse command-line arguments."""
    valid_devices = {device.name.lower() for device in DeviceTypes}

    parser = argparse.ArgumentParser(
        description="Run release workflow for all models on a specified device.",
        epilog="Additional arguments will be passed through to run.py",
    )

    parser.add_argument(
        "--device",
        required=True,
        choices=valid_devices,
        help=f"Device to run models on (choices: {', '.join(sorted(valid_devices))})",
    )

    parser.add_argument(
        "--evals-only",
        action="store_true",
        help="Run only models that have evals configured in EVAL_CONFIGS",
    )

    # Use parse_known_args to capture pass-through arguments
    args, passthrough_args = parser.parse_known_args()

    return args, passthrough_args


def filter_models(device_str, evals_only):
    """Filter MODEL_SPECS by device and optionally by evals availability."""
    device_type = DeviceTypes.from_string(device_str)

    filtered_models = []
    for model_id, model_spec in MODEL_SPECS.items():
        # Filter by device type
        if model_spec.device_type != device_type:
            continue

        # Filter by evals availability if requested
        if evals_only and model_spec.model_name not in EVAL_CONFIGS:
            continue

        filtered_models.append(model_spec)

    return filtered_models


def run_release_workflow(model_spec, device_str, passthrough_args):
    """Run the release workflow for a single model."""
    repo_root = get_repo_root_path()
    run_py = repo_root / "run.py"

    # Construct command
    cmd = [
        sys.executable,
        str(run_py),
        "--model",
        model_spec.model_name,
        "--device",
        device_str,
        "--workflow",
        "release",
    ]

    # Add pass-through arguments
    if passthrough_args:
        cmd.extend(passthrough_args)

    # Print command for visibility
    print(f"\n{'=' * 80}")
    print(f"Running release workflow for: {model_spec.model_name} on {device_str}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    # Execute command
    result = subprocess.run(cmd, cwd=repo_root)

    return result.returncode


def main():
    """Main entry point."""
    args, passthrough_args = parse_arguments()

    # Filter models based on criteria
    filtered_models = filter_models(args.device, args.evals_only)

    if not filtered_models:
        print(f"No models found for device '{args.device}' with the specified filters.")
        sys.exit(1)

    print(f"\nFound {len(filtered_models)} model(s) to run on {args.device}:")
    for model_spec in filtered_models:
        print(f"  - {model_spec.model_name} (impl: {model_spec.impl.impl_name})")
    print()

    # Run release workflow for each model
    for i, model_spec in enumerate(filtered_models, 1):
        print(f"\n[{i}/{len(filtered_models)}] Processing {model_spec.model_name}...")

        return_code = run_release_workflow(model_spec, args.device, passthrough_args)

        if return_code != 0:
            print(
                f"\n⛔ Release workflow failed for {model_spec.model_name} with exit code {return_code}"
            )
            print("Stopping execution as requested (stop on first failure).")
            sys.exit(return_code)

        print(
            f"\n✅ Release workflow completed successfully for {model_spec.model_name}"
        )

    print(f"\n{'=' * 80}")
    print(f"✅ All {len(filtered_models)} model(s) completed successfully!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

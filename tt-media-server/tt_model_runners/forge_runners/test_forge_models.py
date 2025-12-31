# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


# export TT_METAL_HOME=venv-worker/lib/python3.11/site-packages/pjrt_plugin_tt/tt-metal

import json
import os
import sys
from pathlib import Path

# Add tt-media-server to Python path
current_file = Path(__file__)
tt_media_server_dir = (
    current_file.parent.parent.parent
)  # Go up 3 levels to tt-media-server
sys.path.insert(0, str(tt_media_server_dir))

from typing import Any, List

import pytest
from domain.image_search_request import ImageSearchRequest

from .runners import (
    ForgeEfficientnetRunner,
    ForgeMobilenetv2Runner,
    ForgeResnetRunner,
    ForgeSegformerRunner,
    ForgeVitRunner,
    ForgeVovnetRunner,
)

pytestmark = pytest.mark.asyncio


class TestForgeRunners:
    expected_results = {
        "ForgeResnetRunner": {
            "label": "lion, king of beasts, Panthera leo",
            "min_accuracy": 0.80,
        },
        "ForgeSegformerRunner": {
            "label": "lion, king of beasts, Panthera leo",
            "min_accuracy": 0.05,
        },
        "_default": {
            "label": "lion",
            "min_accuracy": 0.80,
        },
    }

    @pytest.mark.parametrize(
        "mode,runner_class",
        [
            (mode, runner_class)
            for mode in [
                "cpu",
                "device",
                # "optimizer"
            ]
            for runner_class in [
                ForgeMobilenetv2Runner,
                ForgeResnetRunner,
                ForgeVovnetRunner,
                ForgeEfficientnetRunner,
                ForgeSegformerRunner,
                # ForgeUnetRunner,
                ForgeVitRunner,
            ]
        ],
    )
    async def test_forge_runner_modes(self, mode, runner_class):
        """Test ForgeRunner with different execution modes."""

        # Set environment based on mode
        if mode == "cpu":
            os.environ["RUNS_ON_CPU"] = "true"
            os.environ["USE_OPTIMIZER"] = "false"
        elif mode == "device":
            os.environ["RUNS_ON_CPU"] = "false"
            os.environ["USE_OPTIMIZER"] = "false"
        elif mode == "optimizer":
            os.environ["RUNS_ON_CPU"] = "false"
            os.environ["USE_OPTIMIZER"] = "true"

        try:
            runner = runner_class(device_id="0")
            await runner.warmup()
            requests = create_image_search_request()
            result = runner.run(requests)

            # Print runner class and result for debugging/expected output generation
            print(f"\n=== {runner_class.__name__} in {mode} mode ===")
            print(f"Runner class: {runner_class.__name__}")
            print(f"Result (JSON): {json.dumps(result, indent=2)}")
            print("=" * 50)

            # Get expected result for this runner
            config = self.expected_results.get(
                runner_class.__name__
            ) or self.expected_results.get("_default")
            expected_label, expected_min_accuracy = (
                config["label"],
                config["min_accuracy"],
            )

            # Verify result structure and content using expected results
            if not verify_inference_output(
                result,
                expected_label=expected_label,
                min_accuracy=expected_min_accuracy,
            ):
                pytest.fail(
                    f"Output verification failed for {runner_class.__name__} in {mode} mode. "
                    f"Expected '{expected_label}' with >{expected_min_accuracy:.0%} confidence. Actual output: {result}"
                )

        except Exception as e:
            pytest.fail(
                f"{mode.capitalize()} mode test failed for {runner_class.__name__}: {str(e)}"
            )


def create_image_search_request() -> List[ImageSearchRequest]:
    """Create a test ImageSearchRequest with a base64 encoded image loaded from test payload."""
    # Path to the test image payload file
    current_file = Path(__file__)
    utils_dir = (
        current_file.parent.parent.parent.parent / "utils"
    )  # Go up to tt-inference-server/utils
    image_payload_file = utils_dir / "test_payloads" / "image_client_image_payload"

    # Read the base64 image data
    with open(image_payload_file, "r") as f:
        image_data = f.read().strip()

    # Remove the data URL prefix if present
    if image_data.startswith("data:image/jpeg;base64,"):
        image_data = image_data.split(",", 1)[1]

    return [ImageSearchRequest(prompt=image_data)]


def verify_inference_output(
    result: Any, expected_label: str, min_accuracy: float
) -> bool:
    """Verify that the inference output has the expected structure and content."""

    result = result[0]  # Get the first result for verification

    # Handle new JSON format: [{"object": "...", "confidence_level": ...}]
    if isinstance(result, list) and len(result) > 0:
        first_prediction = result[0]
        label = first_prediction.get("object", "")
        prob_raw = first_prediction.get("confidence_level", 0)
    # Handle legacy format: {"top1_class_label": "...", "top1_class_probability": ...}
    elif isinstance(result, dict) and "top1_class_label" in result:
        label = result["top1_class_label"]
        prob_raw = result["top1_class_probability"]
    else:
        return False

    # Normalize probability to float (0.0 to 1.0)
    if isinstance(prob_raw, str):
        prob = float(prob_raw.rstrip("%")) / 100.0
    else:
        # New format returns 0-100, convert to 0-1
        prob = float(prob_raw) / 100.0 if prob_raw > 1 else float(prob_raw)

    # Check if probability meets minimum accuracy requirement
    if prob < min_accuracy:
        return False

    # Check if label contains the expected label
    label_lower = label.lower()
    if expected_label.lower() in label_lower:
        return True

    return False

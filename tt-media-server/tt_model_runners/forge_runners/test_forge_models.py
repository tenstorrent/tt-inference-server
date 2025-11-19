# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


# export TT_METAL_HOME=venv-worker/lib/python3.11/site-packages/pjrt_plugin_tt/tt-metal

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
    ForgeMobilenetv2Runner,
    ForgeResnetRunner,
    ForgeVovnetRunner,
    ForgeEfficientnetRunner,
)

pytestmark = pytest.mark.asyncio


class TestForgeRunners:
    
    @pytest.mark.parametrize("mode,runner_class", [
        (mode, runner_class) 
        for mode in [
            "cpu", 
            # "device", 
            # "optimizer"
        ]
        for runner_class in [
            ForgeMobilenetv2Runner,
            ForgeResnetRunner,
            ForgeVovnetRunner,
            ForgeEfficientnetRunner,
        ]
    ])
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
            await runner.load_model()
            requests = create_image_search_request()
            result = runner.run_inference(requests)
            
            # Verify result structure and content
            expected_labels = ["lion", "dog", "bear"]
            min_accuracy = 0.50
            if not verify_inference_output(result, expected_labels=expected_labels, min_accuracy=min_accuracy):
                pytest.fail(f"Output verification failed for {runner_class.__name__} in {mode} mode. "
                           f"Expected one of {expected_labels} with >{min_accuracy:.0%} confidence. Actual output: {result}")
            
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


def verify_inference_output(result: Any, expected_labels: list[str], min_accuracy: float) -> bool:
    """Verify that the inference output has the expected structure and content."""

    result = result[0]  # Get the first result for verification
    label = result["top1_class_label"]
    prob_raw = result["top1_class_probability"]
    
    # Normalize probability to float (0.0 to 1.0)
    if isinstance(prob_raw, str):
        prob = float(prob_raw.rstrip('%')) / 100.0
    else:
        prob = float(prob_raw)
    
    # Check if probability meets minimum accuracy requirement
    if prob < min_accuracy:
        return False
    
    # Check if label contains any of the expected labels
    label_lower = label.lower()
    for expected_label in expected_labels:
        if expected_label.lower() in label_lower:
            return True
    
    return False



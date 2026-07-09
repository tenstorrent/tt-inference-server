# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""T2V vs I2V endpoint and payload selection for video generation evals."""

from __future__ import annotations

import base64
from pathlib import Path

VIDEO_GENERATION_ENDPOINT = "v1/videos/generations"
VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT = "v1/videos/generations/i2v"
FIXTURE_IMAGE_PATH = (
    Path("server_tests") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
DEFAULT_I2V_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_I2V_SEED = 42


def is_i2v_video_model(model_name: str) -> bool:
    """Return True when ``model_name`` is an image-to-video variant."""
    return "-I2V-" in model_name


def _load_fixture_image_base64() -> str:
    """Read the repo-checked-in fixture image and return it base64-encoded."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from server_tests/datasets/imagenet_subset/."
        )
    return base64.b64encode(FIXTURE_IMAGE_PATH.read_bytes()).decode("ascii")


def build_video_generation_payload(
    *,
    prompt: str,
    num_inference_steps: int,
    model_name: str,
    image_b64: str | None = None,
) -> dict:
    """Build a T2V or I2V submit payload for ``model_name``."""
    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
    }
    if not is_i2v_video_model(model_name):
        return payload

    if image_b64 is None:
        image_b64 = _load_fixture_image_base64()

    payload.update(
        {
            "negative_prompt": DEFAULT_I2V_NEGATIVE_PROMPT,
            "seed": DEFAULT_I2V_SEED,
            "image_prompts": [{"image": image_b64, "frame_pos": 0}],
        }
    )
    return payload


def get_video_generation_submit_endpoint(model_name: str) -> str:
    """Return the POST path for T2V or I2V video generation."""
    if is_i2v_video_model(model_name):
        return VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT
    return VIDEO_GENERATION_ENDPOINT

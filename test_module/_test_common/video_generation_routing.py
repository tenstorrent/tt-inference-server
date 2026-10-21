# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""T2V vs I2V endpoint and payload selection for video generation.

Shared by both the video generation evals and benchmarks so the two flows
submit jobs to the same endpoints with the same payload shape.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

VIDEO_GENERATION_ENDPOINT = "v1/videos/generations"
VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT = "v1/videos/generations/i2v"
FIXTURE_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
# Image-caption pairs for the I2V eval: each image with the caption that
# describes it (see its ATTRIBUTION.md).
I2V_EVAL_PAIRS_DIR = Path("test_fixtures") / "datasets" / "coco_i2v"
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
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    return base64.b64encode(FIXTURE_IMAGE_PATH.read_bytes()).decode("ascii")


def load_i2v_eval_pairs(num_pairs: int) -> list[dict]:
    """The first ``num_pairs`` I2V eval pairs: ``{prompt, image, image_b64, concepts}``.

    ``prompt`` is the caption of ``image``, so the video is scored against text
    that describes what it was conditioned on; ``concepts`` are the things in it
    the video should keep showing, each scored separately.
    """
    manifest = json.loads((I2V_EVAL_PAIRS_DIR / "manifest.json").read_text())
    if num_pairs > len(manifest):
        raise ValueError(
            f"I2V eval wants {num_pairs} pairs; {I2V_EVAL_PAIRS_DIR} has {len(manifest)}"
        )
    return [
        {
            "prompt": entry["caption"],
            "image": entry["image"],
            "concepts": entry.get("concepts", []),
            "image_b64": base64.b64encode(
                (I2V_EVAL_PAIRS_DIR / entry["image"]).read_bytes()
            ).decode("ascii"),
        }
        for entry in manifest[:num_pairs]
    ]


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

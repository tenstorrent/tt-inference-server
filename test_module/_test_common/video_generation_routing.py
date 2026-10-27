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
import os
from pathlib import Path

VIDEO_GENERATION_ENDPOINT = "v1/videos/generations"
VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT = "v1/videos/generations/i2v"
FIXTURE_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
# Image-caption pairs for the I2V eval: each image with the caption that
# describes it (see each set's ATTRIBUTION.md). I2V_EVAL_PAIRS picks the set,
# and each set is evaluated in full at its own size: five COCO pairs by
# default, or fifteen from VBench-I2V's image suite (four with a camera move).
I2V_EVAL_DATASETS_DIR = Path("test_fixtures") / "datasets"
I2V_EVAL_PAIRS_DIR = I2V_EVAL_DATASETS_DIR / "coco_i2v"
I2V_EVAL_PAIR_SETS = {"coco_i2v": 5, "vbench_i2v": 15}
I2V_EVAL_PAIRS_ENV = "I2V_EVAL_PAIRS"
DEFAULT_I2V_EVAL_PAIRS = "coco_i2v"
DEFAULT_I2V_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_I2V_SEED = 42
# Wan2.2 I2V output shapes, as the generation API spec requests them
# (tt-media-server docs/api/generation/models/wan-ai/wan2.2-i2v-a14b.yaml):
# (width, height) -> (aspect_ratio, size).
WAN22_I2V_SHAPES = {
    (832, 480): ("16:9", "480p"),
    (480, 832): ("9:16", "480p"),
    (1280, 720): ("16:9", "720p"),
    (720, 1280): ("9:16", "720p"),
}


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


def i2v_eval_pair_set() -> str:
    """The I2V eval's pair set: ``I2V_EVAL_PAIRS``, or COCO when unset."""
    name = os.getenv(I2V_EVAL_PAIRS_ENV) or DEFAULT_I2V_EVAL_PAIRS
    if name not in I2V_EVAL_PAIR_SETS:
        raise ValueError(
            f"{I2V_EVAL_PAIRS_ENV}={name!r}; expected one of {sorted(I2V_EVAL_PAIR_SETS)}"
        )
    return name


def i2v_eval_pairs_dir() -> Path:
    return I2V_EVAL_DATASETS_DIR / i2v_eval_pair_set()


def i2v_eval_num_pairs() -> int:
    """How many pairs the I2V eval runs: the whole selected set."""
    return I2V_EVAL_PAIR_SETS[i2v_eval_pair_set()]


def load_i2v_eval_pairs(num_pairs: int) -> list[dict]:
    """The first ``num_pairs`` I2V eval pairs: ``{prompt, image, image_b64, concepts}``.

    ``prompt`` is the caption of ``image``, so the video is scored against text
    that describes what it was conditioned on; ``concepts`` are the things in it
    the video should keep showing, each scored separately. A pair that asks for
    a camera move also carries ``camera_motion`` (VBench's name for the move).
    """
    pairs_dir = i2v_eval_pairs_dir()
    manifest = json.loads((pairs_dir / "manifest.json").read_text())
    if num_pairs > len(manifest):
        raise ValueError(
            f"I2V eval wants {num_pairs} pairs; {pairs_dir} has {len(manifest)}"
        )
    pairs = []
    for entry in manifest[:num_pairs]:
        pair = {
            "prompt": entry["caption"],
            "image": entry["image"],
            "concepts": entry.get("concepts", []),
            "image_b64": base64.b64encode(
                (pairs_dir / entry["image"]).read_bytes()
            ).decode("ascii"),
        }
        if entry.get("camera_motion"):
            pair["camera_motion"] = entry["camera_motion"]
        pairs.append(pair)
    return pairs


def build_video_generation_payload(
    *,
    prompt: str,
    num_inference_steps: int,
    model_name: str,
    image_b64: str | None = None,
    shape: tuple[int, int] | None = None,
) -> dict:
    """Build a T2V or I2V submit payload for ``model_name``.

    ``shape`` asks for an output ``(width, height)`` with the spec's
    ``aspect_ratio`` and ``size``. Today's server ignores both and returns its
    deployment's shape, which the caller's output check then catches.
    """
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
    if shape is not None:
        aspect_ratio, size = WAN22_I2V_SHAPES[tuple(shape)]
        payload.update({"aspect_ratio": aspect_ratio, "size": size})
    return payload


def get_video_generation_submit_endpoint(model_name: str) -> str:
    """Return the POST path for T2V or I2V video generation."""
    if is_i2v_video_model(model_name):
        return VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT
    return VIDEO_GENERATION_ENDPOINT

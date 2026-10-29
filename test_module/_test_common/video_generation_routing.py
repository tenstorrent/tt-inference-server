# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""T2V vs I2V endpoint and payload selection for video generation.

Shared by both the video generation evals and benchmarks so the two flows
submit jobs to the same endpoints with the same payload shape.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

VIDEO_GENERATION_ENDPOINT = "v1/videos/generations"
VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT = "v1/videos/generations/i2v"
FIXTURE_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
# The I2V eval's videos: the validation set, 50 VBench-I2V pairs at the four
# output shapes. Each video has its own shape, seed and Wan2.2's negative
# prompt. Its manifest and first frames are read from the eval data on Hugging
# Face (vbench_i2v.I2V_EVAL_HF_REPO), which also holds the tool that made them.
# I2V_EVAL_SUBSET picks dev (5 videos), ci (20, the default) or all (65).
I2V_EVAL_MANIFEST = "pairs/manifest.json"
I2V_EVAL_SUBSETS = ("dev", "ci", "all")
I2V_EVAL_SUBSET_ENV = "I2V_EVAL_SUBSET"
DEFAULT_I2V_EVAL_SUBSET = "ci"
# What the benchmark and spec tests send; the eval sends each video's own.
DEFAULT_I2V_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_I2V_SEED = 42
# Wan2.2's sample_neg_prompt (wan/configs/shared_config.py): the API spec's
# default, and what the I2V eval's GPU references were made with.
WAN22_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)
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


def i2v_eval_subset(model_spec=None) -> str:
    """The I2V eval's subset: ``I2V_EVAL_SUBSET``, or ci when unset.

    Read from the deployment's model spec first, so a run can pick it with
    ``run.py --override-env I2V_EVAL_SUBSET=dev``, then from the environment.
    """
    env_vars = getattr(model_spec, "env_vars", None) or {}
    subset = (
        env_vars.get(I2V_EVAL_SUBSET_ENV)
        or os.getenv(I2V_EVAL_SUBSET_ENV)
        or DEFAULT_I2V_EVAL_SUBSET
    )
    if subset not in I2V_EVAL_SUBSETS:
        raise ValueError(
            f"{I2V_EVAL_SUBSET_ENV}={subset!r}; expected one of {I2V_EVAL_SUBSETS}"
        )
    return subset


def i2v_eval_videos(subset: str | None = None) -> list[dict]:
    """The validation set's manifest entries in ``subset`` (default: the eval's), one per video."""
    from .vbench_i2v import hf_snapshot

    subset = subset or i2v_eval_subset()
    manifest_path = hf_snapshot([I2V_EVAL_MANIFEST]) / I2V_EVAL_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    return [e for e in manifest if subset == "all" or e[subset]]


def i2v_eval_num_pairs(subset: str | None = None) -> int:
    """How many videos the I2V eval runs: its whole subset."""
    return len(i2v_eval_videos(subset))


def load_i2v_eval_pairs(num_pairs: int, subset: str | None = None) -> list[dict]:
    """The I2V eval's first ``num_pairs`` videos: ``{prompt, image, image_b64, ...}``.

    ``prompt`` is VBench's prompt for ``image``, so the video is scored against
    text that describes what it was conditioned on. Each also carries the
    request its GPU reference was made with: ``shape`` ``(width, height)``,
    ``seed`` and ``negative_prompt``, plus its ``video_id``; a pair that asks
    for a camera move carries ``camera_motion`` (VBench's name for the move).
    First frames come from the eval data on Hugging Face and must match the
    manifest's sha256.
    """
    from .vbench_i2v import hf_snapshot

    subset = subset or i2v_eval_subset()
    videos = i2v_eval_videos(subset)
    if num_pairs > len(videos):
        raise ValueError(
            f"I2V eval wants {num_pairs} videos; subset {subset} has {len(videos)}"
        )
    videos = videos[:num_pairs]
    root = hf_snapshot([e["image"] for e in videos])
    pairs = []
    for entry in videos:
        data = (root / entry["image"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["image_sha256"]:
            raise ValueError(f"{entry['image']} does not match the manifest's sha256")
        pair = {
            "prompt": entry["caption"],
            "image": Path(entry["image"]).name,
            "image_b64": base64.b64encode(data).decode("ascii"),
            "shape": (entry["width"], entry["height"]),
            "seed": entry["seed"],
            "negative_prompt": WAN22_NEGATIVE_PROMPT,
            "video_id": entry["video_id"],
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
    seed: int | None = None,
    negative_prompt: str | None = None,
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
            "negative_prompt": negative_prompt or DEFAULT_I2V_NEGATIVE_PROMPT,
            "seed": DEFAULT_I2V_SEED if seed is None else seed,
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

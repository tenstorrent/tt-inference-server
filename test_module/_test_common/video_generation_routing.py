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
# Image-caption pairs for the I2V eval: each image with the caption that
# describes it (see each set's ATTRIBUTION.md). I2V_EVAL_PAIRS picks the set:
# five COCO pairs by default, or fifteen from VBench-I2V's image suite (four
# with a camera move), each run in full at the deployment's shape. Or the
# validation set, vbench_i2v_50[:dev|ci|all] (ci when no subset is named): 50
# VBench-I2V pairs at the four output shapes, each video with its own shape,
# seed and Wan2.2's negative prompt, its first frames read from the eval data
# on Hugging Face (scripts/select_vbench_i2v_pairs.py).
I2V_EVAL_DATASETS_DIR = Path("test_fixtures") / "datasets"
I2V_EVAL_PAIRS_DIR = I2V_EVAL_DATASETS_DIR / "coco_i2v"
VALIDATION_PAIR_SET = "vbench_i2v_50"
I2V_EVAL_SUBSETS = ("dev", "ci", "all")
DEFAULT_I2V_EVAL_SUBSET = "ci"
# Pairs per set; the validation set's count depends on the subset.
I2V_EVAL_PAIR_SETS = {"coco_i2v": 5, "vbench_i2v": 15, VALIDATION_PAIR_SET: None}
I2V_EVAL_PAIRS_ENV = "I2V_EVAL_PAIRS"
DEFAULT_I2V_EVAL_PAIRS = "vbench_i2v_50:dev"
DEFAULT_I2V_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_I2V_SEED = 42
# Wan2.2's sample_neg_prompt (wan/configs/shared_config.py): the API spec's
# default, and what the validation set's GPU references were made with.
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


def i2v_eval_selection() -> tuple[str, str | None]:
    """``I2V_EVAL_PAIRS`` as (pair set, subset); COCO when unset.

    Only the validation set has subsets; it runs ci when none is named.
    """
    value = os.getenv(I2V_EVAL_PAIRS_ENV) or DEFAULT_I2V_EVAL_PAIRS
    name, _, subset = value.partition(":")
    if name not in I2V_EVAL_PAIR_SETS:
        raise ValueError(
            f"{I2V_EVAL_PAIRS_ENV}={value!r}; expected one of {sorted(I2V_EVAL_PAIR_SETS)}"
        )
    if name != VALIDATION_PAIR_SET:
        if subset:
            raise ValueError(
                f"{I2V_EVAL_PAIRS_ENV}={value!r}; only {VALIDATION_PAIR_SET} has subsets"
            )
        return name, None
    subset = subset or DEFAULT_I2V_EVAL_SUBSET
    if subset not in I2V_EVAL_SUBSETS:
        raise ValueError(
            f"{I2V_EVAL_PAIRS_ENV}={value!r}; subset must be one of {I2V_EVAL_SUBSETS}"
        )
    return name, subset


def i2v_eval_pair_set() -> str:
    """The I2V eval's pair set: ``I2V_EVAL_PAIRS`` without its subset."""
    return i2v_eval_selection()[0]


def i2v_eval_pairs_dir() -> Path:
    return I2V_EVAL_DATASETS_DIR / i2v_eval_pair_set()


def _validation_videos(subset: str) -> list[dict]:
    """The validation set's manifest entries in ``subset``, one per video."""
    manifest = json.loads(
        (I2V_EVAL_DATASETS_DIR / VALIDATION_PAIR_SET / "manifest.json").read_text()
    )
    return [e for e in manifest if subset == "all" or e[subset]]


def i2v_eval_num_pairs() -> int:
    """How many pairs (videos, for the validation set) the I2V eval runs."""
    name, subset = i2v_eval_selection()
    count = I2V_EVAL_PAIR_SETS[name]
    return count if count is not None else len(_validation_videos(subset))


def load_i2v_eval_pairs(num_pairs: int) -> list[dict]:
    """The first ``num_pairs`` I2V eval pairs: ``{prompt, image, image_b64, concepts}``.

    ``prompt`` is the caption of ``image``, so the video is scored against text
    that describes what it was conditioned on; ``concepts`` are the things in it
    the video should keep showing, each scored separately. A pair that asks for
    a camera move also carries ``camera_motion`` (VBench's name for the move).
    """
    name, subset = i2v_eval_selection()
    if name == VALIDATION_PAIR_SET:
        return _load_validation_videos(num_pairs, subset)
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


def _load_validation_videos(num_videos: int, subset: str) -> list[dict]:
    """The validation set's first ``num_videos`` videos in ``subset``, as eval pairs.

    Each also carries the request its GPU reference was made with: ``shape``
    ``(width, height)``, ``seed`` and ``negative_prompt``, plus its
    ``video_id``. First frames come from the eval data on Hugging Face and
    must match the manifest's sha256.
    """
    from .vbench_i2v import hf_snapshot

    videos = _validation_videos(subset)
    if num_videos > len(videos):
        raise ValueError(
            f"I2V eval wants {num_videos} videos; {VALIDATION_PAIR_SET}:{subset} "
            f"has {len(videos)}"
        )
    videos = videos[:num_videos]
    root = hf_snapshot([e["image"] for e in videos])
    pairs = []
    for entry in videos:
        data = (root / entry["image"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["image_sha256"]:
            raise ValueError(f"{entry['image']} does not match the manifest's sha256")
        pair = {
            "prompt": entry["caption"],
            "image": Path(entry["image"]).name,
            "concepts": [],
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

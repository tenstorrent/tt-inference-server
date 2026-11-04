# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""T2V vs I2V endpoint and payload selection for video generation.

Shared by both the video generation evals and benchmarks so the two flows
submit jobs to the same endpoints with the same payload shape.

Models in ``GENERATIONS_API_SUBMIT_ENDPOINTS`` are served by the generations
API instead of ``v1/videos``: one submit path per model, jobs under
``v1/jobs``, and the video behind the completed job's output ``url``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

VIDEO_GENERATION_ENDPOINT = "v1/videos/generations"
VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT = "v1/videos/generations/i2v"
GENERATIONS_API_JOBS_ENDPOINT = "v1/jobs"
# Keyed by HF repo name without the org, so both the served model name and the
# full repo ID match.
GENERATIONS_API_SUBMIT_ENDPOINTS = {
    "Wan2.2-I2V-A14B-Diffusers": "v1/generations/wan-ai/wan2.2-i2v-a14b",
    "MiniMax-H3-FL2VA": "v1/generations/minimaxai/minimax-h3-fl2va",
    "MiniMax-H3-Ref2VA": "v1/generations/minimaxai/minimax-h3-ref2va",
    "MiniMax-H3-FL2VA-LightX2V": "v1/generations/tenstorrent/minimax-h3-fl2va-lightx2v",
    "MiniMax-H3-Ref2VA-LightX2V": "v1/generations/tenstorrent/minimax-h3-ref2va-lightx2v",
}
FIXTURE_IMAGE_PATH = (
    Path("test_fixtures") / "datasets" / "imagenet_subset" / "imagenet_002_volcano.jpg"
)
DEFAULT_I2V_NEGATIVE_PROMPT = "blurry, low quality, distorted"
DEFAULT_I2V_SEED = 42
MINIMAX_H3_DURATION_SECONDS = 5


def is_i2v_video_model(model_name: str) -> bool:
    """Return True when ``model_name`` is an image-to-video variant."""
    return "-I2V-" in model_name


def is_minimax_h3_model(model_name: str) -> bool:
    return model_name.split("/")[-1].startswith("MiniMax-H3")


def uses_generations_api(model_name: str) -> bool:
    return model_name.split("/")[-1] in GENERATIONS_API_SUBMIT_ENDPOINTS


def wan_size_for_device(device: Any) -> str:
    """Wan2.2's size tier on ``device``: 720p on a Blackhole Galaxy, the only
    mesh whose tt_dit preset serves it, and 480p elsewhere. The tests send
    16:9, the one aspect ratio every preset serves."""
    return (
        "720p" if device is not None and device.name == "BLACKHOLE_GALAXY" else "480p"
    )


def _load_fixture_image_base64() -> str:
    """Read the repo-checked-in fixture image and return it base64-encoded."""
    if not FIXTURE_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"I2V fixture image missing at {FIXTURE_IMAGE_PATH}. "
            "Expected a tracked sample from test_fixtures/datasets/imagenet_subset/."
        )
    return base64.b64encode(FIXTURE_IMAGE_PATH.read_bytes()).decode("ascii")


def fixture_image_data_uri(image_b64: str | None = None) -> str:
    """The (JPEG) fixture image as a data URI, the generations API's media form."""
    return "data:image/jpeg;base64," + (image_b64 or _load_fixture_image_base64())


def build_video_generation_payload(
    *,
    prompt: str,
    num_inference_steps: int,
    model_name: str,
    image_b64: str | None = None,
    device: Any = None,
) -> dict:
    """Build a T2V or I2V submit payload for ``model_name``. The generations
    API takes no step count: it runs the model's reference default."""
    if is_minimax_h3_model(model_name):
        payload = {
            "prompt": prompt,
            "duration": MINIMAX_H3_DURATION_SECONDS,
            "seed": 0,
        }
        if "Ref2VA" in model_name:
            # Let the server pick the canvas for 16:9.
            payload.update(
                aspect_ratio="16:9", images=[fixture_image_data_uri(image_b64)]
            )
        else:
            # The clip takes the keyframe's shape.
            payload["first_frame"] = fixture_image_data_uri(image_b64)
        return payload
    if uses_generations_api(model_name):
        return {
            "prompt": prompt,
            "negative_prompt": DEFAULT_I2V_NEGATIVE_PROMPT,
            "seed": DEFAULT_I2V_SEED,
            "first_frame": fixture_image_data_uri(image_b64),
            "size": wan_size_for_device(device),
            "aspect_ratio": "16:9",
        }

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
    if uses_generations_api(model_name):
        return GENERATIONS_API_SUBMIT_ENDPOINTS[model_name.split("/")[-1]]
    if is_i2v_video_model(model_name):
        return VIDEO_GENERATION_I2V_SUBMIT_ENDPOINT
    return VIDEO_GENERATION_ENDPOINT


def get_video_job_endpoint(model_name: str, job_id: str) -> str:
    """Return the GET path of a submitted job's status."""
    if uses_generations_api(model_name):
        return f"{GENERATIONS_API_JOBS_ENDPOINT}/{job_id}"
    return f"{VIDEO_GENERATION_ENDPOINT}/{job_id}"


def get_video_download_url(
    base_url: str, model_name: str, job_id: str, headers: dict
) -> str:
    """Return the URL of a completed job's video.

    For the generations API that is the ``url`` of the completed job's output,
    which takes a request to look up. A relative url is resolved against
    ``base_url``.
    """
    if not uses_generations_api(model_name):
        return f"{base_url}/{VIDEO_GENERATION_ENDPOINT}/{job_id}/download"
    response = requests.get(
        f"{base_url}/{GENERATIONS_API_JOBS_ENDPOINT}/{job_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return urljoin(base_url + "/", response.json()["outputs"][0]["url"])

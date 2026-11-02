# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Wan2.2 I2V generation API the spec tests hold a deployment to.

Paths, fields, limits and job rules follow the design in
``tt-media-server/docs/api/generation/models/wan-ai/wan2.2-i2v-a14b.yaml``
(and the ``common.yaml`` it refers to) from tenstorrent/tt-inference-server
PR #5466. tt-media-server does not implement that API yet, so until it does
the Wan I2V spec tests are expected to fail: every submit 404s. They have no
fallback to today's ``/v1/videos`` API on purpose; they test the design.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlsplit

from . import generation_api
from .generation_api import (
    ACTIVE_STATUSES,
    CANCEL_PATH,
    CONTENT_PATH,
    DEFAULT_API_KEY,
    DEFAULT_RETENTION_SECONDS,
    FINAL_STATUSES,
    JOB_PATH,
    JOBS_PATH,
    MAX_IMAGE_MB,
    SEED_MAX,
    auth_headers,
    data_uri,
    resolve_api_key,
    validation_error_problems,
)
from .video_generation_routing import WAN22_I2V_SHAPES

MODEL_ID = "wan-ai/wan2.2-i2v-a14b"
SUBMIT_PATH = f"/v1/generations/{MODEL_ID}"
# Another model's submit route: a deployment serves one model and answers 404
# (NotServed) for the rest. Wan2.2 T2V, which an I2V deployment used to half-serve.
UNSERVED_SUBMIT_PATH = "/v1/generations/wan-ai/wan2.2-t2v-a14b"

REQUEST_FIELDS = frozenset(
    {
        "prompt",
        "negative_prompt",
        "first_frame",
        "last_frame",
        "aspect_ratio",
        "size",
        "frames",
        "seed",
    }
)
ASPECT_RATIOS = ("16:9", "9:16")
SIZES = ("480p", "720p")
# (aspect_ratio, size) -> output (width, height), Wan2.2's reference sizes.
OUTPUT_SHAPES = {request: shape for shape, request in WAN22_I2V_SHAPES.items()}
NUM_FRAMES = 81
FPS = 16
DURATION_SECONDS = NUM_FRAMES / FPS
# The duration is a float the server computes; allow for its rounding.
DURATION_TOLERANCE_SECONDS = 0.05
VIDEO_CONTENT_TYPE = "video/mp4"


def shape_fields(shape: tuple[int, int]) -> dict[str, str]:
    """The request's ``aspect_ratio`` and ``size`` for an output ``(width, height)``."""
    aspect_ratio, size = WAN22_I2V_SHAPES[tuple(shape)]
    return {"aspect_ratio": aspect_ratio, "size": size}


def requested_shape(payload: dict[str, Any]) -> tuple[int, int]:
    """The output ``(width, height)`` a valid request asks for."""
    return OUTPUT_SHAPES[(payload["aspect_ratio"], payload["size"])]


def job_problems(
    job: Any,
    *,
    job_id: str | None = None,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> list[str]:
    """Where a Job body breaks the schema and the rules of its status.

    ``outputs`` is only checked for presence here; ``video_output_problems``
    checks what a completed job's output says.
    """
    return generation_api.job_problems(
        job, model_id=MODEL_ID, job_id=job_id, retention_seconds=retention_seconds
    )


def video_output_problems(job: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """Where a completed job's one VideoOutput disagrees with its request.

    The seed is the requested one, or any valid seed when the request had none
    (the output reports the one chosen). The URL is the job's
    ``content/0`` path, or an absolute URL if outputs ever move to storage.
    """
    outputs = job.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        return [f"expected one output, got {outputs!r:.200}"]
    output = outputs[0]
    if not isinstance(output, dict):
        return [f"output is not an object: {output!r:.200}"]
    width, height = requested_shape(request)
    expected = {
        "type": "video",
        "content_type": VIDEO_CONTENT_TYPE,
        "prompt": request["prompt"],
        "width": width,
        "height": height,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
    }
    problems = [
        f"{key} is {output.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if output.get(key) != value
    ]

    url = output.get("url")
    content_path = CONTENT_PATH.format(job_id=job.get("id"), index=0)
    if not isinstance(url, str) or (
        url != content_path and urlsplit(url).scheme not in ("http", "https")
    ):
        problems.append(f"url is {url!r}, expected {content_path!r}")

    seed = output.get("seed")
    if "seed" in request:
        if seed != request["seed"]:
            problems.append(f"seed is {seed!r}, expected {request['seed']!r}")
    elif not (generation_api.is_int(seed) and 0 <= seed <= SEED_MAX):
        problems.append(f"seed is {seed!r}, expected the one chosen")

    duration = output.get("duration")
    if not (
        isinstance(duration, (int, float))
        and math.isclose(duration, DURATION_SECONDS, abs_tol=DURATION_TOLERANCE_SECONDS)
    ):
        problems.append(f"duration is {duration!r}, expected ~{DURATION_SECONDS} s")
    # Wan2.2 makes no sound: no audio track, and audio is null (or left out).
    if output.get("audio") is not None:
        problems.append(f"audio is {output['audio']!r}, expected null")
    return problems


__all__ = [
    "ACTIVE_STATUSES",
    "ASPECT_RATIOS",
    "CANCEL_PATH",
    "CONTENT_PATH",
    "DEFAULT_API_KEY",
    "DEFAULT_RETENTION_SECONDS",
    "DURATION_SECONDS",
    "FINAL_STATUSES",
    "FPS",
    "JOBS_PATH",
    "JOB_PATH",
    "MAX_IMAGE_MB",
    "MODEL_ID",
    "NUM_FRAMES",
    "OUTPUT_SHAPES",
    "REQUEST_FIELDS",
    "SEED_MAX",
    "SIZES",
    "SUBMIT_PATH",
    "UNSERVED_SUBMIT_PATH",
    "VIDEO_CONTENT_TYPE",
    "auth_headers",
    "data_uri",
    "job_problems",
    "requested_shape",
    "resolve_api_key",
    "shape_fields",
    "validation_error_problems",
    "video_output_problems",
]

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

import base64
import math
import os
from typing import Any
from urllib.parse import urlsplit

from .video_generation_routing import WAN22_I2V_SHAPES

MODEL_ID = "wan-ai/wan2.2-i2v-a14b"
SUBMIT_PATH = f"/v1/generations/{MODEL_ID}"
# Another model's submit route: a deployment serves one model and answers 404
# (NotServed) for the rest. Wan2.2 T2V, which an I2V deployment used to half-serve.
UNSERVED_SUBMIT_PATH = "/v1/generations/wan-ai/wan2.2-t2v-a14b"
JOBS_PATH = "/v1/jobs"
JOB_PATH = "/v1/jobs/{job_id}"
CONTENT_PATH = "/v1/jobs/{job_id}/content/{index}"
CANCEL_PATH = "/v1/jobs/{job_id}/cancel"

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
SEED_MAX = 2**31 - 1
# 30 MB per image, inline or fetched (413 above it).
MAX_IMAGE_MB = 30
# Final jobs are deleted completed_at + this many seconds later, by default.
DEFAULT_RETENTION_SECONDS = 24 * 3600
ACTIVE_STATUSES = ("queued", "in_progress", "cancelling")
FINAL_STATUSES = ("completed", "failed", "cancelled")
VIDEO_CONTENT_TYPE = "video/mp4"
DEFAULT_API_KEY = "your-secret-key"

_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),
)


def resolve_api_key() -> str:
    """The key the server was started with (security/api_key_checker.py default)."""
    return os.getenv("API_KEY") or DEFAULT_API_KEY


def auth_headers(api_key: str | None = None) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key or resolve_api_key()}",
    }


def data_uri(image: bytes) -> str:
    """``image`` as the data URI the API takes inline (bare base64 is a 422)."""
    content_type = next(
        (kind for magic, kind in _IMAGE_SIGNATURES if image.startswith(magic)),
        "application/octet-stream",
    )
    return f"data:{content_type};base64,{base64.b64encode(image).decode('ascii')}"


def shape_fields(shape: tuple[int, int]) -> dict[str, str]:
    """The request's ``aspect_ratio`` and ``size`` for an output ``(width, height)``."""
    aspect_ratio, size = WAN22_I2V_SHAPES[tuple(shape)]
    return {"aspect_ratio": aspect_ratio, "size": size}


def requested_shape(payload: dict[str, Any]) -> tuple[int, int]:
    """The output ``(width, height)`` a valid request asks for."""
    return OUTPUT_SHAPES[(payload["aspect_ratio"], payload["size"])]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
    if not isinstance(job, dict):
        return [f"job is not an object: {job!r:.200}"]
    problems = []
    if not isinstance(job.get("id"), str) or not job["id"]:
        problems.append(f"id is {job.get('id')!r}")
    elif job_id is not None and job["id"] != job_id:
        problems.append(f"id is {job['id']!r}, expected {job_id!r}")
    if job.get("model") != MODEL_ID:
        problems.append(f"model is {job.get('model')!r}, expected {MODEL_ID!r}")
    status = job.get("status")
    if status not in ACTIVE_STATUSES + FINAL_STATUSES:
        problems.append(f"status is {status!r}")
    created_at = job.get("created_at")
    if not _is_int(created_at):
        problems.append(f"created_at is {created_at!r}, expected Unix seconds")

    completed_at, expires_at = job.get("completed_at"), job.get("expires_at")
    if status in FINAL_STATUSES:
        if not _is_int(completed_at):
            problems.append(f"{status} job has completed_at {completed_at!r}")
        else:
            if _is_int(created_at) and completed_at < created_at:
                problems.append(
                    f"completed_at {completed_at} is before created_at {created_at}"
                )
            if expires_at != completed_at + retention_seconds:
                problems.append(
                    f"expires_at is {expires_at!r}, expected completed_at + "
                    f"{retention_seconds} = {completed_at + retention_seconds}"
                )
    elif completed_at is not None or expires_at is not None:
        problems.append(
            f"{status} job has completed_at {completed_at!r}, expires_at {expires_at!r}"
        )

    error = job.get("error")
    if status == "failed":
        if not (
            isinstance(error, dict)
            and isinstance(error.get("code"), str)
            and isinstance(error.get("message"), str)
        ):
            problems.append(f"failed job's error is {error!r}")
    elif error is not None:
        problems.append(f"{status} job has error {error!r}")

    outputs = job.get("outputs")
    if status == "completed":
        if not isinstance(outputs, list) or not outputs:
            problems.append(f"completed job's outputs are {outputs!r}")
    elif outputs is not None:
        problems.append(f"{status} job has outputs")
    return problems


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
    elif not (_is_int(seed) and 0 <= seed <= SEED_MAX):
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


def validation_error_problems(
    detail: Any, *, fields: tuple[str, ...] = (), min_errors: int = 1
) -> list[str]:
    """Where a 422 ``detail`` breaks ValidationError: ``[{loc, msg, type}, ...]``.

    ``fields`` must each appear in some entry's ``loc``: the error names what
    was wrong, and every problem is listed, not just the first.
    """
    if not isinstance(detail, list):
        return [f"detail is {type(detail).__name__}, expected a list of errors"]
    problems = []
    if len(detail) < min_errors:
        problems.append(f"{len(detail)} error(s), expected at least {min_errors}")
    for entry in detail:
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("loc"), list)
            and isinstance(entry.get("msg"), str)
            and isinstance(entry.get("type"), str)
        ):
            problems.append(f"error is not {{loc, msg, type}}: {entry!r:.200}")
    locs = [
        entry["loc"]
        for entry in detail
        if isinstance(entry, dict) and isinstance(entry.get("loc"), list)
    ]
    problems += [
        f"no error names {field!r}"
        for field in fields
        if not any(field in loc for loc in locs)
    ]
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

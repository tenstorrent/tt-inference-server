# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""What every model's generation API shares, for the spec tests.

The common part of the design in ``tt-media-server/docs/api/generation/``
(``common.yaml``) from tenstorrent/tt-inference-server PR #5466: the ``/v1/jobs``
paths and the Job schema, auth, inline images as data URIs, seeds, the image
size limit and the ValidationError body. Each model's own request schema is in
``wan_i2v_generation_api`` and ``image_generation_api``.
"""

from __future__ import annotations

import base64
import os
from typing import Any
from urllib.parse import urlsplit

JOBS_PATH = "/v1/jobs"
JOB_PATH = "/v1/jobs/{job_id}"
CONTENT_PATH = "/v1/jobs/{job_id}/content/{index}"
CANCEL_PATH = "/v1/jobs/{job_id}/cancel"

SEED_MAX = 2**31 - 1
# 30 MB per image, inline or fetched (413 above it).
MAX_IMAGE_MB = 30
# Final jobs are deleted completed_at + this many seconds later, by default.
DEFAULT_RETENTION_SECONDS = 24 * 3600
ACTIVE_STATUSES = ("queued", "in_progress", "cancelling")
FINAL_STATUSES = ("completed", "failed", "cancelled")
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


def image_content_type(image: bytes) -> str:
    """PNG, JPEG or WebP by its signature, else ``application/octet-stream``."""
    return next(
        (kind for magic, kind in _IMAGE_SIGNATURES if image.startswith(magic)),
        "application/octet-stream",
    )


def data_uri(image: bytes) -> str:
    """``image`` as the data URI the API takes inline (bare base64 is a 422)."""
    return f"data:{image_content_type(image)};base64,{base64.b64encode(image).decode('ascii')}"


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def output_url_problem(url: Any, job_id: Any, index: int = 0) -> str | None:
    """Why an output's ``url`` is not where the job's file is, or None.

    The URL is a relative reference to the job's ``content/{index}`` path,
    which a client resolves against the server's base URL (RFC 3986), or an
    absolute URL if outputs ever move to storage. The server writes the
    relative one without a leading ``/`` so a base URL with a path keeps it;
    the design's example has one. Both resolve to the same path on a base URL
    without one, so both are taken.
    """
    content_path = CONTENT_PATH.format(job_id=job_id, index=index)
    if isinstance(url, str) and (
        url.lstrip("/") == content_path.lstrip("/")
        or urlsplit(url).scheme in ("http", "https")
    ):
        return None
    return f"url is {url!r}, expected {content_path.lstrip('/')!r}"


def job_problems(
    job: Any,
    *,
    model_id: str,
    job_id: str | None = None,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
) -> list[str]:
    """Where a Job body breaks the schema and the rules of its status.

    ``outputs`` is only checked for presence here; each model's output check
    looks at what a completed job's outputs say.
    """
    if not isinstance(job, dict):
        return [f"job is not an object: {job!r:.200}"]
    problems = []
    if not isinstance(job.get("id"), str) or not job["id"]:
        problems.append(f"id is {job.get('id')!r}")
    elif job_id is not None and job["id"] != job_id:
        problems.append(f"id is {job['id']!r}, expected {job_id!r}")
    if job.get("model") != model_id:
        problems.append(f"model is {job.get('model')!r}, expected {model_id!r}")
    status = job.get("status")
    if status not in ACTIVE_STATUSES + FINAL_STATUSES:
        problems.append(f"status is {status!r}")
    created_at = job.get("created_at")
    if not is_int(created_at):
        problems.append(f"created_at is {created_at!r}, expected Unix seconds")

    completed_at, expires_at = job.get("completed_at"), job.get("expires_at")
    if status in FINAL_STATUSES:
        if not is_int(completed_at):
            problems.append(f"{status} job has completed_at {completed_at!r}")
        else:
            if is_int(created_at) and completed_at < created_at:
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

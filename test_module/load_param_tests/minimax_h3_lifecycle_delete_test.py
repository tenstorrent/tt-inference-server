# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end lifecycle and deletion checks for MiniMax-H3 video tasks.

The API assertions are derived from MiniMax's published V2 documentation:

* Create: https://platform.minimax.io/docs/api-reference/video-generation-v2-create
* Query: https://platform.minimax.io/docs/api-reference/video-generation-v2-query
* Delete: https://platform.minimax.io/docs/api-reference/video-generation-v2-delete

The test creates one inexpensive text-to-video task, polls it to a terminal
state, downloads and inspects the successful output, and deletes the terminal
task record. It never sends the MiniMax Bearer token to the output CDN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aiohttp  # pyright: ignore[reportMissingImports]

from report_module.schema import Block
from test_module._test_common import BaseTest, HardwareRequirement, TestConfig

if TYPE_CHECKING:
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

CREATE_PATH = "/v2/video_generation"
QUERY_PATH = "/v2/query/video_generation/{task_id}"
DELETE_PATH = "/v2/video_generation/{task_id}"
MODEL_NAME = "MiniMax-H3"

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
DOCUMENTED_STATUSES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled"}
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_POLL_TIMEOUT_SECONDS = 900.0
DEFAULT_TEST_TIMEOUT_SECONDS = 1500
MAX_RESPONSE_EXCERPT = 500
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MP4_HEADER_BYTES = 64

EXPECTED_RESOLUTION = "768P"
EXPECTED_DURATION_SECONDS = 4
EXPECTED_RATIO = "16:9"
RATIO_TOLERANCE = 0.03
DURATION_TOLERANCE_SECONDS = 0.5

FRAME_WIDTH = 64
FRAME_HEIGHT = 64
FRAME_CHANNELS = 3
MIN_FRAME_BRIGHTNESS = 2.0
MIN_MEAN_FRAME_DELTA = 0.5


def _create_payload() -> dict[str, Any]:
    """A low-cost valid T2V request with visible motion and a bright scene."""

    return {
        "model": MODEL_NAME,
        "content": [
            {
                "type": "text",
                "text": (
                    "A bright red kite flies across a clear blue daytime sky "
                    "above a sunlit green field, with smooth visible motion."
                ),
            }
        ],
        "resolution": EXPECTED_RESOLUTION,
        "duration": EXPECTED_DURATION_SECONDS,
        "ratio": EXPECTED_RATIO,
    }


def _resolve_api_key() -> str:
    for env_name in ("MINIMAX_API_KEY", "MINIMAX_MOCK_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError(
        "Set MINIMAX_API_KEY (real API) or MINIMAX_MOCK_API_KEY (mock API)"
    )


def _api_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _error_result(
    check: str,
    *,
    expected_status: int | str,
    actual_status: int | str,
    message: str,
) -> dict[str, Any]:
    return {
        "check": check,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status": "FAIL",
        "passed": False,
        "message": message,
    }


def _success_result(
    check: str,
    *,
    expected_status: int | str,
    actual_status: int | str,
    message: str = "",
    **details: Any,
) -> dict[str, Any]:
    return {
        "check": check,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status": "PASS",
        "passed": True,
        "message": message,
        **details,
    }


def _decode_json(response_text: str) -> Any:
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def _response_excerpt(response_text: str) -> str:
    return response_text.replace("\n", " ")[:MAX_RESPONSE_EXCERPT]


def _validate_error_envelope(
    data: Any,
    *,
    expected_status: int,
    expected_error_type: str,
) -> tuple[bool, str]:
    if not isinstance(data, dict) or data.get("type") != "error":
        return False, "response is not a MiniMax error object"
    error = data.get("error")
    if not isinstance(error, dict):
        return False, "response is missing the error object"
    if error.get("type") != expected_error_type:
        return (
            False,
            f"expected error.type={expected_error_type!r}, got {error.get('type')!r}",
        )
    if error.get("http_code") != str(expected_status):
        return (
            False,
            (
                f"expected error.http_code={str(expected_status)!r}, "
                f"got {error.get('http_code')!r}"
            ),
        )
    if not isinstance(error.get("message"), str) or not error["message"].strip():
        return False, "error.message is missing or empty"
    if not isinstance(data.get("request_id"), str) or not data["request_id"].strip():
        return False, "request_id is missing or empty"
    return True, ""


async def _delete_error_case(
    session: aiohttp.ClientSession,
    *,
    endpoint_url: str,
    check: str,
    expected_status: int,
    expected_error_type: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    try:
        async with session.delete(endpoint_url, headers=headers) as response:
            response_text = await response.text()
            data = _decode_json(response_text)
            if response.status != expected_status:
                return _error_result(
                    check,
                    expected_status=expected_status,
                    actual_status=response.status,
                    message=(
                        f"unexpected HTTP status; "
                        f"response={_response_excerpt(response_text)!r}"
                    ),
                )
            passed, message = _validate_error_envelope(
                data,
                expected_status=expected_status,
                expected_error_type=expected_error_type,
            )
            if not passed:
                return _error_result(
                    check,
                    expected_status=expected_status,
                    actual_status=response.status,
                    message=message,
                )
            return _success_result(
                check,
                expected_status=expected_status,
                actual_status=response.status,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return _error_result(
            check,
            expected_status=expected_status,
            actual_status="request_error",
            message=f"{type(exc).__name__}: {exc}",
        )


async def _check_delete_errors(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
) -> list[dict[str, Any]]:
    unknown_task_url = (
        f"{base_url.rstrip('/')}"
        f"{DELETE_PATH.format(task_id='definitely-not-a-real-task-id')}"
    )
    json_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    return [
        await _delete_error_case(
            session,
            endpoint_url=unknown_task_url,
            check="delete_requires_bearer_authentication",
            expected_status=HTTP_UNAUTHORIZED,
            expected_error_type="authorized_error",
            headers=json_headers,
        ),
        await _delete_error_case(
            session,
            endpoint_url=unknown_task_url,
            check="delete_rejects_invalid_bearer_authentication",
            expected_status=HTTP_UNAUTHORIZED,
            expected_error_type="authorized_error",
            headers={
                **json_headers,
                "Authorization": "Bearer definitely-invalid-minimax-key",
            },
        ),
        await _delete_error_case(
            session,
            endpoint_url=unknown_task_url,
            check="delete_rejects_unknown_task_id",
            expected_status=HTTP_BAD_REQUEST,
            expected_error_type="bad_request_error",
            headers=_api_headers(api_key),
        ),
    ]


async def _create_task(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
) -> tuple[dict[str, Any], str | None]:
    endpoint_url = f"{base_url.rstrip('/')}{CREATE_PATH}"
    try:
        async with session.post(
            endpoint_url,
            headers=_api_headers(api_key),
            json=_create_payload(),
        ) as response:
            response_text = await response.text()
            data = _decode_json(response_text)
            if response.status != HTTP_OK:
                return (
                    _error_result(
                        "create_lifecycle_task",
                        expected_status=HTTP_OK,
                        actual_status=response.status,
                        message=(
                            f"task creation failed; "
                            f"response={_response_excerpt(response_text)!r}"
                        ),
                    ),
                    None,
                )
            task_id = data.get("task_id") if isinstance(data, dict) else None
            if not isinstance(task_id, str) or not task_id.strip():
                return (
                    _error_result(
                        "create_lifecycle_task",
                        expected_status=HTTP_OK,
                        actual_status=response.status,
                        message="response is missing a non-empty string task_id",
                    ),
                    None,
                )
            return (
                _success_result(
                    "create_lifecycle_task",
                    expected_status=HTTP_OK,
                    actual_status=response.status,
                    task_id=task_id,
                ),
                task_id,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return (
            _error_result(
                "create_lifecycle_task",
                expected_status=HTTP_OK,
                actual_status="request_error",
                message=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


def _validate_poll_task(
    task: Any,
    *,
    task_id: str,
    previous_created_at: int | None,
    previous_updated_at: int | None,
) -> tuple[bool, str, int | None, int | None]:
    if not isinstance(task, dict):
        return False, "query response is missing the task object", None, None
    if task.get("id") != task_id:
        return (
            False,
            f"expected task.id={task_id!r}, got {task.get('id')!r}",
            None,
            None,
        )
    if task.get("status") not in DOCUMENTED_STATUSES:
        return (
            False,
            f"undocumented task status {task.get('status')!r}",
            None,
            None,
        )

    created_at = task.get("created_at")
    updated_at = task.get("updated_at")
    if not isinstance(created_at, int) or not isinstance(updated_at, int):
        return (
            False,
            "created_at and updated_at must be Unix integer timestamps",
            None,
            None,
        )
    if created_at > updated_at:
        return False, "created_at is later than updated_at", None, None
    if previous_created_at is not None and created_at != previous_created_at:
        return False, "created_at changed between polls", None, None
    if previous_updated_at is not None and updated_at < previous_updated_at:
        return False, "updated_at moved backwards between polls", None, None
    return True, "", created_at, updated_at


async def _poll_task(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    poll_interval: float,
    poll_timeout: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    endpoint_url = f"{base_url.rstrip('/')}{QUERY_PATH.format(task_id=task_id)}"
    start = time.monotonic()
    observed_statuses: list[str] = []
    previous_created_at: int | None = None
    previous_updated_at: int | None = None

    while time.monotonic() - start < poll_timeout:
        try:
            async with session.get(
                endpoint_url,
                headers=_api_headers(api_key),
            ) as response:
                response_text = await response.text()
                data = _decode_json(response_text)
                if response.status != HTTP_OK:
                    return (
                        _error_result(
                            "query_task_lifecycle",
                            expected_status=HTTP_OK,
                            actual_status=response.status,
                            message=(
                                f"query failed; "
                                f"response={_response_excerpt(response_text)!r}"
                            ),
                        ),
                        None,
                    )

                task = data.get("task") if isinstance(data, dict) else None
                passed, message, created_at, updated_at = _validate_poll_task(
                    task,
                    task_id=task_id,
                    previous_created_at=previous_created_at,
                    previous_updated_at=previous_updated_at,
                )
                if not passed:
                    return (
                        _error_result(
                            "query_task_lifecycle",
                            expected_status=HTTP_OK,
                            actual_status=response.status,
                            message=message,
                        ),
                        None,
                    )

                previous_created_at = created_at
                previous_updated_at = updated_at
                status = task["status"]
                if not observed_statuses or observed_statuses[-1] != status:
                    observed_statuses.append(status)
                logger.info("MiniMax task %s status=%s", task_id, status)

                if status in TERMINAL_STATUSES:
                    return (
                        _success_result(
                            "query_task_lifecycle",
                            expected_status=HTTP_OK,
                            actual_status=response.status,
                            observed_statuses=observed_statuses,
                        ),
                        task,
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            return (
                _error_result(
                    "query_task_lifecycle",
                    expected_status=HTTP_OK,
                    actual_status="request_error",
                    message=f"{type(exc).__name__}: {exc}",
                ),
                None,
            )

        await asyncio.sleep(poll_interval)

    return (
        _error_result(
            "query_task_lifecycle",
            expected_status=HTTP_OK,
            actual_status="timeout",
            message=f"task did not finish within {poll_timeout:.1f} seconds",
        ),
        None,
    )


def _validate_succeeded_task(
    task: dict[str, Any],
    *,
    task_id: str,
) -> tuple[dict[str, Any], str | None]:
    if task.get("status") != "succeeded":
        error = task.get("error")
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="succeeded",
                actual_status=str(task.get("status")),
                message=f"task reached a non-success terminal state; error={error!r}",
            ),
            None,
        )

    expected_fields = {
        "id",
        "model",
        "status",
        "created_at",
        "updated_at",
        "content",
        "resolution",
        "duration",
        "usage",
        "ratio",
        "task_type",
        "modality",
    }
    missing = sorted(expected_fields - task.keys())
    if missing:
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="documented task fields",
                actual_status="missing fields",
                message=f"missing fields: {missing}",
            ),
            None,
        )

    expected_values = {
        "id": task_id,
        "model": MODEL_NAME,
        "resolution": EXPECTED_RESOLUTION,
        "duration": EXPECTED_DURATION_SECONDS,
        "ratio": EXPECTED_RATIO,
        "task_type": "generation",
        "modality": "video",
    }
    mismatches = {
        field: {"expected": expected, "actual": task.get(field)}
        for field, expected in expected_values.items()
        if task.get(field) != expected
    }
    if mismatches:
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="documented task metadata",
                actual_status="metadata mismatch",
                message=json.dumps(mismatches, sort_keys=True),
            ),
            None,
        )
    if "error" in task:
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="no error field",
                actual_status="error field present",
                message=f"unexpected error={task.get('error')!r}",
            ),
            None,
        )

    usage = task.get("usage")
    if not isinstance(usage, dict):
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="usage object",
                actual_status=type(usage).__name__,
                message="task usage is missing or invalid",
            ),
            None,
        )
    usage_fields = (
        "total_seconds",
        "input_seconds",
        "output_seconds",
        "input_image_count",
    )
    if any(not isinstance(usage.get(field), int) for field in usage_fields):
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="integer video usage fields",
                actual_status="invalid usage",
                message=f"usage={usage!r}",
            ),
            None,
        )
    if (
        usage["input_seconds"] != 0
        or usage["input_image_count"] != 0
        or usage["output_seconds"] != EXPECTED_DURATION_SECONDS
        or usage["total_seconds"] != usage["input_seconds"] + usage["output_seconds"]
    ):
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="T2V usage matching the requested duration",
                actual_status="usage mismatch",
                message=f"usage={usage!r}",
            ),
            None,
        )

    content = task.get("content")
    content_url = content.get("url") if isinstance(content, dict) else None
    if not isinstance(content_url, str) or not content_url.strip():
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="non-empty content.url",
                actual_status="missing URL",
                message=f"content={content!r}",
            ),
            None,
        )
    parsed = urlparse(content_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return (
            _error_result(
                "validate_succeeded_task",
                expected_status="absolute HTTP(S) content URL",
                actual_status=content_url,
                message="content.url is not an absolute download URL",
            ),
            None,
        )

    return (
        _success_result(
            "validate_succeeded_task",
            expected_status="succeeded",
            actual_status="succeeded",
            usage=usage,
        ),
        content_url,
    )


async def _download_video(
    *,
    content_url: str,
    destination: Path,
    download_timeout: float,
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=download_timeout)
    try:
        # This session intentionally has no Authorization header: content.url
        # can point to a third-party CDN and must not receive the API secret.
        async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
            async with session.get(content_url) as response:
                if response.status != HTTP_OK:
                    response_text = await response.text()
                    return _error_result(
                        "download_generated_video",
                        expected_status=HTTP_OK,
                        actual_status=response.status,
                        message=(
                            f"download failed; "
                            f"response={_response_excerpt(response_text)!r}"
                        ),
                    )

                total_bytes = 0
                first_bytes = b""
                with destination.open("wb") as output:
                    async for chunk in response.content.iter_chunked(
                        DOWNLOAD_CHUNK_BYTES
                    ):
                        if not chunk:
                            continue
                        if len(first_bytes) < MP4_HEADER_BYTES:
                            needed = MP4_HEADER_BYTES - len(first_bytes)
                            first_bytes += chunk[:needed]
                        output.write(chunk)
                        total_bytes += len(chunk)

                if total_bytes == 0:
                    return _error_result(
                        "download_generated_video",
                        expected_status="non-empty video",
                        actual_status="empty response",
                        message="content URL returned zero bytes",
                    )
                if b"ftyp" not in first_bytes:
                    return _error_result(
                        "download_generated_video",
                        expected_status="MP4 file signature",
                        actual_status="signature missing",
                        message=f"first bytes={first_bytes[:32]!r}",
                    )
                return _success_result(
                    "download_generated_video",
                    expected_status=HTTP_OK,
                    actual_status=response.status,
                    bytes_downloaded=total_bytes,
                    content_type=response.headers.get("Content-Type", ""),
                )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return _error_result(
            "download_generated_video",
            expected_status=HTTP_OK,
            actual_status="request_error",
            message=f"{type(exc).__name__}: {exc}",
        )


async def _run_process(command: list[str]) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout, stderr


async def _probe_video(
    video_path: Path,
    *,
    require_media_probe: bool,
) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        if require_media_probe:
            return _error_result(
                "probe_generated_video",
                expected_status="ffprobe available",
                actual_status="ffprobe missing",
                message="install ffmpeg or pass --skip-media-probe",
            )
        return _success_result(
            "probe_generated_video",
            expected_status="media probe skipped",
            actual_status="SKIP",
            message="ffprobe is unavailable",
            skipped=True,
        )

    return_code, stdout, stderr = await _run_process(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    if return_code != 0:
        return _error_result(
            "probe_generated_video",
            expected_status="decodable video",
            actual_status=f"ffprobe exit {return_code}",
            message=stderr.decode(errors="replace")[:MAX_RESPONSE_EXCERPT],
        )

    try:
        metadata = json.loads(stdout)
        video_stream = next(
            stream
            for stream in metadata["streams"]
            if stream.get("codec_type") == "video"
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(metadata["format"]["duration"])
    except (
        KeyError,
        TypeError,
        ValueError,
        StopIteration,
        json.JSONDecodeError,
    ) as exc:
        return _error_result(
            "probe_generated_video",
            expected_status="valid ffprobe metadata",
            actual_status="invalid metadata",
            message=f"{type(exc).__name__}: {exc}",
        )

    actual_ratio = width / height if height else 0.0
    expected_ratio = 16 / 9
    ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
    if width <= 0 or height <= 0:
        return _error_result(
            "probe_generated_video",
            expected_status="positive video dimensions",
            actual_status=f"{width}x{height}",
            message="video dimensions are invalid",
        )
    if abs(duration - EXPECTED_DURATION_SECONDS) > DURATION_TOLERANCE_SECONDS:
        return _error_result(
            "probe_generated_video",
            expected_status=f"{EXPECTED_DURATION_SECONDS}s video",
            actual_status=f"{duration:.3f}s",
            message="decoded duration is outside tolerance",
        )
    if ratio_error > RATIO_TOLERANCE:
        return _error_result(
            "probe_generated_video",
            expected_status=EXPECTED_RATIO,
            actual_status=f"{width}:{height}",
            message=f"decoded aspect-ratio error is {ratio_error:.3%}",
        )

    return _success_result(
        "probe_generated_video",
        expected_status="decodable video matching duration and ratio",
        actual_status="valid media",
        width=width,
        height=height,
        duration_seconds=duration,
        codec=video_stream.get("codec_name"),
        frame_rate=video_stream.get("r_frame_rate"),
        frame_count=video_stream.get("nb_frames"),
    )


def _mean_absolute_delta(left: bytes, right: bytes) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


async def _check_video_frames(
    video_path: Path,
    *,
    require_frame_checks: bool,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if require_frame_checks:
            return _error_result(
                "check_generated_video_frames",
                expected_status="ffmpeg available",
                actual_status="ffmpeg missing",
                message="install ffmpeg or pass --skip-frame-checks",
            )
        return _success_result(
            "check_generated_video_frames",
            expected_status="frame checks skipped",
            actual_status="SKIP",
            message="ffmpeg is unavailable",
            skipped=True,
        )

    frame_size = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS
    return_code, stdout, stderr = await _run_process(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1,scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    if return_code != 0:
        return _error_result(
            "check_generated_video_frames",
            expected_status="decodable video frames",
            actual_status=f"ffmpeg exit {return_code}",
            message=stderr.decode(errors="replace")[:MAX_RESPONSE_EXCERPT],
        )

    frame_count = len(stdout) // frame_size
    frames = [
        stdout[index * frame_size : (index + 1) * frame_size]
        for index in range(frame_count)
    ]
    if frame_count < 2:
        if require_frame_checks:
            return _error_result(
                "check_generated_video_frames",
                expected_status="at least two sampled frames",
                actual_status=f"{frame_count} frame(s)",
                message="not enough frames were decoded for temporal validation",
            )
        return _success_result(
            "check_generated_video_frames",
            expected_status="frame-quality checks are diagnostic",
            actual_status=f"{frame_count} frame(s)",
            message="not enough frames were decoded for temporal validation",
            quality_warning=True,
        )

    average_brightness = sum(sum(frame) / len(frame) for frame in frames) / len(frames)
    frame_deltas = [
        _mean_absolute_delta(left, right) for left, right in zip(frames, frames[1:])
    ]
    mean_frame_delta = sum(frame_deltas) / len(frame_deltas)
    quality_issues = []
    if average_brightness < MIN_FRAME_BRIGHTNESS:
        quality_issues.append("sampled frames are effectively black")
    if mean_frame_delta < MIN_MEAN_FRAME_DELTA:
        quality_issues.append("sampled frames appear frozen")
    if quality_issues and require_frame_checks:
        return _error_result(
            "check_generated_video_frames",
            expected_status="non-black video with temporal variation",
            actual_status=(
                f"brightness={average_brightness:.3f}, "
                f"mean_delta={mean_frame_delta:.3f}"
            ),
            message="; ".join(quality_issues),
        )

    return _success_result(
        "check_generated_video_frames",
        expected_status=(
            "non-black video with temporal variation"
            if require_frame_checks
            else "decodable sampled frames"
        ),
        actual_status=(
            "valid sampled frames" if not quality_issues else "quality warning"
        ),
        message="; ".join(quality_issues),
        sampled_frames=frame_count,
        average_brightness=round(average_brightness, 3),
        mean_frame_delta=round(mean_frame_delta, 3),
        quality_warning=bool(quality_issues),
    )


async def _delete_succeeded_task(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any]:
    endpoint_url = f"{base_url.rstrip('/')}{DELETE_PATH.format(task_id=task_id)}"
    try:
        async with session.delete(
            endpoint_url,
            headers=_api_headers(api_key),
        ) as response:
            response_text = await response.text()
            data = _decode_json(response_text)
            if response.status != HTTP_OK:
                return _error_result(
                    "delete_succeeded_task",
                    expected_status=HTTP_OK,
                    actual_status=response.status,
                    message=(
                        f"delete failed; response={_response_excerpt(response_text)!r}"
                    ),
                )
            expected = {
                "task_id": task_id,
                "action": "deleted",
                "status": "deleted",
            }
            if not isinstance(data, dict) or any(
                data.get(field) != value for field, value in expected.items()
            ):
                return _error_result(
                    "delete_succeeded_task",
                    expected_status="deleted response",
                    actual_status=response.status,
                    message=f"unexpected response={data!r}",
                )
            return _success_result(
                "delete_succeeded_task",
                expected_status=HTTP_OK,
                actual_status=response.status,
                task_id=task_id,
                action=data["action"],
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return _error_result(
            "delete_succeeded_task",
            expected_status=HTTP_OK,
            actual_status="request_error",
            message=f"{type(exc).__name__}: {exc}",
        )


async def _check_deleted_record(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict[str, Any]:
    endpoint_url = f"{base_url.rstrip('/')}{QUERY_PATH.format(task_id=task_id)}"
    try:
        async with session.get(
            endpoint_url,
            headers=_api_headers(api_key),
        ) as response:
            response_text = await response.text()
            data = _decode_json(response_text)
            if response.status != HTTP_BAD_REQUEST:
                return _error_result(
                    "query_deleted_task_record",
                    expected_status=HTTP_BAD_REQUEST,
                    actual_status=response.status,
                    message=(
                        f"deleted task record is still queryable; "
                        f"response={_response_excerpt(response_text)!r}"
                    ),
                )
            passed, message = _validate_error_envelope(
                data,
                expected_status=HTTP_BAD_REQUEST,
                expected_error_type="bad_request_error",
            )
            if not passed:
                return _error_result(
                    "query_deleted_task_record",
                    expected_status=HTTP_BAD_REQUEST,
                    actual_status=response.status,
                    message=message,
                )
            return _success_result(
                "query_deleted_task_record",
                expected_status=HTTP_BAD_REQUEST,
                actual_status=response.status,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return _error_result(
            "query_deleted_task_record",
            expected_status=HTTP_BAD_REQUEST,
            actual_status="request_error",
            message=f"{type(exc).__name__}: {exc}",
        )


def _summarize_results(
    *,
    base_url: str,
    results: list[dict[str, Any]],
    task_id: str | None,
) -> dict[str, Any]:
    passed = sum(1 for result in results if result["passed"])
    return {
        "base_url": base_url.rstrip("/"),
        "task_name": "minimax_h3_lifecycle_delete",
        "summary": f"{passed}/{len(results)} checks passed",
        "task_id": task_id,
        "detailed_test_results": results,
        "success": bool(results) and passed == len(results),
    }


async def run_lifecycle_delete(
    *,
    base_url: str,
    api_key: str,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    require_media_probe: bool = True,
    require_frame_checks: bool = False,
) -> dict[str, Any]:
    """Run one create-query-download-delete lifecycle against ``base_url``."""

    results: list[dict[str, Any]] = []
    task_id: str | None = None
    timeout = aiohttp.ClientTimeout(total=request_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results.extend(
            await _check_delete_errors(
                session,
                base_url=base_url,
                api_key=api_key,
            )
        )

        create_result, task_id = await _create_task(
            session,
            base_url=base_url,
            api_key=api_key,
        )
        results.append(create_result)
        if task_id is None:
            return _summarize_results(
                base_url=base_url,
                results=results,
                task_id=None,
            )

        poll_result, terminal_task = await _poll_task(
            session,
            base_url=base_url,
            api_key=api_key,
            task_id=task_id,
            poll_interval=poll_interval,
            poll_timeout=poll_timeout,
        )
        results.append(poll_result)
        if terminal_task is None:
            return _summarize_results(
                base_url=base_url,
                results=results,
                task_id=task_id,
            )

        succeeded_result, content_url = _validate_succeeded_task(
            terminal_task,
            task_id=task_id,
        )
        results.append(succeeded_result)

        if content_url is not None:
            with tempfile.TemporaryDirectory(prefix="minimax_h3_") as temp_dir:
                video_path = Path(temp_dir) / f"{task_id}.mp4"
                download_result = await _download_video(
                    content_url=content_url,
                    destination=video_path,
                    download_timeout=download_timeout,
                )
                results.append(download_result)
                if download_result["passed"]:
                    results.append(
                        await _probe_video(
                            video_path,
                            require_media_probe=require_media_probe,
                        )
                    )
                    results.append(
                        await _check_video_frames(
                            video_path,
                            require_frame_checks=require_frame_checks,
                        )
                    )

        if terminal_task.get("status") in {"succeeded", "failed"}:
            delete_result = await _delete_succeeded_task(
                session,
                base_url=base_url,
                api_key=api_key,
                task_id=task_id,
            )
            results.append(delete_result)
            if delete_result["passed"]:
                results.append(
                    await _check_deleted_record(
                        session,
                        base_url=base_url,
                        api_key=api_key,
                        task_id=task_id,
                    )
                )

    return _summarize_results(
        base_url=base_url,
        results=results,
        task_id=task_id,
    )


class MiniMaxH3LifecycleDeleteTest(BaseTest):
    """Workflow-compatible wrapper around the standalone lifecycle test."""

    KIND = "minimax_h3_lifecycle_delete"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        return await run_lifecycle_delete(
            base_url=self.base_url,
            api_key=_resolve_api_key(),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
            download_timeout=float(
                self.targets.get(
                    "download_timeout",
                    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
                )
            ),
            poll_interval=float(
                self.targets.get(
                    "poll_interval",
                    DEFAULT_POLL_INTERVAL_SECONDS,
                )
            ),
            poll_timeout=float(
                self.targets.get(
                    "poll_timeout",
                    DEFAULT_POLL_TIMEOUT_SECONDS,
                )
            ),
            require_media_probe=bool(self.targets.get("require_media_probe", True)),
            require_frame_checks=bool(self.targets.get("require_frame_checks", False)),
        )


def run_minimax_h3_lifecycle_delete(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    """Run the lifecycle/delete test under a workflow ``MediaContext``."""

    test_config = TestConfig(
        {
            "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
            # A retry would create another potentially billable task.
            "retry_attempts": 0,
            "retry_delay": 0,
            "break_on_failure": False,
        }
    )
    return MiniMaxH3LifecycleDeleteTest(
        test_config,
        targets or {},
        ctx=ctx,
    ).run_tests()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MiniMax-H3 create/query/download/delete lifecycle "
            "against an already-running endpoint."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server origin, for example http://127.0.0.1:8001",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=DEFAULT_POLL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--skip-media-probe",
        action="store_true",
        help="Skip ffprobe metadata and decodability checks",
    )
    parser.add_argument(
        "--require-frame-checks",
        action="store_true",
        help="Fail on black or frozen sampled frames instead of reporting a warning",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; stdout is always populated",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_lifecycle_delete(
                base_url=args.base_url,
                api_key=_resolve_api_key(),
                request_timeout=args.request_timeout,
                download_timeout=args.download_timeout,
                poll_interval=args.poll_interval,
                poll_timeout=args.poll_timeout,
                require_media_probe=not args.skip_media_probe,
                require_frame_checks=args.require_frame_checks,
            )
        )
    except Exception as exc:
        logger.exception("MiniMax-H3 lifecycle/delete test could not run")
        result = {
            "task_name": "minimax_h3_lifecycle_delete",
            "success": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

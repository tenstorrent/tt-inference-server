# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Async client for the inference server's MiniMax-H3 video job API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

from .video_generation_routing import FIXTURE_IMAGE_PATH

logger = logging.getLogger(__name__)

CREATE_PATH = "/v1/videos/generations"
# One route per H3 task. A deployment serves one task and answers 422 (string detail) on a
# route it cannot serve; an FL2VA deployment also accepts text-only CREATE_PATH.
CREATE_PATHS = {
    "t2va": CREATE_PATH,
    "fl2va": "/v1/videos/generations/i2v",
    "ref2va": "/v1/videos/generations/ref2va",
}
H3_TASKS = tuple(CREATE_PATHS)
DEFAULT_TASK = "t2va"
# tt-media-server ModelRunners.TT_MINIMAX_H3_<TASK> values: "tt-minimax-h3-" + task.
H3_MODEL_RUNNER_PREFIX = "tt-minimax-h3-"
# Request fields that carry media. A job echoes its request in request_parameters with inline
# base64 replaced by a size note (tt-media-server utils/job_manager.redact_inline_media), so
# echo checks compare only the other fields.
MEDIA_FIELDS = frozenset({"image_prompts", "references"})
# The committed 500x375 JPEG the Wan I2V tests send; within the H3 image card (256-5760 px).
REFERENCE_IMAGE_PATH = Path(__file__).resolve().parents[2] / FIXTURE_IMAGE_PATH
QUERY_PATH = "/v1/videos/generations/{job_id}"
LIST_PATH = "/v1/videos/jobs"
DOWNLOAD_PATH = "/v1/videos/generations/{job_id}/download"
CANCEL_PATH = "/v1/videos/generations/{job_id}/cancel"

DOCUMENTED_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "failed", "cancelled", "cancelling"}
)
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
RESPONSE_EXCERPT_LENGTH = 500
# Below uvicorn's default --timeout-keep-alive (5 s); see MiniMaxH3Client.__aenter__.
CLIENT_KEEPALIVE_SECONDS = 1.0
DEFAULT_API_KEY = "your-secret-key"


class MiniMaxClientError(RuntimeError):
    """Raised when the inference server violates its video job contract."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        task_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.task_id = task_id

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": type(self).__name__,
            "message": str(self),
        }
        if self.status_code is not None:
            data["status_code"] = self.status_code
        if self.response_body:
            data["response_body"] = self.response_body
        if self.task_id:
            data["task_id"] = self.task_id
        return data


class MiniMaxTransportError(MiniMaxClientError):
    """The request never got an HTTP response (timeout, reset, refused)."""


@dataclass(frozen=True)
class MiniMaxTerminalTask:
    """Terminal job metadata plus statuses observed while polling."""

    task_id: str
    task: dict[str, Any]
    observed_statuses: tuple[str, ...]


@dataclass(frozen=True)
class MiniMaxDownload:
    """Metadata for a downloaded generated-video artifact."""

    path: Path
    bytes_downloaded: int
    content_type: str


# Every H3 test (contract, lifecycle, quality, benchmark) reads the key from these names, in
# this order. The benchmark engine mirrors the tuple; a unit test keeps them equal.
API_KEY_ENV_VARS = ("API_KEY", "MINIMAX_API_KEY", "TT_MINIMAX_API_KEY")


def resolve_server_api_key() -> str:
    """Resolve the literal bearer token used by the media server."""

    for env_name in API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return DEFAULT_API_KEY


def resolve_h3_task(ctx: Any = None) -> str:
    """The H3 task the deployment under test serves: t2va, fl2va or ref2va.

    Read from the model spec's MODEL_RUNNER (``ctx.model_spec.env_vars``, the env the
    harness starts the container with) or, without a context (standalone CLI), from this
    process's MODEL_RUNNER. No runner, or a non-H3 one, is t2va; an H3 runner naming an
    unknown task raises rather than run the wrong request shapes.
    """

    if ctx is not None:
        env = getattr(getattr(ctx, "model_spec", None), "env_vars", None) or {}
    else:
        env = os.environ
    runner = str(env.get("MODEL_RUNNER") or "").strip()
    if not runner.startswith(H3_MODEL_RUNNER_PREFIX):
        return DEFAULT_TASK
    task = runner[len(H3_MODEL_RUNNER_PREFIX) :]
    if task not in CREATE_PATHS:
        raise ValueError(
            f"MODEL_RUNNER={runner!r} names no MiniMax-H3 task; expected "
            f"{H3_MODEL_RUNNER_PREFIX}{{{','.join(H3_TASKS)}}}"
        )
    return task


def request_task_for(deployment_task: str) -> str:
    """The request shape the lifecycle, quality and contract tests send to a deployment.

    FL2VA runs text-only requests on the same transformer/ as T2VA and accepts them on
    CREATE_PATH, so it gets the t2va shape (h3-benchmark's FL2VA cases send keyframes);
    Ref2VA refuses text-only, so it gets one reference image.
    """

    if deployment_task not in CREATE_PATHS:
        raise ValueError(f"unknown MiniMax-H3 task {deployment_task!r}")
    return "ref2va" if deployment_task == "ref2va" else "t2va"


@lru_cache(maxsize=1)
def _reference_image_b64() -> str:
    return base64.b64encode(REFERENCE_IMAGE_PATH.read_bytes()).decode("ascii")


def build_create_payload(
    task: str,
    *,
    prompt: str,
    aspect_ratio: str,
    duration_seconds: int,
    seed: int = 0,
) -> dict[str, Any]:
    """The JSON body of one ``task`` job: the shape fields, plus REFERENCE_IMAGE_PATH as the
    first keyframe (fl2va) or as the one reference image (ref2va)."""

    payload: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration_seconds": duration_seconds,
        "seed": seed,
    }
    if task == "fl2va":
        payload["image_prompts"] = [{"image": _reference_image_b64(), "frame_pos": 0}]
    elif task == "ref2va":
        payload["references"] = {"images": [{"b64": _reference_image_b64()}]}
    elif task != "t2va":
        raise ValueError(f"unknown MiniMax-H3 task {task!r}")
    return payload


def echoed_request_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """The fields of ``payload`` a job's request_parameters echoes verbatim."""

    return {key: value for key, value in payload.items() if key not in MEDIA_FIELDS}


class MiniMaxH3Client:
    """Create, poll, download, and cancel inference-server video jobs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        request_timeout: float = 60.0,
        download_timeout: float = 300.0,
        poll_interval: float = 5.0,
        poll_timeout: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.download_timeout = download_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._session: aiohttp.ClientSession | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def __aenter__(self) -> "MiniMaxH3Client":
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        # uvicorn drops an idle keep-alive connection after 5 s, the same as the poll
        # interval. A poll that reuses a connection just as the server closes it can be lost
        # in the Docker port proxy and hang for the whole request timeout: in tt-shield run
        # 36042442377 the second status poll never reached the server (its access log shows
        # the POST and first GET on one connection and nothing after). Retire idle
        # connections well before the server does, so every poll after a sleep reconnects.
        connector = aiohttp.TCPConnector(keepalive_timeout=CLIENT_KEEPALIVE_SECONDS)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def create_video(
        self, payload: dict[str, Any], *, task: str = DEFAULT_TASK
    ) -> str:
        """POST ``payload`` to ``task``'s create route and return the new job id."""

        path = CREATE_PATHS.get(task)
        if path is None:
            raise ValueError(f"unknown MiniMax-H3 task {task!r}")
        status, data, response_text = await self._api_json_request(
            "POST",
            f"{self.base_url}{path}",
            json_payload=payload,
        )
        if status != 202:
            raise MiniMaxClientError(
                f"video job creation ({path}) returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
            )
        task_id = data.get("id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise MiniMaxClientError(
                "video job creation response has no non-empty id",
                status_code=status,
                response_body=_excerpt(response_text),
            )
        return task_id

    async def query_task(self, task_id: str) -> dict[str, Any]:
        status, data, response_text = await self._api_json_request(
            "GET",
            f"{self.base_url}{QUERY_PATH.format(job_id=task_id)}",
        )
        if status != 200:
            raise MiniMaxClientError(
                f"video job query returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        if not isinstance(data, dict):
            raise MiniMaxClientError(
                "video job query response is not an object",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        if data.get("id") != task_id:
            raise MiniMaxClientError(
                f"query returned id={data.get('id')!r}, expected {task_id!r}",
                task_id=task_id,
            )
        if data.get("status") not in DOCUMENTED_STATUSES:
            raise MiniMaxClientError(
                f"query returned unknown status {data.get('status')!r}",
                task_id=task_id,
            )
        return data

    async def list_tasks(self) -> list[dict[str, Any]]:
        """List V1 jobs and validate the shared public metadata shape."""

        status, data, response_text = await self._api_json_request(
            "GET",
            f"{self.base_url}{LIST_PATH}",
        )
        if status != 200:
            raise MiniMaxClientError(
                f"video job list returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
            )
        if not isinstance(data, list):
            raise MiniMaxClientError(
                "video job list response is not an array",
                status_code=status,
                response_body=_excerpt(response_text),
            )

        for index, task in enumerate(data):
            if not isinstance(task, dict):
                raise MiniMaxClientError(
                    f"video job list item {index} is not an object",
                    response_body=_excerpt(response_text),
                )
            if not isinstance(task.get("id"), str) or not task["id"]:
                raise MiniMaxClientError(
                    f"video job list item {index} has no non-empty id"
                )
            if task.get("status") not in DOCUMENTED_STATUSES:
                raise MiniMaxClientError(
                    f"video job list item {index} has unknown status "
                    f"{task.get('status')!r}",
                    task_id=task["id"],
                )
        return data

    async def wait_for_terminal(self, task_id: str) -> MiniMaxTerminalTask:
        started = time.monotonic()
        observed: list[str] = []
        created_at: int | None = None

        transport_failures = 0
        last_transport_error: MiniMaxTransportError | None = None

        while time.monotonic() - started < self.poll_timeout:
            # A status GET that gets no response is not a verdict on the job (a lost poll
            # once cost a completed clip its eval). Keep polling to the deadline; HTTP errors
            # and contract violations still fail at once.
            try:
                task = await self.query_task(task_id)
            except MiniMaxTransportError as exc:
                transport_failures += 1
                last_transport_error = exc
                logger.warning(
                    "Status poll %d for video job %s got no response (%s); still polling",
                    transport_failures,
                    task_id,
                    exc,
                )
                await asyncio.sleep(self.poll_interval)
                continue
            status = str(task["status"])
            if not observed or observed[-1] != status:
                observed.append(status)

            next_created_at = task.get("created_at")
            if not isinstance(next_created_at, int):
                raise MiniMaxClientError(
                    "video job created_at must be a Unix integer",
                    task_id=task_id,
                )
            if created_at is not None and next_created_at != created_at:
                raise MiniMaxClientError(
                    "video job created_at changed between polls",
                    task_id=task_id,
                )
            created_at = next_created_at

            if status in TERMINAL_STATUSES:
                completed_at = task.get("completed_at")
                if not isinstance(completed_at, int) or completed_at < created_at:
                    raise MiniMaxClientError(
                        "terminal video job has an invalid completed_at",
                        task_id=task_id,
                    )
                return MiniMaxTerminalTask(
                    task_id=task_id,
                    task=task,
                    observed_statuses=tuple(observed),
                )
            await asyncio.sleep(self.poll_interval)

        message = f"video job did not finish within {self.poll_timeout:.1f} seconds"
        if last_transport_error is not None:
            message += (
                f" ({transport_failures} status poll(s) got no response; last: "
                f"{last_transport_error})"
            )
        raise MiniMaxClientError(message, task_id=task_id)

    async def download_video(
        self,
        task_id: str,
        destination: Path,
    ) -> MiniMaxDownload:
        destination.parent.mkdir(parents=True, exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=self.download_timeout)
        total_bytes = 0
        first_bytes = b""
        url = f"{self.base_url}{DOWNLOAD_PATH.format(job_id=task_id)}"

        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers,
            ) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise MiniMaxClientError(
                            f"video download returned HTTP {response.status}",
                            status_code=response.status,
                            response_body=_excerpt(response_text),
                            task_id=task_id,
                        )
                    with destination.open("wb") as output:
                        async for chunk in response.content.iter_chunked(
                            DOWNLOAD_CHUNK_BYTES
                        ):
                            if not chunk:
                                continue
                            if len(first_bytes) < 64:
                                first_bytes += chunk[: 64 - len(first_bytes)]
                            output.write(chunk)
                            total_bytes += len(chunk)
                    content_type = response.headers.get("Content-Type", "")
        except MiniMaxClientError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise MiniMaxClientError(
                f"video download failed: {type(exc).__name__}: {exc}",
                task_id=task_id,
            ) from exc

        if total_bytes == 0:
            raise MiniMaxClientError(
                "video download returned zero bytes",
                task_id=task_id,
            )
        if b"ftyp" not in first_bytes:
            raise MiniMaxClientError(
                "downloaded output has no MP4 ftyp signature",
                task_id=task_id,
            )
        return MiniMaxDownload(
            path=destination,
            bytes_downloaded=total_bytes,
            content_type=content_type,
        )

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        status, data, response_text = await self._api_json_request(
            "POST",
            f"{self.base_url}{CANCEL_PATH.format(job_id=task_id)}",
        )
        if status != 200 or not isinstance(data, dict):
            raise MiniMaxClientError(
                f"video job cancellation returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        if data.get("id") != task_id or data.get("status") not in {
            "cancelled",
            "cancelling",
        }:
            raise MiniMaxClientError(
                f"unexpected cancellation response: {data!r}",
                task_id=task_id,
            )
        return data

    async def _api_json_request(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> tuple[int, Any, str]:
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "MiniMaxH3Client must be used as an async context manager"
            )

        try:
            async with self._session.request(
                method,
                url,
                headers=self.headers,
                json=json_payload,
            ) as response:
                response_text = await response.text()
                return response.status, _decode_json(response_text), response_text
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise MiniMaxTransportError(
                f"{method} {url} failed: {type(exc).__name__}: {exc}"
            ) from exc


def _decode_json(response_text: str) -> Any:
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return None


def _excerpt(response_text: str) -> str:
    return response_text.replace("\n", " ")[:RESPONSE_EXCERPT_LENGTH]


__all__ = [
    "CREATE_PATH",
    "CREATE_PATHS",
    "H3_TASKS",
    "LIST_PATH",
    "MiniMaxClientError",
    "MiniMaxDownload",
    "MiniMaxH3Client",
    "MiniMaxTerminalTask",
    "MiniMaxTransportError",
    "build_create_payload",
    "echoed_request_fields",
    "request_task_for",
    "resolve_h3_task",
    "resolve_server_api_key",
]

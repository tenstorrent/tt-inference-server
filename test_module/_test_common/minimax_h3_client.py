# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Async client for the inference server's MiniMax-H3 video job API."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

CREATE_PATH = "/v1/videos/generations"
QUERY_PATH = "/v1/videos/generations/{job_id}"
DOWNLOAD_PATH = "/v1/videos/generations/{job_id}/download"
CANCEL_PATH = "/v1/videos/generations/{job_id}/cancel"

DOCUMENTED_STATUSES = frozenset(
    {"queued", "in_progress", "completed", "failed", "cancelled", "cancelling"}
)
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
RESPONSE_EXCERPT_LENGTH = 500
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


def resolve_server_api_key() -> str:
    """Resolve the literal bearer token used by the media server."""

    for env_name in ("API_KEY", "MINIMAX_API_KEY", "MINIMAX_MOCK_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    return DEFAULT_API_KEY


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
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def create_video(self, payload: dict[str, Any]) -> str:
        status, data, response_text = await self._api_json_request(
            "POST",
            f"{self.base_url}{CREATE_PATH}",
            json_payload=payload,
        )
        if status != 202:
            raise MiniMaxClientError(
                f"video job creation returned HTTP {status}",
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

    async def wait_for_terminal(self, task_id: str) -> MiniMaxTerminalTask:
        started = time.monotonic()
        observed: list[str] = []
        created_at: int | None = None

        while time.monotonic() - started < self.poll_timeout:
            task = await self.query_task(task_id)
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

        raise MiniMaxClientError(
            f"video job did not finish within {self.poll_timeout:.1f} seconds",
            task_id=task_id,
        )

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
            raise MiniMaxClientError(
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
    "MiniMaxClientError",
    "MiniMaxDownload",
    "MiniMaxH3Client",
    "MiniMaxTerminalTask",
    "resolve_server_api_key",
]

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Small async client for the documented MiniMax-H3 video V2 lifecycle."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp  # pyright: ignore[reportMissingImports]

CREATE_PATH = "/v2/video_generation"
QUERY_PATH = "/v2/query/video_generation/{task_id}"
DELETE_PATH = "/v2/video_generation/{task_id}"

DOCUMENTED_STATUSES = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled"}
)
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
RESPONSE_EXCERPT_LENGTH = 500


class MiniMaxClientError(RuntimeError):
    """Raised when the provider response violates the documented contract."""

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
    """Terminal query response plus the statuses observed while polling."""

    task_id: str
    task: dict[str, Any]
    observed_statuses: tuple[str, ...]


@dataclass(frozen=True)
class MiniMaxDownload:
    """Metadata for a downloaded generated-video artifact."""

    path: Path
    bytes_downloaded: int
    content_type: str


class MiniMaxH3Client:
    """Create, poll, download, and delete MiniMax-H3 video tasks.

    API requests carry the Bearer token explicitly. Output downloads use a
    separate unauthenticated session so a provider key is never forwarded to a
    CDN named by ``content.url``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        request_timeout: float = 60.0,
        download_timeout: float = 300.0,
        poll_interval: float = 5.0,
        poll_timeout: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.download_timeout = download_timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._session: aiohttp.ClientSession | None = None

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
        if status != 200:
            raise MiniMaxClientError(
                f"video task creation returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
            )
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise MiniMaxClientError(
                "video task creation response has no non-empty task_id",
                status_code=status,
                response_body=_excerpt(response_text),
            )
        return task_id

    async def query_task(self, task_id: str) -> dict[str, Any]:
        status, data, response_text = await self._api_json_request(
            "GET",
            f"{self.base_url}{QUERY_PATH.format(task_id=task_id)}",
        )
        if status != 200:
            raise MiniMaxClientError(
                f"task query returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        task = data.get("task") if isinstance(data, dict) else None
        if not isinstance(task, dict):
            raise MiniMaxClientError(
                "task query response has no task object",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        if task.get("id") != task_id:
            raise MiniMaxClientError(
                f"query returned task.id={task.get('id')!r}, expected {task_id!r}",
                task_id=task_id,
            )
        if task.get("status") not in DOCUMENTED_STATUSES:
            raise MiniMaxClientError(
                f"query returned undocumented status {task.get('status')!r}",
                task_id=task_id,
            )
        return task

    async def wait_for_terminal(self, task_id: str) -> MiniMaxTerminalTask:
        started = time.monotonic()
        observed: list[str] = []
        created_at: int | None = None
        updated_at: int | None = None

        while time.monotonic() - started < self.poll_timeout:
            task = await self.query_task(task_id)
            status = str(task["status"])
            if not observed or observed[-1] != status:
                observed.append(status)

            next_created_at = task.get("created_at")
            next_updated_at = task.get("updated_at")
            if not isinstance(next_created_at, int) or not isinstance(
                next_updated_at, int
            ):
                raise MiniMaxClientError(
                    "task timestamps must be Unix integers",
                    task_id=task_id,
                )
            if next_created_at > next_updated_at:
                raise MiniMaxClientError(
                    "task created_at is later than updated_at",
                    task_id=task_id,
                )
            if created_at is not None and next_created_at != created_at:
                raise MiniMaxClientError(
                    "task created_at changed between polls",
                    task_id=task_id,
                )
            if updated_at is not None and next_updated_at < updated_at:
                raise MiniMaxClientError(
                    "task updated_at moved backwards",
                    task_id=task_id,
                )
            created_at = next_created_at
            updated_at = next_updated_at

            if status in TERMINAL_STATUSES:
                return MiniMaxTerminalTask(
                    task_id=task_id,
                    task=task,
                    observed_statuses=tuple(observed),
                )
            await asyncio.sleep(self.poll_interval)

        raise MiniMaxClientError(
            f"task did not finish within {self.poll_timeout:.1f} seconds",
            task_id=task_id,
        )

    async def download_video(
        self, content_url: str, destination: Path
    ) -> MiniMaxDownload:
        parsed = urlparse(content_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise MiniMaxClientError(
                "content.url is not an absolute HTTP(S) URL",
                response_body=content_url,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=self.download_timeout)
        total_bytes = 0
        first_bytes = b""
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                async with session.get(content_url) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        raise MiniMaxClientError(
                            f"video download returned HTTP {response.status}",
                            status_code=response.status,
                            response_body=_excerpt(response_text),
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
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            if isinstance(exc, MiniMaxClientError):
                raise
            raise MiniMaxClientError(
                f"video download failed: {type(exc).__name__}: {exc}"
            ) from exc

        if total_bytes == 0:
            raise MiniMaxClientError("video download returned zero bytes")
        if b"ftyp" not in first_bytes:
            raise MiniMaxClientError("downloaded output has no MP4 ftyp signature")
        return MiniMaxDownload(
            path=destination,
            bytes_downloaded=total_bytes,
            content_type=content_type,
        )

    async def delete_terminal_task(self, task_id: str) -> dict[str, Any]:
        status, data, response_text = await self._api_json_request(
            "DELETE",
            f"{self.base_url}{DELETE_PATH.format(task_id=task_id)}",
        )
        if status != 200:
            raise MiniMaxClientError(
                f"task deletion returned HTTP {status}",
                status_code=status,
                response_body=_excerpt(response_text),
                task_id=task_id,
            )
        if not isinstance(data, dict):
            raise MiniMaxClientError(
                "task deletion response is not a JSON object",
                task_id=task_id,
            )
        expected = {
            "task_id": task_id,
            "action": "deleted",
            "status": "deleted",
        }
        mismatches = {
            field: {"expected": value, "actual": data.get(field)}
            for field, value in expected.items()
            if data.get(field) != value
        }
        if mismatches:
            raise MiniMaxClientError(
                f"unexpected task deletion response: {mismatches}",
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

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            async with self._session.request(
                method,
                url,
                headers=headers,
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
    "MiniMaxClientError",
    "MiniMaxDownload",
    "MiniMaxH3Client",
    "MiniMaxTerminalTask",
]

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 client polling: a status GET that gets no response is retried until the
deadline, while HTTP errors and contract violations still end the wait at once."""

from __future__ import annotations

import asyncio
import logging

import pytest

from test_module._test_common import minimax_h3_client as M

JOB_ID = "2e4bae2d-c22d-4e81-bbc5-b5e215d3d7e6"


def _task(status, **extra):
    task = {"id": JOB_ID, "status": status, "created_at": 1790000000}
    if status in M.TERMINAL_STATUSES:
        task["completed_at"] = 1790000090
    task.update(extra)
    return task


class _ScriptedClient(M.MiniMaxH3Client):
    """query_task replays a script: a dict is returned, an exception is raised."""

    def __init__(self, script, **kwargs):
        super().__init__(
            base_url="http://127.0.0.1:8000", api_key="k", poll_interval=0, **kwargs
        )
        self.script = list(script)
        self.calls = 0

    async def query_task(self, task_id):
        self.calls += 1
        step = self.script.pop(0) if self.script else self.last
        self.last = step
        if isinstance(step, BaseException):
            raise step
        return step


def _no_response():
    return M.MiniMaxTransportError(
        f"GET http://127.0.0.1:8000/v1/videos/generations/{JOB_ID} failed: TimeoutError: "
    )


def test_status_timeout_is_retried_until_the_job_completes(caplog):
    client = _ScriptedClient(
        [_task("queued"), _no_response(), _task("in_progress"), _task("completed")]
    )
    with caplog.at_level(logging.WARNING, logger=M.logger.name):
        terminal = asyncio.run(client.wait_for_terminal(JOB_ID))
    assert terminal.task["status"] == "completed"
    assert terminal.observed_statuses == ("queued", "in_progress", "completed")
    assert "Status poll 1 for video job" in caplog.text


def test_http_error_still_fails_at_once():
    client = _ScriptedClient(
        [
            _task("queued"),
            M.MiniMaxClientError("video job query returned HTTP 500", status_code=500),
            _task("completed"),
        ]
    )
    with pytest.raises(M.MiniMaxClientError, match="HTTP 500"):
        asyncio.run(client.wait_for_terminal(JOB_ID))
    assert client.calls == 2


def test_deadline_reports_the_unanswered_polls():
    client = _ScriptedClient([_task("queued"), _no_response()], poll_timeout=0.05)
    with pytest.raises(M.MiniMaxClientError) as excinfo:
        asyncio.run(client.wait_for_terminal(JOB_ID))
    assert not isinstance(excinfo.value, M.MiniMaxTransportError)
    assert "status poll(s) got no response" in str(excinfo.value)
    assert "TimeoutError" in str(excinfo.value)


def test_transport_failure_is_a_client_error():
    # Callers that catch MiniMaxClientError (every H3 test) keep catching it.
    assert issubclass(M.MiniMaxTransportError, M.MiniMaxClientError)


def test_request_without_response_raises_transport_error():
    async def run():
        async with M.MiniMaxH3Client(
            base_url="http://127.0.0.1:9", api_key="k", request_timeout=2
        ) as client:
            await client.query_task(JOB_ID)

    with pytest.raises(M.MiniMaxTransportError):
        asyncio.run(run())


def test_idle_connections_retire_before_the_server_drops_them():
    # uvicorn closes idle keep-alive connections after 5 s and the client polls every 5 s;
    # a reused connection must never outlive the server's side of it.
    async def run():
        async with M.MiniMaxH3Client(
            base_url="http://127.0.0.1:8000", api_key="k"
        ) as c:
            return c._session.connector._keepalive_timeout

    keepalive = asyncio.run(run())
    assert keepalive == M.CLIENT_KEEPALIVE_SECONDS
    assert keepalive < 5.0

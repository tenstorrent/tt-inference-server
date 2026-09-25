# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 lifecycle/download and cancel tests create their job in the request shape of
the task the spec deploys, and check only the request fields the server echoes verbatim
(inline media comes back redacted)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module._test_common import minimax_h3_client as M
from test_module.load_param_tests import minimax_h3_cancel_lifecycle_test as K
from test_module.load_param_tests import minimax_h3_lifecycle_delete_test as L

JOB_ID = "2e4bae2d-c22d-4e81-bbc5-b5e215d3d7e6"


def _redacted(value):
    """What tt-media-server's redact_inline_media does to a request dump."""
    if isinstance(value, dict):
        return {
            key: (
                f"<inline media omitted: {len(item)} base64 chars>"
                if key in ("image", "b64") and isinstance(item, str) and len(item) > 256
                else _redacted(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


class _FakeClient:
    """Stands in for MiniMaxH3Client: one job that completes, echoing its request."""

    instances: list = []

    def __init__(self, **kwargs):
        self.created = []
        self.job = None
        _FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def create_video(self, payload, *, task="t2va"):
        self.created.append((task, payload))
        self.job = {
            "id": JOB_ID,
            "job_type": "video",
            "status": "queued",
            "created_at": 1790000000,
            "request_parameters": {**_redacted(payload), "num_inference_steps": 50},
        }
        return JOB_ID

    async def query_task(self, task_id):
        return dict(self.job)

    async def list_tasks(self):
        return [dict(self.job)]

    async def wait_for_terminal(self, task_id):
        if self.job["status"] != "cancelled":
            self.job.update(status="completed", completed_at=1790000090)
        return M.MiniMaxTerminalTask(
            task_id=task_id, task=dict(self.job), observed_statuses=("completed",)
        )

    async def download_video(self, task_id, destination):
        return M.MiniMaxDownload(
            path=Path(destination), bytes_downloaded=1, content_type="video/mp4"
        )

    async def cancel_task(self, task_id):
        self.job.update(status="cancelled", completed_at=1790000001)
        return {"id": task_id, "status": "cancelled"}


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances.clear()
    monkeypatch.setattr(L, "MiniMaxH3Client", _FakeClient)
    monkeypatch.setattr(K, "MiniMaxH3Client", _FakeClient)
    monkeypatch.setattr(
        L, "analyze_video_quality", lambda *a, **k: {"valid_video": True}
    )
    monkeypatch.setattr(L, "_probe_non_silent_audio", lambda path: {"rms_pcm16": 100.0})
    return _FakeClient


CASES = [
    ("t2va", "t2va", set()),
    # FL2VA serves text-only /generations, the t2va shape.
    ("fl2va", "t2va", set()),
    ("ref2va", "ref2va", {"references"}),
]


@pytest.mark.parametrize(("task", "request_task", "media"), CASES)
def test_lifecycle_creates_in_the_request_shape(
    task, request_task, media, fake_client, tmp_path
):
    result = asyncio.run(
        L.run_lifecycle_download(
            base_url="http://127.0.0.1:8000",
            api_key="k",
            output_path=tmp_path / "out.mp4",
            task=task,
        )
    )
    ((created_task, payload),) = fake_client.instances[0].created
    assert created_task == request_task
    assert {"references", "image_prompts"} & set(payload) == media
    assert payload["prompt"] == L.PROMPT
    assert result["success"] is True
    assert (result["deployment_task"], result["request_task"]) == (task, request_task)


def test_redacted_references_do_not_fail_the_echo_check():
    payload = L._create_payload("ref2va")
    task = {
        "id": JOB_ID,
        "job_type": "video",
        "request_parameters": _redacted(payload),
    }
    assert task["request_parameters"]["references"] != payload["references"]
    L._validate_job_metadata(task, task_id=JOB_ID)


def test_echo_mismatch_on_a_shape_field_still_fails():
    task = {
        "id": JOB_ID,
        "job_type": "video",
        "request_parameters": {**L._create_payload(), "duration_seconds": 10},
    }
    with pytest.raises(M.MiniMaxClientError, match="duration_seconds"):
        L._validate_job_metadata(task, task_id=JOB_ID)


@pytest.mark.parametrize(("task", "request_task", "media"), CASES)
def test_cancel_lifecycle_creates_in_the_request_shape(
    task, request_task, media, fake_client
):
    result = asyncio.run(
        K.run_cancel_lifecycle(base_url="http://127.0.0.1:8000", api_key="k", task=task)
    )
    ((created_task, payload),) = fake_client.instances[0].created
    assert created_task == request_task
    assert {"references", "image_prompts"} & set(payload) == media
    assert (result["deployment_task"], result["request_task"]) == (task, request_task)


def test_workflow_test_passes_the_spec_task(monkeypatch, tmp_path):
    seen = {}

    async def fake_run(**kwargs):
        seen.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(L, "run_lifecycle_download", fake_run)
    ctx = SimpleNamespace(
        service_port=8000,
        base_url="http://127.0.0.1:8000",
        output_path=str(tmp_path),
        model_spec=SimpleNamespace(env_vars={"MODEL_RUNNER": "tt-minimax-h3-ref2va"}),
    )
    test = L.MiniMaxH3LifecycleDownloadTest(
        L.TestConfig({"timeout": 5, "retry_attempts": 0, "retry_delay": 0}), {}, ctx=ctx
    )
    asyncio.run(test._run_specific_test_async())
    assert seen["task"] == "ref2va"

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 task routing shared by the non-benchmark tests: the task comes from the
spec's MODEL_RUNNER, and each task gets its own route and request shape."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from test_module._test_common import minimax_h3_client as M
from test_module._test_common.minimax_h3_bench import adapters
from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path


def _ctx(env_vars=None):
    return SimpleNamespace(model_spec=SimpleNamespace(env_vars=env_vars or {}))


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_task_comes_from_the_spec_model_runner(task, monkeypatch):
    # The spec wins over whatever this process happens to export.
    monkeypatch.setenv("MODEL_RUNNER", "tt-minimax-h3-t2va")
    assert M.resolve_h3_task(_ctx({"MODEL_RUNNER": f"tt-minimax-h3-{task}"})) == task


@pytest.mark.parametrize(
    "ctx",
    [
        _ctx(),
        _ctx({"MODEL_RUNNER": ""}),
        _ctx({"MODEL_RUNNER": "tt-wan2.2"}),
        SimpleNamespace(model_spec=SimpleNamespace(model_name="MiniMax-H3")),
    ],
    ids=["no-runner", "empty-runner", "non-h3-runner", "spec-without-env"],
)
def test_no_h3_runner_is_t2va(ctx):
    assert M.resolve_h3_task(ctx) == "t2va"


def test_standalone_reads_the_process_model_runner(monkeypatch):
    monkeypatch.setenv("MODEL_RUNNER", "tt-minimax-h3-ref2va")
    assert M.resolve_h3_task() == "ref2va"
    monkeypatch.delenv("MODEL_RUNNER")
    assert M.resolve_h3_task() == "t2va"


def test_unknown_h3_runner_task_raises():
    with pytest.raises(ValueError, match="names no MiniMax-H3 task"):
        M.resolve_h3_task(_ctx({"MODEL_RUNNER": "tt-minimax-h3-i2v"}))


def test_the_dev_spec_runner_resolves_to_a_task():
    # Whatever task a branch deploys, its spec (defaults + device env merged) must name one.
    path = get_repo_root_path() / "workflows" / "model_specs" / "dev" / "video.yaml"
    template = next(
        t
        for t in load_templates_from_yaml(path)
        if t.weights == ["MiniMaxAI/MiniMax-H3"]
    )
    for spec in template.expand_to_specs():
        runner = spec.env_vars["MODEL_RUNNER"]
        task = M.resolve_h3_task(SimpleNamespace(model_spec=spec))
        assert task in M.H3_TASKS
        assert runner == M.H3_MODEL_RUNNER_PREFIX + task


def test_create_routes_match_the_benchmark_engine():
    assert M.CREATE_PATHS == adapters.ROUTE
    assert M.H3_TASKS == adapters.TASKS


@pytest.mark.parametrize(
    ("deployment", "request_task"),
    [("t2va", "t2va"), ("fl2va", "t2va"), ("ref2va", "ref2va")],
)
def test_request_shape_per_deployment(deployment, request_task):
    # FL2VA also serves text-only /generations; Ref2VA refuses it.
    assert M.request_task_for(deployment) == request_task


def test_request_task_for_unknown_task_raises():
    with pytest.raises(ValueError):
        M.request_task_for("i2v")


def _payload(task):
    return M.build_create_payload(
        task, prompt="a fox", aspect_ratio="16:9", duration_seconds=5
    )


SHAPE_FIELDS = {
    "prompt": "a fox",
    "aspect_ratio": "16:9",
    "duration_seconds": 5,
    "seed": 0,
}


def test_t2va_payload_is_the_shape_fields_only():
    assert _payload("t2va") == SHAPE_FIELDS


def test_fl2va_payload_adds_the_image_as_first_keyframe():
    payload = _payload("fl2va")
    (keyframe,) = payload.pop("image_prompts")
    assert payload == SHAPE_FIELDS
    assert keyframe["frame_pos"] == 0
    assert base64.b64decode(keyframe["image"]) == M.REFERENCE_IMAGE_PATH.read_bytes()


def test_ref2va_payload_adds_one_reference_image():
    payload = _payload("ref2va")
    references = payload.pop("references")
    assert payload == SHAPE_FIELDS
    assert list(references) == ["images"]
    (image,) = references["images"]
    assert list(image) == ["b64"]
    assert base64.b64decode(image["b64"]) == M.REFERENCE_IMAGE_PATH.read_bytes()


def test_unknown_payload_task_raises():
    with pytest.raises(ValueError):
        _payload("i2v")


def test_reference_image_is_committed_and_within_the_h3_image_card():
    from PIL import Image

    # A tracked file (so CI has it) inside the card tt-media-server admits: <= 30 MB,
    # JPG/PNG/WEBP, each side in [256, 5760] px, width/height in [0.4, 2.5].
    raw = M.REFERENCE_IMAGE_PATH.read_bytes()
    assert len(raw) < 1024 * 1024
    with Image.open(M.REFERENCE_IMAGE_PATH) as image:
        width, height = image.size
        assert image.format == "JPEG"
    assert 256 <= width <= 5760 and 256 <= height <= 5760
    assert 0.4 <= width / height <= 2.5


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
def test_echoed_fields_drop_only_the_media(task):
    assert M.echoed_request_fields(_payload(task)) == SHAPE_FIELDS


class _RecordingClient(M.MiniMaxH3Client):
    def __init__(self):
        super().__init__(base_url="http://127.0.0.1:8000/", api_key="k")
        self.urls = []

    async def _api_json_request(self, method, url, *, json_payload=None):
        self.urls.append((method, url))
        return 202, {"id": "job-1"}, '{"id": "job-1"}'


@pytest.mark.parametrize(
    ("task", "path"),
    [
        (None, "/v1/videos/generations"),
        ("t2va", "/v1/videos/generations"),
        ("fl2va", "/v1/videos/generations/i2v"),
        ("ref2va", "/v1/videos/generations/ref2va"),
    ],
)
def test_create_video_posts_to_the_task_route(task, path):
    client = _RecordingClient()
    kwargs = {} if task is None else {"task": task}
    assert asyncio.run(client.create_video({"prompt": "x"}, **kwargs)) == "job-1"
    assert client.urls == [("POST", f"http://127.0.0.1:8000{path}")]


def test_create_video_refuses_an_unknown_task_before_sending():
    client = _RecordingClient()
    with pytest.raises(ValueError):
        asyncio.run(client.create_video({"prompt": "x"}, task="i2v"))
    assert client.urls == []


def test_create_error_names_the_route():
    class _Refused(_RecordingClient):
        async def _api_json_request(self, method, url, *, json_payload=None):
            return 422, {"detail": "This deployment ..."}, '{"detail": "..."}'

    with pytest.raises(M.MiniMaxClientError) as excinfo:
        asyncio.run(_Refused().create_video({"prompt": "x"}, task="ref2va"))
    assert "/v1/videos/generations/ref2va" in str(excinfo.value)
    assert excinfo.value.status_code == 422

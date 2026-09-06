# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""MiniMax-H3 admission on the SP (multi-host) deployment: contract tests that currently FAIL.

On OM Quad1 the API process runs with ``MODEL_RUNNER=sp_runner`` (the device workers are separate
``tt-minimax-h3-*`` processes reached over shared memory) and ``MODEL=MiniMax-H3[-FL2VA|-Ref2VA]``.
``domain.video_generate_request._is_minimax_h3`` keys on ``model_runner`` alone, so in that process
every H3-specific validator is switched off: an out-of-policy ``duration_seconds`` or
``aspect_ratio`` and unknown fields such as ``duration`` are accepted with 202 and only fail later
inside the worker as a *failed job* -- exactly the "caller believes it asked for something it did
not get" outcome the validators exist to prevent (observed on quad1 2026-09-05: ``duration: 9`` was
accepted; ``duration_seconds: 9`` reached the worker only after the side-file fix).

These tests pin the desired behaviour -- the SP frontend must recognise an H3 deployment from
``MODEL`` too -- and are ``xfail(strict=True)`` so they flip to XPASS (and fail the run, asking to
be un-marked) the moment the gate is fixed.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from domain.video_generate_request import VideoGenerateRequest
from pydantic import ValidationError

SP_API_PROCESS = SimpleNamespace(model_runner="sp_runner", model="MiniMax-H3-FL2VA")
SP_GATE_OPEN = pytest.mark.xfail(
    strict=True,
    reason="_is_minimax_h3() keys on settings.model_runner, which is 'sp_runner' in the quad's API "
    "process; H3 admission validators never run there and bad requests become failed jobs",
)


@pytest.fixture
def sp_api_settings():
    with patch("domain.video_generate_request.get_settings", return_value=SP_API_PROCESS):
        yield


def _valid(**over) -> dict:
    body = {"prompt": "A fox runs through wet grass.", "aspect_ratio": "16:9", "duration_seconds": 5}
    body.update(over)
    return body


@pytest.mark.usefixtures("sp_api_settings")
def test_sp_api_process_accepts_a_valid_h3_request():
    """Control: the happy path must keep working whatever the gate does."""
    req = VideoGenerateRequest(**_valid())
    assert req.duration_seconds == 5 and req.aspect_ratio == "16:9"


@SP_GATE_OPEN
@pytest.mark.usefixtures("sp_api_settings")
@pytest.mark.parametrize(("field", "value"), [("duration_seconds", 3), ("duration_seconds", 16), ("aspect_ratio", "2:1")])
def test_sp_api_process_rejects_out_of_policy_shape(field, value):
    with pytest.raises(ValidationError):
        VideoGenerateRequest(**_valid(**{field: value}))


@SP_GATE_OPEN
@pytest.mark.usefixtures("sp_api_settings")
@pytest.mark.parametrize("unknown", [{"duration": 9}, {"resolution": "1080P"}, {"model": "NotMiniMax"}])
def test_sp_api_process_rejects_unknown_fields(unknown):
    with pytest.raises(ValidationError):
        VideoGenerateRequest(**_valid(**unknown))


def test_h3_worker_process_still_rejects_out_of_policy_shape():
    """The same validators do fire when model_runner names an H3 runner (regression guard)."""
    worker = SimpleNamespace(model_runner="tt-minimax-h3-t2va", model="MiniMax-H3")
    with patch("domain.video_generate_request.get_settings", return_value=worker):
        with pytest.raises(ValidationError):
            VideoGenerateRequest(**_valid(duration_seconds=3))
        with pytest.raises(ValidationError):
            VideoGenerateRequest(**_valid(duration=9))

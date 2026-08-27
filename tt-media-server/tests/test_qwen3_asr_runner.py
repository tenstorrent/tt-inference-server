# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Unit tests for the Qwen3-ASR media-server runner's segment handling.

The runner hard-imports the tt-metal qwen3_asr demo tree at module load, so we
stub that tree (ttnn/torch mocks come from conftest) and re-import the real
module -- mirroring tests/test_whisper_chunk_metrics.py. The method under test is
bound onto a MagicMock instance, so no device or model weights are needed.
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

SR = 16000


def _install_stubs():
    """tt-metal demo modules the runner imports off its sys.path, plus the
    tt_transformers config + safetensors it pulls in. Stubbed so the import
    succeeds without a tt-metal checkout."""
    stubs = {}

    def _mod(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        stubs[name] = module

    _mod("audio_encoder")
    _mod("audio_encoder_ref")
    _mod("transcribe")
    _mod("qwen3_asr_decoder", Qwen3ASRDecoder=MagicMock())
    _mod("safetensors", safe_open=MagicMock())
    _mod("models")
    _mod("models.tt_transformers")
    _mod("models.tt_transformers.tt")
    _mod("models.tt_transformers.tt.model_config", ModelArgs=MagicMock())
    return stubs


@pytest.fixture
def qwen_runner_module(tmp_path):
    # _qwen_demo_root() returns the first candidate dir that has a demo/ subdir;
    # point TT_METAL_HOME at a temp tree so the module import resolves it.
    (tmp_path / "models" / "demos" / "audio" / "qwen3_asr" / "demo").mkdir(parents=True)
    saved_env = os.environ.get("TT_METAL_HOME")
    os.environ["TT_METAL_HOME"] = str(tmp_path)

    stubs = _install_stubs()
    saved_modules = {
        name: sys.modules.get(name)
        for name in list(stubs)
        + [n for n in sys.modules if n.startswith("tt_model_runners")]
    }
    sys.modules.update(stubs)
    for name in list(sys.modules):
        if name.startswith("tt_model_runners"):
            sys.modules.pop(name, None)
    try:
        yield importlib.import_module("tt_model_runners.qwen3_asr_runner")
    finally:
        for name in list(sys.modules):
            if name.startswith("tt_model_runners"):
                sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_env is None:
            os.environ.pop("TT_METAL_HOME", None)
        else:
            os.environ["TT_METAL_HOME"] = saved_env


def _bound_run_segments(module, infer_capture):
    """Bind the real _run_segments onto a MagicMock whose _infer_long records the
    audio it receives and returns a fixed transcription."""
    runner = MagicMock()

    def fake_infer_long(wav):
        infer_capture.append(np.asarray(wav))
        return ("HELLO", 3, 0.1)

    runner._infer_long = fake_infer_long
    return module.TTQwen3AsrRunner._run_segments.__get__(runner)


def _seg_request(segments, audio_len_s):
    request = MagicMock()
    request._segments = segments
    request._audio_array = np.zeros(int(audio_len_s * SR), dtype=np.float32)
    request._duration = float(audio_len_s)
    return request


def test_nonzero_offset_segment_is_not_dropped(qwen_runner_module):
    """Regression for the double-slice bug: AudioService.create_segment_request()
    already crops _audio_array to the segment before dispatch, so the runner must
    consume it as-is. The old code re-sliced [15s:28s] out of an already-13s clip
    -> empty -> `continue` -> the segment silently vanished."""
    captured = []
    run_segments = _bound_run_segments(qwen_runner_module, captured)
    request = _seg_request(
        [{"start": 15.0, "end": 28.0, "speaker": "SPEAKER_00"}], audio_len_s=13.0
    )

    responses = run_segments(request)

    assert len(captured) == 1, "segment was dropped (double-slice regression)"
    assert len(captured[0]) == int(13.0 * SR), (
        "runner must transcribe the pre-cropped array, not a re-sliced empty window"
    )
    resp = responses[0]
    assert resp.text == "HELLO"
    assert resp.segments[0].start_time == 15.0
    assert resp.segments[0].end_time == 28.0


def test_zero_offset_segment_still_works(qwen_runner_module):
    """A segment starting at t=0 must keep working after the fix."""
    captured = []
    run_segments = _bound_run_segments(qwen_runner_module, captured)
    request = _seg_request(
        [{"start": 0.0, "end": 12.0, "speaker": "SPEAKER_00"}], audio_len_s=12.0
    )

    responses = run_segments(request)

    assert len(captured) == 1
    assert len(captured[0]) == int(12.0 * SR)
    assert responses[0].text == "HELLO"


def test_all_segments_transcribed_none_dropped(qwen_runner_module):
    """Every segment is transcribed and speaker labels/spans preserved -- no
    segment disappears due to its file offset."""
    captured = []
    run_segments = _bound_run_segments(qwen_runner_module, captured)
    request = _seg_request(
        [
            {"start": 0.0, "end": 12.0, "speaker": "SPEAKER_00"},
            {"start": 15.0, "end": 28.0, "speaker": "SPEAKER_01"},
            {"start": 32.0, "end": 45.0, "speaker": "SPEAKER_00"},
        ],
        audio_len_s=13.0,
    )

    responses = run_segments(request)

    assert len(captured) == 3, "a segment was dropped"
    resp = responses[0]
    assert len(resp.segments) == 3
    assert resp.speaker_count == 2
    assert resp.speakers == ["SPEAKER_00", "SPEAKER_01"]

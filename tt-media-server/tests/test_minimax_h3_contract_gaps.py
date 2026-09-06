# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""MiniMax-H3 request/serving contract gaps found by the 2026-09-06 gap hunt -- unit-level tripwires.

Every test here encodes the behaviour the API *documents or implies* and is ``xfail(strict=True)``
while the code does something else, so a fix flips it to XPASS and asks for the mark to go.
"""
import os
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from domain.video_generate_request import VideoGenerateRequest
from pydantic import ValidationError

H3_WORKER = SimpleNamespace(model_runner="tt-minimax-h3-t2va", model="MiniMax-H3")
TMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@pytest.fixture
def h3_settings():
    with patch("domain.video_generate_request.get_settings", return_value=H3_WORKER):
        yield


# --------------------------------------------------------------------------- denoise steps

@pytest.mark.xfail(strict=True, reason="MINIMAX_H3_NUM_INFERENCE_STEPS is 50 and the policy calls steps 'not a request lever', "
                   "but the shared DEFAULT_VIDEO_INFERENCE_STEPS=20 is forwarded (sp_runner `or 20`, dit_runners `or 50` never "
                   "fires): every served H3 request runs 20 steps while warmup runs 50")
@pytest.mark.usefixtures("h3_settings")
def test_default_steps_resolve_to_the_h3_policy_value():
    from tt_model_runners.minimax_h3_policy import MINIMAX_H3_NUM_INFERENCE_STEPS
    req = VideoGenerateRequest(prompt="a fox")
    assert req.num_inference_steps == MINIMAX_H3_NUM_INFERENCE_STEPS


@pytest.mark.xfail(strict=True, reason="an explicit num_inference_steps is accepted although H3 pins the step count")
@pytest.mark.usefixtures("h3_settings")
def test_explicit_steps_are_refused_for_h3():
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt="a fox", num_inference_steps=20)


# --------------------------------------------------------------------------- prompt length / bytes

@pytest.mark.xfail(strict=True, reason="the SHM request slot holds 2048 prompt bytes (ipc/video_shm.py MAX_PROMPT_LEN) and "
                   "_pack_string slices silently; a longer prompt is served truncated with a 202 (4679- and 11699-char "
                   "prompts produced byte-identical videos on quad1 2026-09-06)")
@pytest.mark.usefixtures("h3_settings")
def test_prompt_longer_than_the_wire_slot_is_refused_or_delivered_whole():
    long_prompt = "a red fox trots through fresh snow " * 80    # ~2800 bytes
    from ipc.video_shm import VideoShm
    assert len(long_prompt.encode()) > VideoShm.MAX_PROMPT_LEN
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt=long_prompt)


@pytest.mark.xfail(strict=True, reason="negative_prompt is packed into a 512-byte slot and silently cut")
@pytest.mark.usefixtures("h3_settings")
def test_negative_prompt_longer_than_the_wire_slot_is_refused():
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt="a fox", negative_prompt="blurry, " * 100)


@pytest.mark.xfail(strict=True, reason="#5044: blank and whitespace-only prompts must be a 422; VideoGenerateRequest.prompt has "
                   "no min_length/strip, so they reach the mesh")
@pytest.mark.usefixtures("h3_settings")
@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_blank_prompt_is_refused(prompt):
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt=prompt)


# --------------------------------------------------------------------------- levers that do nothing

@pytest.mark.xfail(strict=True, reason="negative_prompt is accepted and advertised but never reaches the H3 pipeline "
                   "(guidance-distilled model, dit_runners passes prompt/shape/steps/seed only): a silent no-op of the kind "
                   "_reject_unknown_fields exists to prevent")
@pytest.mark.usefixtures("h3_settings")
def test_negative_prompt_is_refused_as_not_a_lever():
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt="a fox", negative_prompt="blurry, text, watermark")


@pytest.mark.xfail(strict=True, reason="seed has no bounds: values >= 2**63 pass admission and die in struct.pack_into ('<q') "
                   "inside the SP runner, surfacing as a failed job instead of a 422")
@pytest.mark.usefixtures("h3_settings")
def test_out_of_int64_seed_is_refused():
    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt="a fox", seed=2**63)


# --------------------------------------------------------------------------- worker loop: rank outcomes

@pytest.mark.xfail(strict=True, reason="rank 0's own outcome alone decides what the encoder ships: when another rank errors "
                   "(seen 2026-09-04 ref2va task 7795fbdb: ranks 2/3 TT_THROW while rank 0 wrote the mp4) the job is reported "
                   "completed. The loop exchanges no status across ranks (only bcast + barrier)")
def test_inference_loop_exchanges_rank_outcomes():
    import queue as _queue
    import sys
    sys.path.insert(0, TMS)
    from tt_model_runners.video_runner import _run_inference_loop, VideoRequest
    import tt_model_runners.video_runner as vr
    vr._shutdown = False
    req = VideoRequest(task_id="t1", prompt="p", negative_prompt="", num_inference_steps=20, seed=1, height=768, width=1344,
                       num_frames=124, guidance_scale=3.0, guidance_scale_2=4.0)
    comm = MagicMock()
    comm.Get_rank.return_value = 0
    comm.bcast.side_effect = [(req, None, False), (None, None, False)]
    runner = MagicMock(); runner.requires_image_conditioning = False; runner.run.return_value = MagicMock()
    shm = MagicMock(); shm.read_request.return_value = req
    q: _queue.Queue = _queue.Queue()
    _run_inference_loop(comm, runner, shm, q)
    exchanged = {name for name, *_ in comm.method_calls} & {"allgather", "allreduce", "gather", "reduce"}
    assert exchanged, f"no per-request status exchange between ranks; comm calls were {sorted({n for n, *_ in comm.method_calls})}"


# --------------------------------------------------------------------------- route defaults

@pytest.mark.xfail(strict=True, reason="/generations/i2v/upload declares its own Form defaults (num_inference_steps=12) and has no "
                   "duration_seconds / aspect_ratio fields, so fl2va via multipart diverges from the JSON route and always "
                   "serves 5 s at 16:9")
def test_upload_route_matches_the_json_route():
    src = open(os.path.join(TMS, "open_ai_api", "video.py")).read()
    start = src.index("i2v/upload")
    block = src[start:start + 3000]
    assert "duration_seconds" in block and "aspect_ratio" in block, "upload route cannot carry duration/aspect"
    m = re.search(r"num_inference_steps[^\n]*Form\((\d+)", block)
    assert not m or int(m.group(1)) == 20, f"upload default steps {m.group(1)} != JSON default 20"


# --------------------------------------------------------------------------- pins (documenting current limits)

def test_worker_error_messages_are_capped_at_256_bytes():
    """Documented limit: OOM texts lose their allocated/free numbers past byte 256 (ipc/video_shm.py MAX_ERROR_LEN)."""
    from ipc.video_shm import VideoShm
    assert VideoShm.MAX_ERROR_LEN == 256

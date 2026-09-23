# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from unittest.mock import patch

import pytest
from domain.video_generate_request import VideoGenerateRequest
from pydantic import ValidationError


@pytest.fixture
def minimax_request_validation():
    with patch("domain.video_generate_request._is_minimax_h3", return_value=True):
        yield


@pytest.mark.usefixtures("minimax_request_validation")
@pytest.mark.parametrize("duration_seconds", [4, 5, 10, 15])
def test_minimax_accepts_documented_durations(duration_seconds):
    request = VideoGenerateRequest(
        prompt="A fox runs through wet grass.",
        aspect_ratio="16:9",
        duration_seconds=duration_seconds,
    )
    assert request.duration_seconds == duration_seconds


@pytest.mark.usefixtures("minimax_request_validation")
@pytest.mark.parametrize(
    "aspect_ratio",
    ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
)
def test_minimax_accepts_served_aspect_ratios(aspect_ratio):
    request = VideoGenerateRequest(
        prompt="A fox runs through wet grass.",
        aspect_ratio=aspect_ratio,
        duration_seconds=5,
    )
    assert request.aspect_ratio == aspect_ratio


@pytest.mark.usefixtures("minimax_request_validation")
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aspect_ratio", "2:1"),
        ("duration_seconds", 3),
        ("duration_seconds", 16),
        ("num_inference_steps", 50),
        ("resolution", "768P"),
    ],
)
def test_minimax_rejects_unsupported_request_fields(field, value):
    payload = {
        "prompt": "A fox runs through wet grass.",
        "aspect_ratio": "16:9",
        "duration_seconds": 5,
        field: value,
    }
    with pytest.raises(ValidationError):
        VideoGenerateRequest(**payload)


def test_shared_video_schema_keeps_non_minimax_behavior():
    with patch("domain.video_generate_request._is_minimax_h3", return_value=False):
        request = VideoGenerateRequest(
            prompt="A fox runs through wet grass.",
            num_inference_steps=20,
            resolution="ignored-by-shared-schema",
        )
    assert request.num_inference_steps == 20


@pytest.mark.usefixtures("minimax_request_validation")
def test_minimax_pins_the_fixed_schedule_when_steps_are_omitted():
    from tt_model_runners.minimax_h3_policy import MINIMAX_H3_NUM_INFERENCE_STEPS

    request = VideoGenerateRequest(
        prompt="A fox runs through wet grass.",
        aspect_ratio="16:9",
        duration_seconds=5,
    )
    assert request.num_inference_steps == MINIMAX_H3_NUM_INFERENCE_STEPS


def test_shm_rebuild_drops_the_step_count_for_minimax():
    """The multi-host rank workers rebuild the request from shared memory, where
    num_inference_steps is always present; that path must not trip the admission rule."""
    from ipc.video_shm import VideoRequest
    from tt_model_runners.minimax_h3_policy import MINIMAX_H3_NUM_INFERENCE_STEPS
    from tt_model_runners.video_runner import video_request_to_generate_request

    req = VideoRequest(
        task_id="t-1",
        prompt="A fox runs through wet grass.",
        negative_prompt="",
        num_inference_steps=20,
        seed=42,
        height=768,
        width=1344,
        num_frames=124,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
    )
    with patch("domain.video_generate_request._is_minimax_h3", return_value=True):
        with patch("tt_model_runners.video_runner._is_minimax_h3", return_value=True):
            gen = video_request_to_generate_request(req)
    assert gen.prompt == req.prompt and gen.seed == 42
    assert gen.num_inference_steps == MINIMAX_H3_NUM_INFERENCE_STEPS
    with patch("domain.video_generate_request._is_minimax_h3", return_value=False):
        with patch("tt_model_runners.video_runner._is_minimax_h3", return_value=False):
            assert video_request_to_generate_request(req).num_inference_steps == 20

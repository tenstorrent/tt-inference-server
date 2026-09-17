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

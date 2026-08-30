# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from unittest.mock import MagicMock, patch

import pytest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from pydantic import ValidationError

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _fl2va_settings():
    settings = MagicMock()
    settings.model_runner = "tt-minimax-h3-fl2va"
    return settings


class TestFL2VAFramePos:
    @patch(
        "domain.video_i2v_generate_request.get_settings",
        _fl2va_settings,
    )
    def test_first_and_last_sentinels(self):
        request = VideoI2VGenerateRequest(
            prompt="brad pitt",
            image_prompts=[
                ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=0),
                ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=-1),
            ],
        )
        assert [e.frame_pos for e in request.image_prompts] == [0, -1]

    @patch(
        "domain.video_i2v_generate_request.get_settings",
        _fl2va_settings,
    )
    def test_mid_clip_rejected(self):
        with pytest.raises(ValidationError, match="0 \\(first\\) or -1"):
            VideoI2VGenerateRequest(
                prompt="brad pitt",
                image_prompts=[
                    ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=40),
                ],
            )

    @patch(
        "domain.video_i2v_generate_request.get_settings",
        _fl2va_settings,
    )
    def test_more_than_two_rejected(self):
        with pytest.raises(ValidationError, match="at most two"):
            VideoI2VGenerateRequest(
                prompt="brad pitt",
                image_prompts=[
                    ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=0),
                    ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=-1),
                    ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=1),
                ],
            )

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from PIL import Image
from pydantic import ValidationError
from tt_model_runners.minimax_h3_policy import MINIMAX_H3_MEDIA_MIN_SIDE_PX


def _png_b64(side: int = MINIMAX_H3_MEDIA_MIN_SIDE_PX) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (side, side), (90, 120, 150)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# On an FL2VA deployment each keyframe is admitted against the MiniMax input media
# card, whose floor is 256 px per side -- so the fixture is that, not a 1x1 pixel.
_TINY_PNG_BASE64 = _png_b64()


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

    @patch("domain.video_i2v_generate_request.get_settings")
    def test_sp_runner_with_fl2va_model_rejects_mid_clip(self, mock_settings):
        mock_settings.return_value.model_runner = "sp_runner"
        with patch.dict(os.environ, {"MODEL": "MiniMax-H3-FL2VA"}):
            with pytest.raises(ValidationError, match="0 \\(first\\) or -1"):
                VideoI2VGenerateRequest(
                    prompt="brad pitt",
                    image_prompts=[
                        ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=40),
                    ],
                )

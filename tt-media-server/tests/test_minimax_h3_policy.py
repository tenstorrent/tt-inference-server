# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import pytest
from tt_model_runners.minimax_h3_policy import (
    MINIMAX_H3_ASPECT_RATIOS,
    MINIMAX_H3_DEFAULT_ASPECT_RATIO,
    MINIMAX_H3_DEFAULT_DURATION_S,
    MINIMAX_H3_DURATIONS_S,
    MINIMAX_H3_NUM_INFERENCE_STEPS,
    minimax_h3_frames_are_aligned,
    minimax_h3_parse_aspect_ratio,
)


class TestParseAspectRatio:
    @pytest.mark.parametrize(
        "text,expected", [("16:9", (16, 9)), ("9:16", (9, 16)), ("1:1", (1, 1))]
    )
    def test_published_ratios(self, text, expected):
        assert minimax_h3_parse_aspect_ratio(text) == expected

    def test_accepts_x_and_slash(self):
        assert minimax_h3_parse_aspect_ratio("16x9") == (16, 9)
        assert minimax_h3_parse_aspect_ratio("16/9") == (16, 9)

    def test_rejects_unpublished_ratio(self):
        with pytest.raises(ValueError, match="is not served"):
            minimax_h3_parse_aspect_ratio("2:1")

    def test_rejects_malformed(self):
        with pytest.raises(ValueError, match="must look like"):
            minimax_h3_parse_aspect_ratio("widescreen")

    def test_error_names_the_supported_set(self):
        with pytest.raises(ValueError, match="21:9") as exc_info:
            minimax_h3_parse_aspect_ratio("2:1")
        for width, height in MINIMAX_H3_ASPECT_RATIOS:
            assert f"{width}:{height}" in str(exc_info.value)


class TestDurationAllowList:
    def test_every_integer_from_4_to_15(self):
        assert MINIMAX_H3_DURATIONS_S == tuple(range(4, 16))

    def test_default_is_served(self):
        assert MINIMAX_H3_DEFAULT_DURATION_S in MINIMAX_H3_DURATIONS_S

    def test_default_aspect_ratio_is_served(self):
        assert MINIMAX_H3_DEFAULT_ASPECT_RATIO in MINIMAX_H3_ASPECT_RATIOS


class TestInferenceSteps:
    def test_fixed_at_fifty(self):
        assert MINIMAX_H3_NUM_INFERENCE_STEPS == 50


class TestFramesAreAligned:
    def test_default_5s_aligns_to_124(self):
        packing = pytest.importorskip(
            "models.tt_dit.pipelines.minimax_h3.packing",
            exc_type=ImportError,
        )
        frames = packing.align_num_frames(
            round(MINIMAX_H3_DEFAULT_DURATION_S * packing.MINIMAX_H3_FPS)
        )
        assert frames == 124
        assert minimax_h3_frames_are_aligned(frames)

    def test_unaligned_120_is_rejected(self):
        pytest.importorskip(
            "models.tt_dit.pipelines.minimax_h3.packing",
            exc_type=ImportError,
        )
        assert not minimax_h3_frames_are_aligned(120)

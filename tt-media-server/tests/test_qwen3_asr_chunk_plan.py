# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

import pytest
from model_services.audio_service import qwen3_asr_chunk_plan


def _settings(min_split=15, short_max=30, short_chunk=15, long_chunk=10):
    return SimpleNamespace(
        audio_min_split_duration_seconds=min_split,
        audio_short_clip_max_seconds=short_max,
        audio_short_clip_chunk_seconds=short_chunk,
        audio_chunk_duration_seconds=long_chunk,
    )


@pytest.mark.parametrize("duration", [1.0, 5.0, 10.0, 15.0])
def test_clips_at_or_under_min_split_stay_whole(duration):
    assert qwen3_asr_chunk_plan(duration, None, _settings()) is None


@pytest.mark.parametrize("duration", [15.1, 20.0, 28.8, 30.0])
def test_short_tier_uses_gentle_15s_window(duration):
    """15-30s clips fan out at 15s (2 runners) rather than the aggressive 10s."""
    assert qwen3_asr_chunk_plan(duration, None, _settings()) == 15


@pytest.mark.parametrize("duration", [30.1, 60.0, 120.0, 320.0])
def test_long_tier_uses_full_worker_count_window(duration):
    """>30s keeps the 10s fan-out for maximum parallelism -- unchanged."""
    assert qwen3_asr_chunk_plan(duration, None, _settings()) == 10


def test_explicit_override_wins_once_splittable():
    """Dp2Chunk5/Chunk30 spec tests send an explicit size on 60s audio."""
    assert qwen3_asr_chunk_plan(60.0, 5, _settings()) == 5
    assert qwen3_asr_chunk_plan(60.0, 30, _settings()) == 30


def test_override_ignored_when_clip_kept_whole():
    """Below the split threshold there is nothing to fan out, so an override
    cannot force a genuinely short clip to be chopped."""
    assert qwen3_asr_chunk_plan(10.0, 5, _settings()) is None


def test_boundary_is_inclusive_of_short_tier_at_exactly_30s():
    """A 30.0s clip is the short tier (15s); 30.0001s tips into the long tier."""
    assert qwen3_asr_chunk_plan(30.0, None, _settings()) == 15
    assert qwen3_asr_chunk_plan(30.0001, None, _settings()) == 10

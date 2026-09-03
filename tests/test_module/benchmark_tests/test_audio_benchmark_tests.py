# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from test_module.benchmark_tests.audio_benchmark_tests import (
    _join_stream_chunks,
    _MIN_STREAMING_WINDOW_S,
)


class TestJoinStreamChunks:
    """Transcript reconstruction across the two streaming payload shapes."""

    def test_token_deltas_followed_by_full_text_are_not_counted_twice(self):
        """Short clips stream token deltas then repeat the whole transcript.

        Summing tokens over every chunk double-counted the transcript and
        inflated T/S/U.
        """
        chunks = ["Con", "cord", " returned", " home", "Concord returned home"]
        assert _join_stream_chunks(chunks) == "Concord returned home"

    def test_one_text_per_segment_is_concatenated(self):
        """Fan-out sends a complete transcript per audio segment."""
        chunks = ["First segment.", "Second segment.", "Third segment."]
        assert (
            _join_stream_chunks(chunks)
            == "First segment. Second segment. Third segment."
        )

    def test_partial_repeat_is_not_treated_as_aggregate(self):
        """Only an exact repeat of the deltas counts as a final aggregate."""
        chunks = ["alpha", "beta", "alpha beta gamma"]
        assert _join_stream_chunks(chunks) == "alpha beta alpha beta gamma"

    def test_single_chunk_and_empty(self):
        assert _join_stream_chunks(["only"]) == "only"
        assert _join_stream_chunks([]) == ""

    def test_whitespace_and_case_differences_still_dedupe(self):
        chunks = ["Hello ", "world", "hello   WORLD"]
        assert _join_stream_chunks(chunks) == "hello   WORLD"


class TestStreamingWindowGuard:
    def test_burst_threshold_is_positive(self):
        """A burst response must fall back to a non-zero denominator.

        Parallel fan-out delivers every chunk at once, so the first->last
        window collapses to ~0 and previously divided by zero.
        """
        assert _MIN_STREAMING_WINDOW_S > 0

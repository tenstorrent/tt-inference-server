# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""TextUtils.extract_text over the shapes tt-metal's WhisperGenerator returns.

The generator yields ``(final_result, avg_logprob, no_speech_probs, ...)`` with
one entry per batch item; an item is a plain string, or with
``return_timestamps`` a list of ``{"text", "start", "end"}`` segments. The
batch formatter in whisper_runner passes each item on its own.
"""

from utils.text_utils import TextUtils

SEGMENTS = [
    {"text": " Hello there,", "start": 0.0, "end": 1.2},
    {"text": " how are you?", "start": 1.2, "end": 2.5},
    {"text": " Fine.", "start": 2.8, "end": 3.4},
]


def test_batch_1_timestamp_result_joins_all_segments():
    chunk = ([SEGMENTS], -0.2, [0.01], {})
    assert TextUtils.extract_text(chunk) == (
        "Hello there, how are you? Fine.",
        0.0,
        3.4,
    )


def test_batch_item_segment_list_joins_all_segments():
    assert TextUtils.extract_text(SEGMENTS) == (
        "Hello there, how are you? Fine.",
        0.0,
        3.4,
    )


def test_single_segment_item():
    assert TextUtils.extract_text(SEGMENTS[:1]) == ("Hello there,", 0.0, 1.2)


def test_plain_text_result_has_no_times():
    assert TextUtils.extract_text((["hello world"], -0.2, [0.01])) == (
        "hello world",
        None,
        None,
    )
    assert TextUtils.extract_text("hello world") == ("hello world", None, None)


def test_empty_results():
    assert TextUtils.extract_text(([], -0.2, [0.01])) == ("", None, None)
    assert TextUtils.extract_text([]) == ("", None, None)
    assert TextUtils.extract_text(([[]], -0.2, [0.01])) == ("", None, None)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Job metadata echoes ``request_parameters``; inline base64 media must not ride along."""

from utils.job_manager import redact_inline_media


def test_long_inline_media_is_replaced_by_a_size_note():
    params = {
        "prompt": "a quiet room",
        "image_prompts": [{"image": "A" * 5000, "frame_pos": 0}],
        "references": {
            "images": [{"b64": "B" * 4321, "url": None}],
            "videos": [{"b64": None, "url": "https://cdn.example.com/clip.mp4"}],
            "audios": [],
        },
    }
    out = redact_inline_media(params)
    assert out["image_prompts"][0] == {
        "image": "<inline media omitted: 5000 base64 chars>",
        "frame_pos": 0,
    }
    assert out["references"]["images"][0] == {
        "b64": "<inline media omitted: 4321 base64 chars>",
        "url": None,
    }
    assert out["references"]["videos"][0]["url"] == "https://cdn.example.com/clip.mp4"
    assert out["prompt"] == "a quiet room"
    # the caller's dict is left alone
    assert params["image_prompts"][0]["image"] == "A" * 5000


def test_short_values_urls_and_other_keys_are_kept():
    params = {
        "image": "https://cdn.example.com/a.png",
        "b64": "short",
        "prompt": "x" * 10_000,
        "nested": [{"image": "y" * 100}],
    }
    assert redact_inline_media(params) == params


def test_non_container_values_pass_through():
    assert redact_inline_media("text") == "text"
    assert redact_inline_media(None) is None
    assert redact_inline_media(7) == 7

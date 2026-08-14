# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for data-driven MiniMax fixture selection."""

from __future__ import annotations

import pytest
from minimax_mock.fixture_resolver import (
    FixtureCatalog,
    FixtureCatalogError,
    GenerationMode,
    classify_request,
)
from minimax_mock.schemas import VideoGenerationRequest


def _request(content: list[dict], ratio: str | None = None) -> VideoGenerationRequest:
    payload = {
        "model": "MiniMax-H3",
        "content": content,
        "resolution": "2K",
        "duration": 5,
    }
    if ratio is not None:
        payload["ratio"] = ratio
    return VideoGenerationRequest.model_validate(payload)


@pytest.mark.parametrize(
    "generation_request,mode,fixture_name",
    [
        (
            _request(
                [{"type": "text", "text": "Generate a city skyline."}],
                ratio="16:9",
            ),
            GenerationMode.TEXT_TO_VIDEO,
            "t2v-success",
        ),
        (
            _request(
                [
                    {"type": "text", "text": "Animate this frame."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/first.png"},
                    },
                ]
            ),
            GenerationMode.IMAGE_TO_VIDEO_FIRST,
            "i2v-first-frame-success",
        ),
        (
            _request(
                [
                    {"type": "text", "text": "End on this frame."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/last.png"},
                        "role": "last_frame",
                    },
                ]
            ),
            GenerationMode.IMAGE_TO_VIDEO_LAST,
            "i2v-last-frame-success",
        ),
        (
            _request(
                [
                    {"type": "text", "text": "Move between these frames."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/first.png"},
                        "role": "first_frame",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/last.png"},
                        "role": "last_frame",
                    },
                ]
            ),
            GenerationMode.IMAGE_TO_VIDEO_FIRST_LAST,
            "i2v-first-last-success",
        ),
        (
            _request(
                [
                    {"type": "text", "text": "Follow these references."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/reference.png"},
                        "role": "reference_image",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/reference.mp3"},
                        "role": "reference_audio",
                    },
                ]
            ),
            GenerationMode.REFERENCE_TO_VIDEO,
            "reference-success",
        ),
    ],
)
def test_catalog_resolves_each_generation_mode(generation_request, mode, fixture_name):
    catalog = FixtureCatalog()

    assert classify_request(generation_request) is mode
    fixture = catalog.resolve(generation_request)
    assert fixture.manifest.name == fixture_name
    assert fixture.manifest.terminal_status == "succeeded"
    assert fixture.manifest.media_type == "video/mp4"
    assert fixture.asset_path is not None
    assert fixture.asset_path.is_file()


def test_catalog_can_resolve_forced_failure_scenario():
    catalog = FixtureCatalog()
    request = _request(
        [{"type": "text", "text": "Generate a city skyline."}],
        ratio="16:9",
    )

    fixture = catalog.resolve(request, scenario_name="generation-failed")

    assert fixture.manifest.terminal_status == "failed"
    assert fixture.manifest.error is not None
    assert fixture.manifest.error.code == "1026"
    assert fixture.asset_path is None


def test_catalog_rejects_unknown_scenario():
    catalog = FixtureCatalog()
    request = _request(
        [{"type": "text", "text": "Generate a city skyline."}],
        ratio="16:9",
    )

    with pytest.raises(FixtureCatalogError, match="unknown fixture scenario"):
        catalog.resolve(request, scenario_name="not-configured")

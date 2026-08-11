"""Verify fixture videos match MiniMax output metadata."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from minimax_mock.fixture_resolver import FixtureCatalog
from minimax_mock.media_fixtures import (
    FIXTURE_VIDEO_FPS,
    OUTPUT_DIMENSIONS,
    OUTPUT_RATIOS,
    RATIO_ASSET_KEYS,
)
from minimax_mock.schemas import AspectRatio, Resolution, VideoGenerationRequest


def _request(
    resolution: Resolution,
    ratio: AspectRatio,
    duration: int,
) -> VideoGenerationRequest:
    return VideoGenerationRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "A fixture video."}],
            "resolution": resolution.value,
            "duration": duration,
            "ratio": ratio.value,
        }
    )


def test_fixture_catalog_covers_every_output_combination():
    catalog = FixtureCatalog()
    asset_paths = set()

    for resolution in Resolution:
        for ratio in OUTPUT_RATIOS:
            for duration in range(4, 16):
                fixture = catalog.resolve(_request(resolution, ratio, duration))
                assert fixture.output_ratio == ratio.value
                assert fixture.asset_path is not None
                assert fixture.asset_path.is_file()
                assert fixture.asset_path.parts[-3:] == (
                    resolution.value.lower(),
                    RATIO_ASSET_KEYS[ratio],
                    f"{duration}.mp4",
                )
                asset_paths.add(fixture.asset_path)

    assert len(asset_paths) == 144


def test_adaptive_image_request_uses_concrete_widescreen_fixture():
    request = VideoGenerationRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [
                {"type": "text", "text": "Animate this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/first.png"},
                    "role": "first_frame",
                },
            ],
            "resolution": "2K",
            "duration": 5,
            "ratio": "9:16",
        }
    )

    fixture = FixtureCatalog().resolve(request)

    assert fixture.output_ratio == "16:9"
    assert fixture.asset_path.parts[-2:] == ("16x9", "5.mp4")


@pytest.mark.parametrize(
    "resolution,ratio,duration",
    [
        (Resolution.P768, AspectRatio.RATIO_21_9, 4),
        (Resolution.P768, AspectRatio.RATIO_9_16, 15),
        (Resolution.P2K, AspectRatio.RATIO_16_9, 5),
        (Resolution.P2K, AspectRatio.RATIO_1_1, 12),
    ],
)
def test_fixture_media_metadata_matches_request(resolution, ratio, duration):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is required to inspect fixture media")

    fixture = FixtureCatalog().resolve(_request(resolution, ratio, duration))
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(fixture.asset_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    video_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "video"
    )
    audio_stream = next(
        stream for stream in metadata["streams"] if stream["codec_type"] == "audio"
    )

    width, height = OUTPUT_DIMENSIONS[resolution][ratio]
    assert (video_stream["width"], video_stream["height"]) == (width, height)
    assert video_stream["codec_name"] == "h264"
    assert video_stream["r_frame_rate"] == f"{FIXTURE_VIDEO_FPS}/1"
    assert audio_stream["codec_name"] == "aac"
    assert float(metadata["format"]["duration"]) == pytest.approx(duration, abs=0.01)

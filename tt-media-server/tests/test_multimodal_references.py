# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest
from domain.video_ref2va_generate_request import (
    MediaSource,
    MultimodalReferences,
    VideoRef2VAGenerateRequest,
)
from pydantic import ValidationError
from tt_model_runners.minimax_h3_policy import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    check_reference_clip_durations,
)

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class TestMediaSource:
    def test_b64_only(self):
        source = MediaSource(b64=_TINY_PNG_BASE64)
        assert source.b64 == _TINY_PNG_BASE64
        assert source.url is None

    def test_url_only(self):
        source = MediaSource(url="https://bucket.s3.amazonaws.com/clip.mp4")
        assert source.url.startswith("https://")
        assert source.b64 is None

    def test_neither_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            MediaSource()

    def test_both_rejected(self):
        with pytest.raises(ValidationError, match="exactly one"):
            MediaSource(b64=_TINY_PNG_BASE64, url="https://example.com/a.png")

    def test_non_http_url_rejected(self):
        with pytest.raises(ValidationError, match="http"):
            MediaSource(url="ftp://example.com/a.png")


class TestMultimodalReferences:
    def test_one_image(self):
        refs = MultimodalReferences(images=[MediaSource(b64=_TINY_PNG_BASE64)])
        assert len(refs.images) == 1

    def test_empty_rejected(self):
        with pytest.raises(ValidationError, match="at least one"):
            MultimodalReferences()

    def test_audio_alone_rejected(self):
        with pytest.raises(ValidationError, match="paired"):
            MultimodalReferences(audios=[MediaSource(b64="AAAA")])

    def test_too_many_images(self):
        with pytest.raises(ValidationError, match="reference images"):
            MultimodalReferences(
                images=[
                    MediaSource(b64=_TINY_PNG_BASE64)
                    for _ in range(MINIMAX_H3_MAX_REFERENCE_IMAGES + 1)
                ]
            )

    def test_too_many_videos(self):
        with pytest.raises(ValidationError, match="reference videos"):
            MultimodalReferences(
                videos=[
                    MediaSource(url="https://example.com/v.mp4")
                    for _ in range(MINIMAX_H3_MAX_REFERENCE_VIDEOS + 1)
                ]
            )

    def test_too_many_audios(self):
        with pytest.raises(ValidationError, match="reference audios"):
            MultimodalReferences(
                images=[MediaSource(b64=_TINY_PNG_BASE64)],
                audios=[
                    MediaSource(url="https://example.com/a.wav")
                    for _ in range(MINIMAX_H3_MAX_REFERENCE_AUDIOS + 1)
                ],
            )

    def test_mixed_accepted(self):
        refs = MultimodalReferences(
            images=[MediaSource(b64=_TINY_PNG_BASE64)],
            videos=[MediaSource(url="https://example.com/v.mp4")],
            audios=[MediaSource(url="https://example.com/a.wav")],
        )
        assert len(refs.images) == 1
        assert len(refs.videos) == 1
        assert len(refs.audios) == 1


class TestVideoRef2VAGenerateRequest:
    def test_requires_references(self):
        with pytest.raises(ValidationError):
            VideoRef2VAGenerateRequest(prompt="a quiet room")

    def test_valid(self):
        request = VideoRef2VAGenerateRequest(
            prompt="a quiet room",
            references=MultimodalReferences(images=[MediaSource(b64=_TINY_PNG_BASE64)]),
        )
        assert request.prompt == "a quiet room"


class TestReferenceClipDurations:
    def test_in_window(self):
        check_reference_clip_durations(
            video_durations=[5.0, 8.0], audio_durations=[2.0]
        )

    def test_clip_too_short(self):
        with pytest.raises(ValueError, match="videos\\[0\\]"):
            check_reference_clip_durations(video_durations=[1.5], audio_durations=[])

    def test_clip_too_long(self):
        with pytest.raises(ValueError, match="audios\\[0\\]"):
            check_reference_clip_durations(video_durations=[], audio_durations=[16.0])

    def test_combined_too_long(self):
        with pytest.raises(ValueError, match="combined videos"):
            check_reference_clip_durations(
                video_durations=[8.0, 8.0], audio_durations=[]
            )

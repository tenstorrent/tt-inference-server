# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import numpy as np
import pytest
from domain.video_i2v_generate_request import MAX_BASE64_IMAGE_LEN
from domain.video_ref2va_generate_request import (
    MAX_BASE64_AUDIO_LEN,
    MAX_BASE64_MEDIA_LEN,
    MediaSource,
    MultimodalReferences,
    VideoRef2VAGenerateRequest,
)
from pydantic import ValidationError
from tt_model_runners.minimax_h3_policy import (
    check_reference_clip_durations,
    probe_reference_media,
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

    def test_mixed_accepted(self):
        refs = MultimodalReferences(
            images=[MediaSource(b64=_TINY_PNG_BASE64)],
            videos=[MediaSource(url="https://example.com/v.mp4")],
            audios=[MediaSource(url="https://example.com/a.wav")],
        )
        assert len(refs.images) == 1
        assert len(refs.videos) == 1
        assert len(refs.audios) == 1

    def test_caps_match_minimax_limits(self):
        assert MAX_BASE64_IMAGE_LEN == 40_000_000
        assert MAX_BASE64_MEDIA_LEN == 66_666_668
        assert MAX_BASE64_AUDIO_LEN == 20_000_000

    def test_oversized_inline_image_rejected(self):
        with pytest.raises(ValidationError, match=r"images\[0\] base64 length"):
            MultimodalReferences(
                images=[MediaSource(b64="A" * (MAX_BASE64_IMAGE_LEN + 1))]
            )

    def test_oversized_inline_audio_rejected(self):
        with pytest.raises(ValidationError, match=r"audios\[0\] base64 length"):
            MultimodalReferences(
                images=[MediaSource(b64=_TINY_PNG_BASE64)],
                audios=[MediaSource(b64="A" * (MAX_BASE64_AUDIO_LEN + 1))],
            )

    def test_inline_audio_at_cap_accepted(self):
        refs = MultimodalReferences(
            images=[MediaSource(b64=_TINY_PNG_BASE64)],
            audios=[MediaSource(b64="A" * MAX_BASE64_AUDIO_LEN)],
        )
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

    def test_soundtracks_count_toward_combined_audio(self):
        with pytest.raises(ValueError, match="combined audio duration is 17.08 s"):
            check_reference_clip_durations(
                video_durations=[10.125],
                audio_durations=[7.0],
                soundtrack_durations=[10.08],
            )

    def test_soundtracks_and_audio_at_the_limit(self):
        check_reference_clip_durations(
            video_durations=[8.0], audio_durations=[7.0], soundtrack_durations=[8.0]
        )


def _encode_video_with_soundtrack(path, video_s, audio_s):
    import av

    fps, sample_rate = 16, 32000
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = 64, 32, "yuv420p"
        audio = container.add_stream("aac", rate=sample_rate, layout="stereo")
        for _ in range(int(video_s * fps)):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 64, 3), dtype=np.uint8), format="rgb24"
            )
            container.mux(video.encode(frame))
        container.mux(video.encode())
        samples = np.zeros((2, 1024), dtype=np.float32)
        for index in range(int(audio_s * sample_rate) // 1024):
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate, frame.pts = sample_rate, index * 1024
            container.mux(audio.encode(frame))
        container.mux(audio.encode())


def test_probe_times_a_soundtrack_on_its_own_stream(tmp_path):
    path = tmp_path / "clip.mp4"
    _encode_video_with_soundtrack(path, video_s=4, audio_s=2)
    probe = probe_reference_media(path.read_bytes())
    assert probe.audio_channels == 2
    assert probe.duration_s == pytest.approx(4, abs=0.1)
    assert probe.audio_duration_s == pytest.approx(2, abs=0.1)

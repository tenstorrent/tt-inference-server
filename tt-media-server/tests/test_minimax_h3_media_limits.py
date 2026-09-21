# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Admission against the MiniMax-H3 input media card.

The card (tt_model_runners/minimax_h3_policy.py) is: request body <= 64 MB; images
JPG/PNG/WEBP/HEIC/HEIF, <= 30 MB, [256, 5760] px, aspect 0.4-2.5; reference videos MP4/MOV,
H.264/H.265 + AAC/MP3, <= 50 MB, same pixel window, 23.976-60 fps, 2-15 s; reference audio
WAV/MP3, <= 15 MB, 2-15 s; 9 / 3 / 3 references and 12 in total. Clips are made with ffmpeg
and skipped when it is missing.
"""

import base64
import importlib.util
import io
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, features
from pydantic import ValidationError
from tt_model_runners import minimax_h3_policy as policy
from tt_model_runners.minimax_h3_policy import (
    MINIMAX_H3_AUDIO_MAX_BYTES,
    MINIMAX_H3_IMAGE_MAX_BYTES,
    MINIMAX_H3_MAX_REFERENCES_TOTAL,
    MINIMAX_H3_MAX_REQUEST_BODY_BYTES,
    MINIMAX_H3_VIDEO_MAX_BYTES,
    MediaTooLargeError,
    base64_len_for_bytes,
    check_h3_image,
    check_h3_reference_audio,
    check_h3_reference_video,
    check_reference_clip_durations,
    decode_base64_media,
    probe_media,
)


def _image(width: int, height: int, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buf, format=fmt)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# conftest swaps in a MagicMock for PIL when Pillow is missing; nothing here can run on that.
pytestmark = pytest.mark.skipif(
    not _image(8, 8).startswith(b"\x89PNG"), reason="needs a real Pillow"
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_HEIF = importlib.util.find_spec("pillow_heif") is not None


def _ffmpeg_has_encoder(name: str) -> bool:
    if not HAVE_FFMPEG:
        return False
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return f" {name} " in out


class TestCaps:
    def test_base64_len_matches_the_encoder(self):
        for n in (0, 1, 2, 3, 4, 5, 100, 4097):
            assert base64_len_for_bytes(n) == len(base64.b64encode(b"x" * n))

    def test_card_constants(self):
        assert MINIMAX_H3_MAX_REQUEST_BODY_BYTES == 64 * 1024 * 1024
        assert MINIMAX_H3_IMAGE_MAX_BYTES == 30 * 1024 * 1024
        assert MINIMAX_H3_VIDEO_MAX_BYTES == 50 * 1024 * 1024
        assert MINIMAX_H3_AUDIO_MAX_BYTES == 15 * 1024 * 1024
        assert MINIMAX_H3_MAX_REFERENCES_TOTAL == 12

    def test_field_caps_follow_the_card(self):
        from domain.video_i2v_generate_request import MAX_BASE64_IMAGE_LEN
        from domain.video_ref2va_generate_request import MAX_BASE64_MEDIA_LEN

        assert MAX_BASE64_IMAGE_LEN == base64_len_for_bytes(MINIMAX_H3_IMAGE_MAX_BYTES)
        assert (
            MAX_BASE64_IMAGE_LEN == 41_943_040
        )  # a 30 MB image fits, 7.5 MB was the old cap
        assert MAX_BASE64_MEDIA_LEN == base64_len_for_bytes(MINIMAX_H3_VIDEO_MAX_BYTES)

    def test_decode_tolerates_data_url_and_stripped_padding(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"abc"
        text = base64.b64encode(raw).decode().rstrip("=")
        assert decode_base64_media(text) == raw
        assert decode_base64_media("data:image/png;base64," + text) == raw

    def test_settings_defaults_follow_the_card(self):
        from config.settings import Settings

        fields = getattr(Settings, "model_fields", None)
        if not isinstance(fields, dict):
            pytest.skip(
                "config.settings is mocked by another test module in this session"
            )
        assert (
            fields["max_request_body_bytes"].default
            == MINIMAX_H3_MAX_REQUEST_BODY_BYTES
        )
        assert fields["media_url_max_bytes"].default == MINIMAX_H3_VIDEO_MAX_BYTES


class TestImageCard:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
    def test_admits_png_and_jpeg(self, fmt):
        image = check_h3_image(_image(256, 256, fmt))
        assert image.size == (256, 256)
        assert image.format == fmt

    @pytest.mark.skipif(not features.check("webp"), reason="Pillow without WebP")
    def test_admits_webp(self):
        assert check_h3_image(_image(384, 256, "WEBP")).format == "WEBP"

    @pytest.mark.skipif(not HAVE_HEIF, reason="pillow-heif not installed")
    def test_admits_heif_when_the_decoder_is_installed(self):
        import utils.image_manager  # noqa: F401  registers the opener

        assert check_h3_image(_image(256, 256, "HEIF")).format == "HEIF"

    def test_heif_without_the_decoder_names_the_gap(self):
        raw = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
        with patch.object(
            Image, "open", side_effect=OSError("cannot identify image file")
        ):
            with pytest.raises(ValueError, match="pillow-heif"):
                check_h3_image(raw)

    @pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
    def test_refuses_formats_off_the_card(self, fmt):
        with pytest.raises(ValueError, match="not supported"):
            check_h3_image(_image(256, 256, fmt))

    @pytest.mark.parametrize("size", [(255, 256), (256, 255), (5761, 2400)])
    def test_refuses_sides_outside_the_window(self, size):
        with pytest.raises(ValueError, match=r"\[256, 5760\] px"):
            check_h3_image(_image(*size))

    def test_admits_the_window_edges(self):
        assert check_h3_image(_image(256, 256)).size == (256, 256)
        assert check_h3_image(_image(5760, 2304)).size == (
            5760,
            2304,
        )  # w/h = 2.5 exactly

    @pytest.mark.parametrize("size", [(256, 641), (643, 256)])  # 0.399 and 2.51
    def test_refuses_aspect_outside_the_window(self, size):
        with pytest.raises(ValueError, match="aspect ratio"):
            check_h3_image(_image(*size))

    @pytest.mark.parametrize("size", [(256, 640), (640, 256)])  # 0.4 and 2.5
    def test_admits_aspect_edges(self, size):
        check_h3_image(_image(*size))

    def test_refuses_oversize_with_a_size_error(self, monkeypatch):
        monkeypatch.setattr(policy, "MINIMAX_H3_IMAGE_MAX_BYTES", 100)
        with pytest.raises(MediaTooLargeError, match="at most 100 bytes") as info:
            check_h3_image(_image(256, 256), label="images[2]")
        assert isinstance(
            info.value, ValueError
        )  # pydantic reports it like any field error
        assert str(info.value).startswith("images[2] is ")

    def test_refuses_junk(self):
        with pytest.raises(ValueError, match="could not be decoded"):
            check_h3_image(b"definitely not an image")

    def test_is_header_only(self):
        """A 30 MB source must cost milliseconds: the check must not decode pixels."""
        raw = _image(4000, 3000, "JPEG")
        with patch.object(
            Image.Image, "load", side_effect=AssertionError("pixels decoded")
        ):
            check_h3_image(raw)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
        timeout=180,
    )


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("h3clips")
    video = ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3"]
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    h264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    out = {}

    def make(name, *args):
        path = d / name
        _ffmpeg(*args, str(path))
        out[name] = path.read_bytes()

    make("ok.mp4", *video, *tone, *h264, "-c:a", "aac", "-shortest")
    make("ok.mov", *video, *h264)
    make("mp3_audio.mp4", *video, *tone, *h264, "-c:a", "libmp3lame", "-shortest")
    make("bad_container.mkv", *video, *h264)
    make("bad_vcodec.mp4", *video, "-c:v", "mpeg4")
    make("bad_acodec.mov", *video, *tone, *h264, "-c:a", "pcm_s16le", "-shortest")
    make(
        "bad_fps.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=15:duration=3",
        *h264,
    )
    make(
        "bad_side.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=200x200:rate=30:duration=3",
        *h264,
    )
    make(
        "bad_aspect.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1000x300:rate=30:duration=3",
        *h264,
    )
    make(
        "short.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30:duration=1",
        *h264,
    )
    make("ok.wav", *tone)
    make("ok.mp3", *tone, "-c:a", "libmp3lame")
    make("bad.ogg", *tone, "-c:a", "libvorbis")
    if _ffmpeg_has_encoder("libx265"):
        make(
            "ok_hevc.mp4",
            *video,
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
        )
    return out


class TestReferenceVideoCard:
    def test_mp4_h264_aac_is_admitted_and_measured(self, clips):
        assert 2.9 < check_h3_reference_video(clips["ok.mp4"]) < 3.2

    def test_mov_is_admitted(self, clips):
        assert 2.9 < check_h3_reference_video(clips["ok.mov"]) < 3.2

    def test_mp3_audio_track_is_admitted(self, clips):
        check_h3_reference_video(clips["mp3_audio.mp4"])

    def test_hevc_is_admitted(self, clips):
        if "ok_hevc.mp4" not in clips:
            pytest.skip("ffmpeg without libx265")
        assert probe_media(clips["ok_hevc.mp4"]).video_codec == "hevc"
        check_h3_reference_video(clips["ok_hevc.mp4"])

    def test_probe_reports_the_fields_the_card_needs(self, clips):
        probe = probe_media(clips["ok.mp4"])
        assert {"mp4", "mov"} <= probe.containers
        assert (probe.video_codec, probe.width, probe.height) == ("h264", 640, 360)
        assert probe.fps == pytest.approx(30.0)
        assert probe.audio_codec == "aac"
        assert probe.duration_s == pytest.approx(3.0, abs=0.2)

    def test_mkv_is_refused(self, clips):
        with pytest.raises(ValueError, match=r"MP4 \(\.mp4\) or MOV"):
            check_h3_reference_video(clips["bad_container.mkv"])

    def test_mpeg4_part2_is_refused(self, clips):
        with pytest.raises(ValueError, match="video codec 'mpeg4'"):
            check_h3_reference_video(clips["bad_vcodec.mp4"])

    def test_pcm_audio_track_is_refused(self, clips):
        with pytest.raises(ValueError, match="audio codec 'pcm_s16le'"):
            check_h3_reference_video(clips["bad_acodec.mov"])

    def test_15_fps_is_refused(self, clips):
        with pytest.raises(ValueError, match=r"frame rate is 15.000 fps"):
            check_h3_reference_video(clips["bad_fps.mp4"])

    def test_small_frame_is_refused(self, clips):
        with pytest.raises(ValueError, match=r"200x200 px"):
            check_h3_reference_video(clips["bad_side.mp4"])

    def test_wide_aspect_is_refused(self, clips):
        with pytest.raises(ValueError, match="aspect ratio"):
            check_h3_reference_video(clips["bad_aspect.mp4"])

    def test_audio_file_is_not_a_video(self, clips):
        with pytest.raises(ValueError, match="MP4"):
            check_h3_reference_video(clips["ok.wav"])

    def test_oversize_is_a_size_error(self, clips, monkeypatch):
        monkeypatch.setattr(policy, "MINIMAX_H3_VIDEO_MAX_BYTES", 1000)
        with pytest.raises(MediaTooLargeError, match="at most 1000 bytes"):
            check_h3_reference_video(clips["ok.mp4"], label="references.videos[1]")

    def test_short_clip_fails_the_duration_window(self, clips):
        seconds = check_h3_reference_video(clips["short.mp4"])
        assert seconds == pytest.approx(1.0, abs=0.2)
        with pytest.raises(ValueError, match=r"videos\[0\]"):
            check_reference_clip_durations(
                video_durations=[seconds], audio_durations=[]
            )

    def test_junk_is_refused(self):
        with pytest.raises(ValueError, match="could not be probed"):
            check_h3_reference_video(b"\x00" * 64)


class TestReferenceAudioCard:
    def test_wav_and_mp3_are_admitted_and_measured(self, clips):
        assert 2.9 < check_h3_reference_audio(clips["ok.wav"]) < 3.2
        assert 2.9 < check_h3_reference_audio(clips["ok.mp3"]) < 3.2

    def test_ogg_is_refused(self, clips):
        with pytest.raises(ValueError, match="WAV or MP3"):
            check_h3_reference_audio(clips["bad.ogg"])

    def test_video_file_is_not_audio(self, clips):
        with pytest.raises(ValueError, match="WAV or MP3"):
            check_h3_reference_audio(clips["ok.mp4"])

    def test_oversize_is_a_size_error(self, clips, monkeypatch):
        monkeypatch.setattr(policy, "MINIMAX_H3_AUDIO_MAX_BYTES", 1000)
        with pytest.raises(MediaTooLargeError, match="at most 1000 bytes"):
            check_h3_reference_audio(clips["ok.wav"])


def _fl2va_settings():
    settings = MagicMock()
    settings.model_runner = "tt-minimax-h3-fl2va"
    return settings


class TestSchemas:
    def test_ref2va_admits_a_256px_image(self):
        from domain.video_ref2va_generate_request import (
            MediaSource,
            MultimodalReferences,
        )

        refs = MultimodalReferences(images=[MediaSource(b64=_b64(_image(256, 256)))])
        assert len(refs.images) == 1

    def test_ref2va_refuses_a_1px_image(self):
        from domain.video_ref2va_generate_request import (
            MediaSource,
            MultimodalReferences,
        )

        with pytest.raises(ValidationError, match=r"images\[0\] is 1x1 px"):
            MultimodalReferences(images=[MediaSource(b64=_b64(_image(1, 1)))])

    def test_ref2va_refuses_a_gif(self):
        from domain.video_ref2va_generate_request import (
            MediaSource,
            MultimodalReferences,
        )

        with pytest.raises(ValidationError, match="format GIF is not supported"):
            MultimodalReferences(
                images=[MediaSource(b64=_b64(_image(256, 256, "GIF")))]
            )

    def test_ref2va_refuses_more_than_twelve_references_in_total(self):
        from domain.video_ref2va_generate_request import (
            MediaSource,
            MultimodalReferences,
        )

        url = lambda n: MediaSource(url=f"https://cdn.example.com/{n}")  # noqa: E731
        MultimodalReferences(
            images=[url(i) for i in range(9)], videos=[url(i) for i in range(3)]
        )  # 12: the pipeline's ceiling, admitted
        with pytest.raises(ValidationError, match="12 references in total"):
            MultimodalReferences(
                images=[url(i) for i in range(9)],
                videos=[url(i) for i in range(3)],
                audios=[url(0)],
            )

    @patch("domain.video_i2v_generate_request.get_settings", _fl2va_settings)
    def test_fl2va_keyframes_follow_the_card(self):
        from domain.video_i2v_generate_request import ImagePromptEntry

        ImagePromptEntry(image=_b64(_image(256, 256)), frame_pos=0)
        with pytest.raises(ValidationError, match=r"\[256, 5760\] px"):
            ImagePromptEntry(image=_b64(_image(1, 1)), frame_pos=0)
        with pytest.raises(ValidationError, match="not supported"):
            ImagePromptEntry(image=_b64(_image(256, 256, "BMP")), frame_pos=0)

    def test_wan_keeps_the_generic_decodability_check(self):
        from domain.video_i2v_generate_request import ImagePromptEntry

        # Not an H3 deployment (default settings): a 1x1 pixel is still a decodable image.
        ImagePromptEntry(image=_b64(_image(1, 1)), frame_pos=0)


class TestEndpointHelpers:
    def test_openapi_placeholder_is_admissible(self):
        from open_ai_api import video as video_api

        image = check_h3_image(base64.b64decode(video_api._OPENAPI_IMAGE_PLACEHOLDER))
        assert image.size == (256, 256)

    def test_upload_cap_and_content_types_follow_the_card(self):
        from open_ai_api import video as video_api

        assert video_api._MAX_UPLOAD_BYTES == MINIMAX_H3_IMAGE_MAX_BYTES
        assert {
            "image/heic",
            "image/heif",
            "image/webp",
        } <= video_api._ALLOWED_IMAGE_CONTENT_TYPES

    def test_media_limits_admit_a_good_clip(self, clips):
        from open_ai_api import video as video_api

        request = SimpleNamespace(
            references=SimpleNamespace(
                videos=[SimpleNamespace(b64=_b64(clips["ok.mp4"]))],
                audios=[SimpleNamespace(b64=_b64(clips["ok.wav"]))],
            )
        )
        video_api._enforce_ref2va_media_limits(request)

    def test_media_limits_refuse_a_bad_container_with_422(self, clips):
        from fastapi import HTTPException
        from open_ai_api import video as video_api

        request = SimpleNamespace(
            references=SimpleNamespace(
                videos=[SimpleNamespace(b64=_b64(clips["bad_container.mkv"]))],
                audios=[],
            )
        )
        with pytest.raises(HTTPException) as info:
            video_api._enforce_ref2va_media_limits(request)
        assert info.value.status_code == 422
        assert "references.videos[0]" in info.value.detail

    def test_media_limits_refuse_oversize_with_413(self, clips, monkeypatch):
        from fastapi import HTTPException
        from open_ai_api import video as video_api

        monkeypatch.setattr(policy, "MINIMAX_H3_AUDIO_MAX_BYTES", 1000)
        request = SimpleNamespace(
            references=SimpleNamespace(
                videos=[], audios=[SimpleNamespace(b64=_b64(clips["ok.wav"]))]
            )
        )
        with pytest.raises(HTTPException) as info:
            video_api._enforce_ref2va_media_limits(request)
        assert info.value.status_code == 413

    def test_media_limits_enforce_the_combined_window(self, clips):
        from fastapi import HTTPException
        from open_ai_api import video as video_api

        # three 3 s clips plus three 3 s soundtracks = 9 s each: fine; six clips of one kind = 18 s: not
        six = [SimpleNamespace(b64=_b64(clips["ok.mp4"]))] * 6
        with pytest.raises(HTTPException) as info:
            video_api._enforce_ref2va_media_limits(
                SimpleNamespace(references=SimpleNamespace(videos=six, audios=[]))
            )
        assert info.value.status_code == 422
        assert "combined videos duration" in info.value.detail

    def test_media_limits_refuse_garbage_base64_with_422(self):
        from fastapi import HTTPException
        from open_ai_api import video as video_api

        request = SimpleNamespace(
            references=SimpleNamespace(videos=[SimpleNamespace(b64="A")], audios=[])
        )
        with pytest.raises(HTTPException) as info:
            video_api._enforce_ref2va_media_limits(request)
        assert info.value.status_code == 422
        assert "references.videos[0]" in info.value.detail

    def test_no_references_is_a_no_op(self):
        from open_ai_api import video as video_api

        video_api._enforce_ref2va_media_limits(SimpleNamespace(prompt="x"))

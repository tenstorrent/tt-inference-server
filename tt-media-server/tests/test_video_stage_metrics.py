# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the video denoise / VAE decode stage metrics.

``telemetry.video_stage_metrics`` imports neither tt-metal nor
``telemetry.telemetry_client``, so these need no device and are immune to the
``sys.modules`` Mock other test modules park under the latter. The
request-level ``tt_media_server_video_*`` family is a different module and is
covered by ``tests/test_video_metrics.py``.

Prometheus collectors are process-global and cumulative, so each test uses its
own ``model_type`` label value.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from prometheus_client import REGISTRY
from telemetry.video_stage_metrics import (
    VideoStageRecorder,
    frames_shape,
    record_video_stages,
)


# Matched by class name, not import; mirrors tt_dit/pipelines/events.py.
@dataclass(frozen=True)
class SectionStart:
    name: str


@dataclass(frozen=True)
class SectionEnd:
    name: str


@dataclass(frozen=True)
class DenoiseStep:
    step: int
    total: int
    sigma: float


class FakeFrame:
    def __init__(self, width=832, height=480):
        self.size = (width, height)


class FakeTensor:
    """Stands in for a decoded frame tensor without allocating one."""

    def __init__(self, *shape):
        self.shape = shape


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


def labels(model_type, resolution="832x480"):
    return dict(model_type=model_type, device_id="0", resolution=resolution)


def drive_full_run(recorder, steps=4):
    """Replay the event order the tt_dit video pipelines actually emit."""
    recorder(SectionStart("encoder"))
    recorder(SectionStart("t5_encoding"))
    recorder(SectionEnd("t5_encoding"))
    recorder(SectionEnd("encoder"))
    recorder(SectionStart("prepare_latents"))
    recorder(SectionEnd("prepare_latents"))
    recorder(SectionStart("denoising"))
    for i in range(steps):
        recorder(SectionStart(f"denoising_step_{i}"))
        recorder(SectionEnd(f"denoising_step_{i}"))
    recorder(SectionEnd("denoising"))
    recorder(SectionStart("vae"))
    recorder(SectionEnd("vae"))


class TestFramesShape:
    def test_reads_the_channels_last_tensor_the_wan_vae_returns(self):
        # (B, T, H, W, C) -- what d2h_permute produces for uint8 / np output.
        assert frames_shape(FakeTensor(1, 81, 480, 832, 3)) == (
            "832x480",
            81,
            832 * 480,
        )

    def test_reads_a_channels_first_tensor(self):
        # (B, C, T, H, W) -- the un-permuted "pt" output type.
        assert frames_shape(FakeTensor(1, 3, 81, 480, 832)) == (
            "832x480",
            81,
            832 * 480,
        )

    def test_counts_every_video_in_the_batch(self):
        # Throughput is frames off the VAE, not videos.
        assert frames_shape(FakeTensor(2, 81, 480, 832, 3))[1] == 162

    def test_reads_a_batchless_tensor(self):
        assert frames_shape(FakeTensor(81, 480, 832, 3)) == ("832x480", 81, 832 * 480)
        assert frames_shape(FakeTensor(81, 3, 480, 832)) == ("832x480", 81, 832 * 480)

    def test_reads_a_real_numpy_array(self):
        assert frames_shape(np.zeros((1, 5, 4, 8, 3), dtype=np.uint8)) == (
            "8x4",
            5,
            32,
        )

    def test_reads_a_list_of_pil_videos(self):
        video = [FakeFrame(), FakeFrame(), FakeFrame()]
        assert frames_shape([video, video]) == ("832x480", 6, 832 * 480)

    def test_reads_a_bare_frame_list(self):
        assert frames_shape([FakeFrame(64, 64), FakeFrame(64, 64)]) == (
            "64x64",
            2,
            4096,
        )

    def test_an_exported_path_is_unknown_not_one_frame(self):
        # LTX and MiniMax-H3 return [path]; counting that as a frame would put
        # a bogus 1 into the throughput numerator.
        assert frames_shape(["/tmp/out.mp4"]) == ("unknown", 0, 0)

    @pytest.mark.parametrize(
        "frames",
        [None, [], object(), FakeTensor(), FakeTensor(4, 4), FakeTensor(9, 9, 9, 9)],
    )
    def test_unreadable_output_is_unknown_not_an_error(self, frames):
        assert frames_shape(frames) == ("unknown", 0, 0)

    def test_a_zero_dimension_is_rejected(self):
        assert frames_shape(FakeTensor(1, 81, 0, 832, 3)) == ("unknown", 0, 0)

    def test_channels_last_wins_when_both_ends_could_be_channels(self):
        # Documented precedence, not a defence: the last axis is checked first
        # because that is what the tt_dit VAE produces. A (B, C, T, H, W) whose
        # width is genuinely 3 or 4 pixels would be misread -- accepted, since
        # no video is three pixels wide, and pinned here so the heuristic is at
        # least specified rather than incidental.
        assert frames_shape(FakeTensor(1, 3, 81, 480, 3)) == ("480x81", 3, 480 * 81)


class TestVideoStageRecorder:
    def test_exports_both_stages_with_frames_and_pixels(self):
        model = "stages-happy-path"
        recorder = VideoStageRecorder(model_type=model, device_id="0")
        drive_full_run(recorder)
        recorder.flush(FakeTensor(1, 81, 480, 832, 3))

        assert (
            sample(
                "tt_media_server_video_denoise_duration_seconds_count", **labels(model)
            )
            == 1
        )
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count",
                **labels(model),
            )
            == 1
        )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) == 81
        assert sample("tt_media_server_video_vae_pixels_total", **labels(model)) == (
            81 * 832 * 480
        )

    def test_each_span_is_timed_exactly_and_nothing_else_is(self, monkeypatch):
        # A scripted clock, so this proves the sections are what gets timed.
        # Asserting "vae_seconds is small" would also pass if the whole call
        # were timed, because the whole call is fast in a test.
        clock = {"t": 0.0}
        monkeypatch.setattr("telemetry.video_stage_metrics._now", lambda: clock["t"])
        model = "stages-scripted-clock"
        recorder = VideoStageRecorder(model_type=model, device_id="0")

        recorder(SectionStart("encoder"))
        clock["t"] = 5.0  # encoding: measured by nothing
        recorder(SectionEnd("encoder"))
        recorder(SectionStart("denoising"))
        clock["t"] = 95.0
        recorder(SectionEnd("denoising"))
        recorder(SectionStart("vae"))
        clock["t"] = 97.5
        recorder(SectionEnd("vae"))
        clock["t"] = 1000.0  # postprocessing after both spans closed
        recorder.flush(FakeTensor(1, 81, 480, 832, 3))

        assert recorder.denoise_seconds == pytest.approx(90.0)
        assert recorder.vae_seconds == pytest.approx(2.5)
        assert sample(
            "tt_media_server_video_denoise_duration_seconds_sum", **labels(model)
        ) == pytest.approx(90.0)
        assert sample(
            "tt_media_server_video_vae_decode_duration_seconds_sum", **labels(model)
        ) == pytest.approx(2.5)

    def test_the_unmeasured_sections_are_ignored(self):
        recorder = VideoStageRecorder(model_type="stages-ignored", device_id="0")
        for name in ("encoder", "prepare_latents", "t5_encoding"):
            recorder(SectionStart(name))
            recorder(SectionEnd(name))
        assert recorder.denoise_seconds is None
        assert recorder.vae_seconds is None

    def test_ignores_events_without_a_name(self):
        recorder = VideoStageRecorder(model_type="stages-nameless", device_id="0")
        recorder(DenoiseStep(step=1, total=4, sigma=0.5))
        assert recorder.denoise_seconds is None
        assert recorder.vae_seconds is None

    def test_a_decode_that_raised_still_reports_its_denoise_loop(self):
        model = "stages-decode-crashed"
        recorder = VideoStageRecorder(model_type=model, device_id="0")
        recorder(SectionStart("denoising"))
        recorder(SectionEnd("denoising"))
        recorder(SectionStart("vae"))  # decode raised; no SectionEnd
        recorder.flush(FakeTensor(1, 81, 480, 832, 3))

        assert (
            sample(
                "tt_media_server_video_denoise_duration_seconds_count", **labels(model)
            )
            == 1
        )
        # No timed decode, so no decode latency and no frames credited to one.
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count",
                **labels(model),
            )
            is None
        )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) is None

    def test_a_run_that_closed_no_span_exports_nothing(self):
        model = "stages-nothing-closed"
        recorder = VideoStageRecorder(model_type=model, device_id="0")
        recorder(SectionStart("denoising"))  # died in the loop
        recorder.flush(FakeTensor(1, 81, 480, 832, 3))

        assert (
            sample(
                "tt_media_server_video_denoise_duration_seconds_count", **labels(model)
            )
            is None
        )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) is None

    def test_unprobeable_output_falls_back_to_the_configured_resolution(self):
        model = "stages-fallback"
        recorder = VideoStageRecorder(
            model_type=model, device_id="0", resolution="1280x720"
        )
        recorder(SectionStart("vae"))
        recorder(SectionEnd("vae"))
        recorder.flush(["/tmp/out.mp4"])

        fallback = labels(model, resolution="1280x720")
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count", **fallback
            )
            == 1
        )
        # Latency is still known; throughput is not, and is skipped rather
        # than recorded as zero.
        assert sample("tt_media_server_video_vae_frames_total", **fallback) is None

    def test_unprobeable_output_with_no_fallback_is_labelled_unknown(self):
        model = "stages-no-fallback"
        recorder = VideoStageRecorder(model_type=model, device_id="0")
        recorder(SectionStart("vae"))
        recorder(SectionEnd("vae"))
        recorder.flush(None)

        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count",
                **labels(model, resolution="unknown"),
            )
            == 1
        )

    def test_a_missing_device_id_is_labelled_unknown(self):
        model = "stages-no-device"
        recorder = VideoStageRecorder(model_type=model, device_id=None)
        recorder(SectionStart("vae"))
        recorder(SectionEnd("vae"))
        recorder.flush(FakeTensor(1, 2, 4, 8, 3))

        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count",
                model_type=model,
                device_id="unknown",
                resolution="8x4",
            )
            == 1
        )

    def test_flush_never_raises_into_the_inference_path(self):
        class Exploding:
            @property
            def shape(self):
                raise RuntimeError("boom")

        recorder = VideoStageRecorder(model_type="stages-explode", device_id="0")
        recorder(SectionStart("vae"))
        recorder(SectionEnd("vae"))
        recorder.flush(Exploding())  # must not propagate


class TestRecordVideoStages:
    def test_frames_without_a_pixel_count_skip_the_pixel_series(self):
        model = "stages-no-pixels"
        record_video_stages(
            model_type=model,
            device_id="0",
            resolution="832x480",
            vae_seconds=2.0,
            frame_count=81,
            pixels_per_frame=0,
        )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) == 81
        assert sample("tt_media_server_video_vae_pixels_total", **labels(model)) is None

    def test_frames_with_no_timed_decode_are_not_counted(self):
        # A denoise-only export must not credit frames to a decode that was
        # never measured, or frames/s divides by a missing denominator.
        model = "stages-denoise-only"
        record_video_stages(
            model_type=model,
            device_id="0",
            resolution="832x480",
            denoise_seconds=90.0,
            frame_count=81,
            pixels_per_frame=10,
        )
        assert (
            sample(
                "tt_media_server_video_denoise_duration_seconds_count", **labels(model)
            )
            == 1
        )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) is None

    def test_accumulates_across_generations(self):
        model = "stages-cumulative"
        for _ in range(3):
            record_video_stages(
                model_type=model,
                device_id="0",
                resolution="832x480",
                denoise_seconds=90.0,
                vae_seconds=2.0,
                frame_count=81,
                pixels_per_frame=10,
            )
        assert sample("tt_media_server_video_vae_frames_total", **labels(model)) == 243
        assert sample(
            "tt_media_server_video_vae_decode_duration_seconds_sum", **labels(model)
        ) == pytest.approx(6.0)
        assert sample(
            "tt_media_server_video_denoise_duration_seconds_sum", **labels(model)
        ) == pytest.approx(270.0)

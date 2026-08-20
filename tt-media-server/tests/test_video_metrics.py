# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the tt_media_server_video_* metric family.

Every test uses its own ``model_type`` label value so the module-level
prometheus_client collectors (shared process-wide, and cumulative by design)
cannot leak counts between tests. That lets each case assert absolute values
instead of deltas.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from prometheus_client import REGISTRY
from telemetry.telemetry_client import (
    VIDEO_REQUEST_TYPE_I2V,
    VIDEO_REQUEST_TYPE_T2V,
    VIDEO_STATUS_CANCELLED,
    VIDEO_STATUS_FAILURE,
    VIDEO_STATUS_SUCCESS,
    VIDEO_UNKNOWN_LABEL,
    TelemetryClient,
    VideoGenerationStats,
)
from utils.video_manager import probe_video


def sample(name: str, **labels) -> float:
    """Metric value as a plain float; a series that does not exist reads 0."""
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


def make_client(model_type: str) -> TelemetryClient:
    """A telemetry client wired to a test-only model_type label.

    Built without ``__init__`` so no background worker thread is started and no
    real ``Settings`` is resolved: the synchronous recorders under test only
    read ``settings.model_runner``.
    """
    client = TelemetryClient.__new__(TelemetryClient)
    client.settings = SimpleNamespace(model_runner=model_type, enable_telemetry=True)
    client.queue = None
    client.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return client


def sync_client(model_type: str) -> TelemetryClient:
    """Client whose async recorder records inline, so tests need no worker."""
    client = make_client(model_type)
    client.record_video_generation_async = client.record_video_generation
    return client


def write_mp4(path, num_frames=8, width=64, height=32, fps=16) -> str:
    """Encode a real (tiny) h264 mp4 with PyAV — no ffmpeg binary required."""
    import av

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    for _ in range(num_frames):
        frame = av.VideoFrame.from_ndarray(
            np.zeros((height, width, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


class TestResolutionLabel:
    """VideoGenerationStats.resolution keeps the label set bounded."""

    def test_known_dimensions(self):
        stats = VideoGenerationStats(
            request_type=VIDEO_REQUEST_TYPE_T2V,
            duration_seconds=1.0,
            width=832,
            height=480,
        )
        assert stats.resolution == "832x480"

    def test_missing_dimensions_fall_back_to_unknown(self):
        stats = VideoGenerationStats(
            request_type=VIDEO_REQUEST_TYPE_T2V, duration_seconds=1.0
        )
        assert stats.resolution == VIDEO_UNKNOWN_LABEL

    def test_zero_dimensions_are_not_a_resolution(self):
        stats = VideoGenerationStats(
            request_type=VIDEO_REQUEST_TYPE_T2V,
            duration_seconds=1.0,
            width=0,
            height=0,
        )
        assert stats.resolution == VIDEO_UNKNOWN_LABEL


class TestSuccessfulGeneration:
    """A fully-probed success populates every metric in the family."""

    MODEL = "test_video_success"

    # Class-scoped: these are cumulative counters, so recording once per test
    # method would multiply every expected value by the method count.
    @pytest.fixture(scope="class", autouse=True)
    def recorded(self, request):
        make_client(request.cls.MODEL).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V,
                duration_seconds=64.0,
                num_inference_steps=20,
                width=832,
                height=480,
                num_frames=81,
                content_seconds=5.0625,
                output_bytes=1_500_000,
            )
        )

    def test_outcome_counted_as_success(self):
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status="success",
            )
            == 1.0
        )

    def test_duration_carries_resolution_and_status(self):
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_sum",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution="832x480",
                status="success",
            )
            == 64.0
        )

    def test_frames_and_steps_totals(self):
        assert (
            sample(
                "tt_media_server_video_frames_generated_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 81.0
        )
        assert (
            sample(
                "tt_media_server_video_denoise_steps_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 20.0
        )

    def test_content_seconds_and_bytes_totals(self):
        assert sample(
            "tt_media_server_video_content_seconds_total",
            model_type=self.MODEL,
            request_type=VIDEO_REQUEST_TYPE_T2V,
        ) == pytest.approx(5.0625)
        assert (
            sample(
                "tt_media_server_video_output_bytes_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 1_500_000
        )

    @pytest.mark.parametrize(
        "metric, expected",
        [
            ("tt_media_server_video_frames_per_second_sum", 81 / 64.0),
            ("tt_media_server_video_step_duration_seconds_sum", 64.0 / 20),
            ("tt_media_server_video_realtime_factor_sum", 64.0 / 5.0625),
            ("tt_media_server_video_pixels_per_second_sum", 832 * 480 * 81 / 64.0),
        ],
    )
    def test_derived_throughputs(self, metric, expected):
        assert sample(
            metric,
            model_type=self.MODEL,
            request_type=VIDEO_REQUEST_TYPE_T2V,
            resolution="832x480",
        ) == pytest.approx(expected)

    def test_shape_distributions(self):
        assert (
            sample(
                "tt_media_server_video_output_frames_sum",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 81.0
        )
        assert (
            sample(
                "tt_media_server_video_requested_inference_steps_sum",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 20.0
        )
        assert (
            sample(
                "tt_media_server_video_output_size_bytes_count",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 1.0
        )


class TestFailedGeneration:
    """A failure must not be mistaken for slow work.

    Regression guard for the split found on hardware: a request that timed out
    after 6s of a 300s budget did not execute its 16 steps in 0.4s each, so
    executed-steps and per-step latency stay untouched while the ask is still
    recorded.
    """

    MODEL = "test_video_failure"

    # Class-scoped: these are cumulative counters, so recording once per test
    # method would multiply every expected value by the method count.
    @pytest.fixture(scope="class", autouse=True)
    def recorded(self, request):
        make_client(request.cls.MODEL).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V,
                duration_seconds=6.0,
                status=VIDEO_STATUS_FAILURE,
                num_inference_steps=16,
            )
        )

    def test_counted_as_failure(self):
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status="failure",
            )
            == 1.0
        )

    def test_time_to_failure_is_its_own_series(self):
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_sum",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution=VIDEO_UNKNOWN_LABEL,
                status="failure",
            )
            == 6.0
        )
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_count",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution=VIDEO_UNKNOWN_LABEL,
                status="success",
            )
            == 0.0
        )

    def test_requested_steps_still_recorded(self):
        assert (
            sample(
                "tt_media_server_video_requested_inference_steps_sum",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 16.0
        )

    def test_executed_steps_not_counted(self):
        assert (
            sample(
                "tt_media_server_video_denoise_steps_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 0.0
        )

    def test_step_latency_not_polluted(self):
        assert (
            sample(
                "tt_media_server_video_step_duration_seconds_count",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution=VIDEO_UNKNOWN_LABEL,
            )
            == 0.0
        )

    def test_nothing_was_produced(self):
        for metric in (
            "tt_media_server_video_frames_generated_total",
            "tt_media_server_video_content_seconds_total",
            "tt_media_server_video_output_bytes_total",
        ):
            assert (
                sample(
                    metric,
                    model_type=self.MODEL,
                    request_type=VIDEO_REQUEST_TYPE_T2V,
                )
                == 0.0
            )


class TestUnknownFields:
    """Unknown probe fields are skipped, never guessed as zero."""

    MODEL = "test_video_partial"

    @pytest.fixture(scope="class", autouse=True)
    def recorded(self, request):
        # Shape unknown (probe failed) but the file size is still readable.
        make_client(request.cls.MODEL).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V,
                duration_seconds=30.0,
                num_inference_steps=20,
                output_bytes=900_000,
            )
        )

    def test_duration_recorded_under_unknown_resolution(self):
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_count",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution=VIDEO_UNKNOWN_LABEL,
                status="success",
            )
            == 1.0
        )

    def test_known_field_still_recorded(self):
        assert (
            sample(
                "tt_media_server_video_output_bytes_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 900_000
        )

    def test_frame_derived_metrics_skipped(self):
        for metric in (
            "tt_media_server_video_frames_per_second_count",
            "tt_media_server_video_pixels_per_second_count",
            "tt_media_server_video_realtime_factor_count",
        ):
            assert (
                sample(
                    metric,
                    model_type=self.MODEL,
                    request_type=VIDEO_REQUEST_TYPE_T2V,
                    resolution=VIDEO_UNKNOWN_LABEL,
                )
                == 0.0
            )


class TestZeroDuration:
    """A zero/absent duration must not divide into a throughput histogram."""

    MODEL = "test_video_zero_duration"

    # Class-scoped: these are cumulative counters, so recording once per test
    # method would multiply every expected value by the method count.
    @pytest.fixture(scope="class", autouse=True)
    def recorded(self, request):
        make_client(request.cls.MODEL).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V,
                duration_seconds=0.0,
                num_inference_steps=20,
                width=832,
                height=480,
                num_frames=81,
                content_seconds=5.0,
                output_bytes=1000,
            )
        )

    def test_outcome_still_counted(self):
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status="success",
            )
            == 1.0
        )

    def test_no_duration_observation(self):
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_count",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution="832x480",
                status="success",
            )
            == 0.0
        )

    def test_no_derived_throughputs(self):
        for metric in (
            "tt_media_server_video_frames_per_second_count",
            "tt_media_server_video_step_duration_seconds_count",
            "tt_media_server_video_realtime_factor_count",
            "tt_media_server_video_pixels_per_second_count",
        ):
            assert (
                sample(
                    metric,
                    model_type=self.MODEL,
                    request_type=VIDEO_REQUEST_TYPE_T2V,
                    resolution="832x480",
                )
                == 0.0
            )

    def test_frame_totals_still_counted(self):
        assert (
            sample(
                "tt_media_server_video_frames_generated_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 81.0
        )


class TestConditioningImages:
    """The I2V image count is recorded only when images were supplied."""

    def test_i2v_images_recorded(self):
        model = "test_video_i2v_images"
        make_client(model).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_I2V,
                duration_seconds=10.0,
                conditioning_images=2,
            )
        )
        assert (
            sample("tt_media_server_video_conditioning_images_sum", model_type=model)
            == 2.0
        )

    def test_t2v_does_not_record_zero(self):
        model = "test_video_t2v_images"
        make_client(model).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V, duration_seconds=10.0
            )
        )
        assert (
            sample("tt_media_server_video_conditioning_images_count", model_type=model)
            == 0.0
        )


class TestRecordVideoEncode:
    """ffmpeg encode accounting."""

    def test_success_records_duration_and_fps(self):
        model = "test_video_encode_ok"
        make_client(model).record_video_encode(
            duration=2.0, num_frames=32, width=832, height=480
        )
        assert (
            sample(
                "tt_media_server_video_encode_total", model_type=model, status="success"
            )
            == 1.0
        )
        assert (
            sample(
                "tt_media_server_video_encode_duration_seconds_sum",
                model_type=model,
                resolution="832x480",
                status="success",
            )
            == 2.0
        )
        assert sample(
            "tt_media_server_video_encode_frames_per_second_sum",
            model_type=model,
            resolution="832x480",
        ) == pytest.approx(16.0)

    def test_failed_encode_labelled_failure(self):
        model = "test_video_encode_fail"
        make_client(model).record_video_encode(duration=0.5, status=False)
        assert (
            sample(
                "tt_media_server_video_encode_total", model_type=model, status="failure"
            )
            == 1.0
        )
        assert (
            sample(
                "tt_media_server_video_encode_duration_seconds_count",
                model_type=model,
                resolution=VIDEO_UNKNOWN_LABEL,
                status="failure",
            )
            == 1.0
        )


class TestClassifyRequest:
    """Request type comes from the payload, not the endpoint or the runner."""

    @staticmethod
    def _classify(request):
        from model_services.video_service import VideoService

        return VideoService._classify_request(request)

    def test_text_only_request_is_t2v(self):
        from domain.video_generate_request import VideoGenerateRequest

        request = VideoGenerateRequest(prompt="a fox in the snow")
        assert self._classify(request) == VIDEO_REQUEST_TYPE_T2V

    def test_image_conditioned_request_is_i2v(self):
        from domain.video_i2v_generate_request import (
            ImagePromptEntry,
            VideoI2VGenerateRequest,
        )

        png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        request = VideoI2VGenerateRequest(
            prompt="a fox turning to camera",
            image_prompts=[ImagePromptEntry(image=png, frame_pos=0)],
        )
        assert self._classify(request) == VIDEO_REQUEST_TYPE_I2V


class TestProbeVideo:
    """probe_video is the only source of shape truth on the SP_RUNNER path."""

    def test_reads_shape_and_length(self, tmp_path):
        path = write_mp4(tmp_path / "clip.mp4", num_frames=8, width=64, height=32)
        probe = probe_video(path)
        assert (probe.width, probe.height) == (64, 32)
        assert probe.num_frames == 8
        assert probe.duration_seconds == pytest.approx(0.5, abs=0.1)
        assert probe.size_bytes > 0

    def test_missing_file_yields_empty_probe(self, tmp_path):
        probe = probe_video(str(tmp_path / "absent.mp4"))
        assert probe.size_bytes is None
        assert probe.num_frames is None

    def test_non_video_file_yields_size_only(self, tmp_path):
        path = tmp_path / "notavideo.mp4"
        path.write_bytes(b"definitely not an mp4")
        probe = probe_video(str(path))
        assert probe.size_bytes == len(b"definitely not an mp4")
        assert probe.width is None
        assert probe.num_frames is None

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_non_path_inputs_are_safe(self, bad):
        probe = probe_video(bad)
        assert probe.size_bytes is None


class TestCancelledGeneration:
    """A cancelled job is neither success nor pipeline failure.

    POST /generations/{id}/cancel is a normal client action; counting it as a
    failure would make the dashboard's success rate track cancel traffic.
    """

    MODEL = "test_video_cancelled"

    @pytest.fixture(scope="class", autouse=True)
    def recorded(self, request):
        make_client(request.cls.MODEL).record_video_generation(
            VideoGenerationStats(
                request_type=VIDEO_REQUEST_TYPE_T2V,
                duration_seconds=4.0,
                status=VIDEO_STATUS_CANCELLED,
                num_inference_steps=20,
            )
        )

    def test_has_its_own_outcome_series(self):
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status=VIDEO_STATUS_CANCELLED,
            )
            == 1.0
        )

    def test_not_counted_as_failure_or_success(self):
        for status in (VIDEO_STATUS_SUCCESS, VIDEO_STATUS_FAILURE):
            assert (
                sample(
                    "tt_media_server_video_generation_total",
                    model_type=self.MODEL,
                    request_type=VIDEO_REQUEST_TYPE_T2V,
                    status=status,
                )
                == 0.0
            )

    def test_no_executed_steps_or_frames(self):
        assert (
            sample(
                "tt_media_server_video_denoise_steps_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 0.0
        )
        assert (
            sample(
                "tt_media_server_video_frames_generated_total",
                model_type=self.MODEL,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 0.0
        )


class TestProcessRequestGauge:
    """The in-flight gauge must be released on every exit path.

    Regression guard: JobManager cancels the task behind /cancel, and
    asyncio.CancelledError derives from BaseException — an `except Exception`
    around the pipeline would leak the gauge upward on every cancelled job,
    leaving a permanent phantom concurrency on the dashboard.
    """

    GAUGE = "tt_media_server_video_generations_in_progress"

    @staticmethod
    def _service(monkeypatch, model_type, outcome):
        """A VideoService with the heavy __init__ and the base pipeline stubbed."""
        from model_services import video_service as module
        from model_services.base_service import BaseService

        async def pipeline(_self, _request):
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(BaseService, "process_request", pipeline)
        monkeypatch.setattr(
            module, "get_telemetry_client", lambda: sync_client(model_type)
        )

        service = module.VideoService.__new__(module.VideoService)
        service.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        return service

    def _gauge(self):
        from config.settings import settings

        return sample(
            self.GAUGE,
            model_type=settings.model_runner,
            request_type=VIDEO_REQUEST_TYPE_T2V,
        )

    async def test_cancellation_releases_gauge(self, monkeypatch):
        from domain.video_generate_request import VideoGenerateRequest

        model = "test_video_cancel_path"
        service = self._service(monkeypatch, model, asyncio.CancelledError())
        before = self._gauge()

        with pytest.raises(asyncio.CancelledError):
            await service.process_request(VideoGenerateRequest(prompt="cancel me"))

        assert self._gauge() == before
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=model,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status=VIDEO_STATUS_CANCELLED,
            )
            == 1.0
        )

    async def test_failure_releases_gauge(self, monkeypatch):
        from domain.video_generate_request import VideoGenerateRequest

        model = "test_video_failure_path"
        service = self._service(monkeypatch, model, RuntimeError("runner exploded"))
        before = self._gauge()

        with pytest.raises(RuntimeError):
            await service.process_request(VideoGenerateRequest(prompt="fail me"))

        assert self._gauge() == before
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=model,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status=VIDEO_STATUS_FAILURE,
            )
            == 1.0
        )

    async def test_success_probes_the_produced_mp4(self, monkeypatch, tmp_path):
        from domain.video_generate_request import VideoGenerateRequest

        model = "test_video_success_path"
        mp4 = write_mp4(tmp_path / "out.mp4", num_frames=8, width=64, height=32)
        service = self._service(monkeypatch, model, mp4)
        before = self._gauge()

        result = await service.process_request(
            VideoGenerateRequest(prompt="a fox", num_inference_steps=8)
        )

        assert result == mp4
        assert self._gauge() == before
        assert (
            sample(
                "tt_media_server_video_generation_total",
                model_type=model,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                status=VIDEO_STATUS_SUCCESS,
            )
            == 1.0
        )
        # Frame count and resolution come from the mp4, not from the request.
        assert (
            sample(
                "tt_media_server_video_frames_generated_total",
                model_type=model,
                request_type=VIDEO_REQUEST_TYPE_T2V,
            )
            == 8.0
        )
        assert (
            sample(
                "tt_media_server_video_generation_duration_seconds_count",
                model_type=model,
                request_type=VIDEO_REQUEST_TYPE_T2V,
                resolution="64x32",
                status=VIDEO_STATUS_SUCCESS,
            )
            == 1.0
        )

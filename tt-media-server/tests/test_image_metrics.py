# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the image-generation stage metrics.

``telemetry.image_metrics`` imports neither tt-metal nor
``telemetry.telemetry_client``, so these need no device and are immune to the
``sys.modules`` Mock other test modules park under the latter.

Prometheus collectors are process-global and cumulative, so each test uses its
own ``model_type`` label value.
"""

import sys
import types
from dataclasses import dataclass

import pytest
from prometheus_client import REGISTRY
from telemetry.image_metrics import (
    ImageStageRecorder,
    SdxlSectionTimings,
    add_conditioning_seconds,
    format_resolution,
    record_image_run,
    resolution_of_images,
    sampler_name,
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


class FakeImage:
    def __init__(self, width=1024, height=1024):
        self.size = (width, height)


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels)


def denoise_labels(model_type, **overrides):
    labels = dict(
        model_type=model_type,
        device_id="0",
        resolution="1024x1024",
        sampler="euler-solver",
        batch="1",
    )
    labels.update(overrides)
    return labels


def shape_labels(model_type, **overrides):
    labels = dict(
        model_type=model_type,
        device_id="0",
        resolution="1024x1024",
        batch="1",
    )
    labels.update(overrides)
    return labels


def drive_full_run(recorder, steps=4, encoders=("clip", "t5")):
    """Replay the event order the tt_dit image pipelines actually emit."""
    recorder(SectionStart("total"))
    recorder(SectionStart("encoder"))
    for encoder in encoders:
        recorder(SectionStart(f"{encoder}_encoding"))
        recorder(SectionEnd(f"{encoder}_encoding"))
    recorder(SectionEnd("encoder"))
    recorder(SectionStart("denoising"))
    for i in range(steps):
        recorder(SectionStart(f"denoising_step_{i}"))
        recorder(SectionEnd(f"denoising_step_{i}"))
    recorder(SectionEnd("denoising"))
    recorder(SectionStart("vae"))
    recorder(SectionEnd("vae"))
    recorder(SectionEnd("total"))


class TestResolutionHelpers:
    def test_formats_width_by_height(self):
        assert format_resolution(1024, 768) == "1024x768"

    @pytest.mark.parametrize(
        "width,height", [(None, 512), (512, None), (0, 512), (-1, 5)]
    )
    def test_rejects_unusable_dimensions(self, width, height):
        assert format_resolution(width, height) == "unknown"

    def test_reads_shape_off_produced_images(self):
        assert resolution_of_images([FakeImage(512, 512), FakeImage(512, 512)]) == (
            "512x512",
            2,
            512 * 512,
        )

    def test_accepts_a_bare_image(self):
        assert resolution_of_images(FakeImage(64, 64)) == ("64x64", 1, 4096)

    def test_missing_output_is_unknown_not_an_error(self):
        assert resolution_of_images(None) == ("unknown", 0, 0)
        assert resolution_of_images([]) == ("unknown", 0, 0)

    def test_output_without_a_size_still_counts_images(self):
        assert resolution_of_images([object()]) == ("unknown", 1, 0)


class TestSamplerName:
    def test_prefers_the_public_diffusers_scheduler(self):
        class EulerDiscreteScheduler:
            pass

        pipeline = types.SimpleNamespace(scheduler=EulerDiscreteScheduler())
        assert sampler_name(pipeline) == "euler-discrete-scheduler"

    def test_falls_back_to_the_tt_dit_solver(self):
        class EulerSolver:
            pass

        assert (
            sampler_name(types.SimpleNamespace(_solvers=[EulerSolver()]))
            == "euler-solver"
        )

    def test_unknown_when_the_pipeline_exposes_neither(self):
        assert sampler_name(object()) == "unknown"
        assert sampler_name(None) == "unknown"


class TestAddConditioningSeconds:
    def test_nested_image_encode_folds_into_all(self):
        store = {}
        add_conditioning_seconds(store, "all", 0.4)
        add_conditioning_seconds(store, "image", 0.6)
        assert store["image"] == pytest.approx(0.6)
        assert store["all"] == pytest.approx(1.0)

    def test_all_does_not_double_count_itself(self):
        store = {}
        add_conditioning_seconds(store, "all", 0.4)
        add_conditioning_seconds(store, "all", 0.1)
        assert store["all"] == pytest.approx(0.5)
        assert list(store) == ["all"]


class TestImageStageRecorder:
    def test_records_every_stage_of_a_full_run(self):
        model_type = "recorder-full"
        recorder = ImageStageRecorder(model_type, "0", sampler="euler-solver", batch=1)
        drive_full_run(recorder, steps=4)
        recorder.flush([FakeImage()])

        labels = denoise_labels(model_type)
        assert sample("tt_media_server_image_denoise_steps_total", **labels) == 4
        assert (
            sample(
                "tt_media_server_image_denoise_step_duration_seconds_count", **labels
            )
            == 4
        )
        assert (
            sample("tt_media_server_image_denoise_duration_seconds_count", **labels)
            == 1
        )

        shape = shape_labels(model_type)
        assert (
            sample("tt_media_server_image_vae_decode_duration_seconds_count", **shape)
            == 1
        )
        assert sample("tt_media_server_image_vae_images_total", **shape) == 1
        assert sample("tt_media_server_image_vae_pixels_total", **shape) == 1024 * 1024
        assert (
            sample("tt_media_server_image_engine_duration_seconds_count", **shape) == 1
        )

    def test_reports_the_wrapper_and_each_nested_encoder_separately(self):
        model_type = "recorder-encoders"
        recorder = ImageStageRecorder(model_type, "0", sampler="euler-solver", batch=1)
        drive_full_run(recorder, encoders=("clip", "t5"))
        recorder.flush([FakeImage()])

        for encoder in ("all", "clip", "t5"):
            count = sample(
                "tt_media_server_image_conditioning_duration_seconds_count",
                model_type=model_type,
                device_id="0",
                encoder=encoder,
                batch="1",
            )
            assert count == 1, f"missing conditioning series for {encoder}"

    def test_accumulates_nested_encoders_when_cfg_encodes_twice(self, monkeypatch):
        """SD3.5 / Motif encode_cfg runs clip+t5 for the prompt, then again for the negative."""
        clock = {"t": 0.0}
        monkeypatch.setattr("telemetry.image_metrics._now", lambda: clock["t"])

        recorder = ImageStageRecorder(
            "recorder-cfg", "0", sampler="euler-solver", batch=1
        )
        clock["t"] = 0.0
        recorder(SectionStart("encoder"))
        recorder(SectionStart("clip_encoding"))
        clock["t"] = 0.10
        recorder(SectionEnd("clip_encoding"))
        recorder(SectionStart("t5_encoding"))
        clock["t"] = 0.40
        recorder(SectionEnd("t5_encoding"))
        recorder(SectionStart("clip_encoding"))
        clock["t"] = 0.52
        recorder(SectionEnd("clip_encoding"))
        recorder(SectionStart("t5_encoding"))
        clock["t"] = 0.82
        recorder(SectionEnd("t5_encoding"))
        recorder(SectionEnd("encoder"))

        assert recorder.conditioning_seconds["clip"] == pytest.approx(0.22)
        assert recorder.conditioning_seconds["t5"] == pytest.approx(0.60)
        assert recorder.conditioning_seconds["all"] == pytest.approx(0.82)

        recorder.flush([FakeImage()])
        assert sample(
            "tt_media_server_image_conditioning_duration_seconds_sum",
            model_type="recorder-cfg",
            device_id="0",
            encoder="clip",
            batch="1",
        ) == pytest.approx(0.22)

    def test_maps_the_qwen_encoder_span(self):
        model_type = "recorder-qwen"
        recorder = ImageStageRecorder(model_type, "0", sampler="euler-solver", batch=1)
        drive_full_run(recorder, encoders=("qwen",))
        recorder.flush([FakeImage()])

        assert (
            sample(
                "tt_media_server_image_conditioning_duration_seconds_count",
                model_type=model_type,
                device_id="0",
                encoder="qwen",
                batch="1",
            )
            == 1
        )

    def test_publishes_nothing_until_flush(self):
        model_type = "recorder-nopublish"
        recorder = ImageStageRecorder(model_type, "0", sampler="euler-solver", batch=1)
        drive_full_run(recorder)

        assert recorder.step_seconds
        assert (
            sample(
                "tt_media_server_image_denoise_steps_total",
                **denoise_labels(model_type),
            )
            is None
        )

    def test_an_unclosed_section_is_never_reported(self):
        recorder = ImageStageRecorder("recorder-unclosed", "0")
        recorder(SectionStart("denoising"))
        recorder(SectionStart("denoising_step_0"))
        # Run dies here: no SectionEnd arrives.
        assert recorder.denoise_seconds is None
        assert recorder.step_seconds == []

    def test_ignores_events_without_a_section_name(self):
        recorder = ImageStageRecorder("recorder-denoisestep", "0")
        recorder(DenoiseStep(step=1, total=10, sigma=0.5))
        assert recorder.step_seconds == []
        assert recorder.engine_seconds is None

    def test_unknown_resolution_when_output_shape_is_unreadable(self):
        model_type = "recorder-noshape"
        recorder = ImageStageRecorder(model_type, "0", sampler="euler-solver", batch=1)
        drive_full_run(recorder, steps=2)
        recorder.flush(None)

        labels = denoise_labels(model_type, resolution="unknown")
        assert sample("tt_media_server_image_denoise_steps_total", **labels) == 2
        # No shape, so no pixel/image throughput is invented.
        assert (
            sample(
                "tt_media_server_image_vae_images_total",
                **shape_labels(model_type, resolution="unknown"),
            )
            is None
        )

    def test_missing_device_id_is_labelled_unknown(self):
        model_type = "recorder-nodevice"
        recorder = ImageStageRecorder(model_type, None, sampler="euler-solver", batch=1)
        drive_full_run(recorder, steps=1)
        recorder.flush([FakeImage()])

        assert (
            sample(
                "tt_media_server_image_denoise_steps_total",
                **denoise_labels(model_type, device_id="unknown"),
            )
            == 1
        )

    def test_flush_swallows_telemetry_failures(self):
        recorder = ImageStageRecorder("recorder-raises", "0")
        drive_full_run(recorder, steps=1)
        # A bad resolution type must not propagate into the inference path.
        recorder.flush(images=object(), resolution=object())


class TestRecordImageRun:
    def test_step_count_does_not_fabricate_per_step_latency(self):
        model_type = "record-fallback"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            denoise_seconds=4.0,
            step_count=20,
        )
        labels = denoise_labels(model_type)
        assert sample("tt_media_server_image_denoise_steps_total", **labels) == 20
        assert (
            sample("tt_media_server_image_denoise_duration_seconds_count", **labels)
            == 1
        )
        assert sample(
            "tt_media_server_image_denoise_duration_seconds_sum", **labels
        ) == pytest.approx(4.0)
        assert (
            sample(
                "tt_media_server_image_denoise_step_duration_seconds_count", **labels
            )
            is None
        )

    def test_step_count_without_a_loop_time_still_advances_the_counter(self):
        model_type = "record-stepsonly"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            engine_seconds=12.0,
            step_count=9,
        )
        labels = denoise_labels(model_type)
        assert sample("tt_media_server_image_denoise_steps_total", **labels) == 9
        # Per-step latency is unknown here and must not be fabricated.
        assert (
            sample(
                "tt_media_server_image_denoise_step_duration_seconds_count", **labels
            )
            is None
        )

    def test_per_step_observations_win_over_step_count(self):
        model_type = "record-perstep"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            denoise_seconds=1.0,
            step_seconds=[0.1, 0.2, 0.3],
            step_count=99,
        )
        labels = denoise_labels(model_type)
        assert sample("tt_media_server_image_denoise_steps_total", **labels) == 3
        assert sample(
            "tt_media_server_image_denoise_step_duration_seconds_sum", **labels
        ) == pytest.approx(0.6)

    def test_batch_is_stringified_for_the_label(self):
        model_type = "record-batch"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="512x512",
            sampler="euler-solver",
            batch=4,
            step_count=1,
        )
        assert (
            sample(
                "tt_media_server_image_denoise_steps_total",
                **denoise_labels(model_type, resolution="512x512", batch="4"),
            )
            == 1
        )

    def test_pixels_scale_with_the_image_count(self):
        model_type = "record-pixels"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="512x512",
            sampler="euler-solver",
            batch=2,
            vae_seconds=1.0,
            image_count=2,
            pixels_per_image=512 * 512,
        )
        shape = shape_labels(model_type, resolution="512x512", batch="2")
        assert sample("tt_media_server_image_vae_images_total", **shape) == 2
        assert (
            sample("tt_media_server_image_vae_pixels_total", **shape) == 2 * 512 * 512
        )

    def test_image_encoder_is_exported_separately_from_all(self):
        model_type = "record-image-enc"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            engine_seconds=5.0,
            conditioning_seconds={"all": 1.0, "image": 0.6},
        )
        for encoder, expected_sum in (("all", 1.0), ("image", 0.6)):
            assert (
                sample(
                    "tt_media_server_image_conditioning_duration_seconds_count",
                    model_type=model_type,
                    device_id="0",
                    encoder=encoder,
                    batch="1",
                )
                == 1
            )
            assert sample(
                "tt_media_server_image_conditioning_duration_seconds_sum",
                model_type=model_type,
                device_id="0",
                encoder=encoder,
                batch="1",
            ) == pytest.approx(expected_sum)

    def test_absent_stages_produce_no_series(self):
        model_type = "record-sparse"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            engine_seconds=5.0,
        )
        labels = denoise_labels(model_type)
        assert sample("tt_media_server_image_denoise_steps_total", **labels) is None
        assert (
            sample("tt_media_server_image_denoise_duration_seconds_count", **labels)
            is None
        )
        shape = shape_labels(model_type)
        assert (
            sample("tt_media_server_image_vae_decode_duration_seconds_count", **shape)
            is None
        )
        assert (
            sample("tt_media_server_image_engine_duration_seconds_count", **shape) == 1
        )

    def test_long_observations_land_in_real_buckets_not_inf(self):
        # A 90s run must stay distinguishable from a 900s one.
        model_type = "record-buckets"
        record_image_run(
            model_type=model_type,
            device_id="0",
            resolution="1024x1024",
            sampler="euler-solver",
            batch=1,
            engine_seconds=90.0,
            denoise_seconds=80.0,
            step_seconds=[20.0, 20.0, 20.0, 20.0],
            vae_seconds=8.0,
            conditioning_seconds={"all": 2.0},
        )
        labels = denoise_labels(model_type)
        assert (
            sample(
                "tt_media_server_image_denoise_step_duration_seconds_bucket",
                le="30.0",
                **labels,
            )
            == 4
        )
        shape = shape_labels(model_type)
        assert (
            sample(
                "tt_media_server_image_engine_duration_seconds_bucket",
                le="120.0",
                **shape,
            )
            == 1
        )


class TestSdxlSectionTimings:
    @pytest.fixture
    def profiler(self):
        """Install a stand-in for tt-metal's global Profiler singleton."""
        saved = {
            k: sys.modules.get(k)
            for k in ("models", "models.common", "models.common.utility_functions")
        }

        class FakeProfiler:
            def __init__(self):
                self.times = {}

        module = types.ModuleType("models.common.utility_functions")
        module.profiler = FakeProfiler()
        sys.modules["models"] = types.ModuleType("models")
        sys.modules["models.common"] = types.ModuleType("models.common")
        sys.modules["models.common.utility_functions"] = module
        try:
            yield module.profiler
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_reads_the_spans_recorded_during_the_call(self, profiler):
        with SdxlSectionTimings() as timings:
            profiler.times["denoising_loop"] = [4.0]
            profiler.times["vae_decode"] = [1.5]
            profiler.times["image_gen"] = [5.5]

        assert timings.denoise_seconds == 4.0
        assert timings.vae_seconds == 1.5
        assert timings.engine_seconds == 5.5

    def test_discards_spans_left_over_from_warmup(self, profiler):
        profiler.times["denoising_loop"] = [99.0]
        profiler.times["vae_decode"] = [99.0]
        profiler.times["image_gen"] = [99.0]

        with SdxlSectionTimings() as timings:
            assert "denoising_loop" not in profiler.times
            profiler.times["denoising_loop"] = [4.0]

        assert timings.denoise_seconds == 4.0
        assert timings.vae_seconds is None

    def test_takes_the_last_span_when_several_were_recorded(self, profiler):
        with SdxlSectionTimings() as timings:
            profiler.times["denoising_loop"] = [1.0, 2.0, 3.0]
        assert timings.denoise_seconds == 3.0

    def test_a_failed_run_reports_nothing(self, profiler):
        timings = SdxlSectionTimings()
        with pytest.raises(RuntimeError):
            with timings:
                profiler.times["denoising_loop"] = [4.0]
                raise RuntimeError("inference blew up")

        assert timings.denoise_seconds is None
        assert timings.engine_seconds is None

    def test_does_not_suppress_the_exception(self, profiler):
        with pytest.raises(ValueError):
            with SdxlSectionTimings():
                raise ValueError("must propagate")

    def test_no_tt_metal_means_no_readings_and_no_crash(self):
        for key in ("models.common.utility_functions", "models.common", "models"):
            sys.modules.pop(key, None)
        with SdxlSectionTimings() as timings:
            pass
        assert timings.denoise_seconds is None
        assert timings.vae_seconds is None
        assert timings.engine_seconds is None

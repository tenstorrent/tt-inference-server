# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The dit_runners side of the video stage metrics: does each video runner
actually hand its pipeline an ``on_event`` recorder and flush it afterwards?

``telemetry.video_stage_metrics`` is exercised on its own in
``tests/test_video_stage_metrics.py``; this file covers the wiring, which is
what silently rots when a runner is added or a call site is rewritten.

``tests/conftest.py`` parks a stub module under
``sys.modules["tt_model_runners.dit_runners"]`` holding seven placeholder
classes, so a plain import cannot reach the real runners. The file is loaded
from disk under a private name instead, leaving that stub in place for every
other test. The tt-metal packages it imports at module scope are faked for the
duration of that load and removed straight after.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from prometheus_client import REGISTRY

_SOURCE = Path(__file__).resolve().parent.parent / "tt_model_runners" / "dit_runners.py"

_TT_METAL_MODULES = (
    "models",
    "models.common",
    "models.common.utility_functions",
    "models.tt_dit",
    "models.tt_dit.pipelines",
    "models.tt_dit.pipelines.flux1",
    "models.tt_dit.pipelines.flux1.pipeline_flux1",
    "models.tt_dit.pipelines.minimax_h3",
    "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3",
    "models.tt_dit.pipelines.mochi",
    "models.tt_dit.pipelines.mochi.pipeline_mochi",
    "models.tt_dit.pipelines.motif",
    "models.tt_dit.pipelines.motif.pipeline_motif",
    "models.tt_dit.pipelines.qwenimage",
    "models.tt_dit.pipelines.qwenimage.pipeline_qwenimage",
    "models.tt_dit.pipelines.stable_diffusion_35_large",
    "models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large",
    "models.tt_dit.pipelines.wan",
    "models.tt_dit.pipelines.wan.pipeline_wan",
    "models.tt_dit.pipelines.wan.pipeline_wan_i2v",
)


# conftest replaces utils.logger with a stub exposing only TTLogger; the
# runner module also imports log_exception_chain from it.
_LOGGER_ATTRS = ("log_exception_chain",)


def _settings_stub():
    """A settings module the runner file can be imported against.

    Several test modules park a bare ``Mock`` under
    ``sys.modules["config.settings"]``, and ``dit_runners`` looks
    ``get_settings().model_runner`` up in ``dit_runner_log_map`` while its
    classes are being defined -- against a Mock that is a KeyError. Installing
    a real value for the load keeps this file independent of collection order.
    """
    settings = SimpleNamespace(
        model_runner="sp_runner",
        model_service="video",
        device_mesh_shape=(1, 1),
        enable_telemetry=False,
    )
    module = ModuleType("config.settings")
    module.settings = settings
    module.get_settings = lambda: settings
    return module


def _load_real_dit_runners():
    faked = [name for name in _TT_METAL_MODULES if name not in sys.modules]
    original_ttnn = sys.modules.get("ttnn")
    original_settings = sys.modules.get("config.settings")
    logger_module = sys.modules.get("utils.logger")
    patched_attrs = [
        attr
        for attr in _LOGGER_ATTRS
        if logger_module is not None and not hasattr(logger_module, attr)
    ]
    for name in faked:
        sys.modules[name] = Mock()
    for attr in patched_attrs:
        setattr(logger_module, attr, Mock())
    sys.modules.setdefault("ttnn", Mock())
    sys.modules["config.settings"] = _settings_stub()
    try:
        spec = importlib.util.spec_from_file_location("_real_dit_runners", _SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in faked:
            sys.modules.pop(name, None)
        for attr in patched_attrs:
            delattr(logger_module, attr)
        if original_ttnn is None:
            sys.modules.pop("ttnn", None)
        if original_settings is None:
            sys.modules.pop("config.settings", None)
        else:
            sys.modules["config.settings"] = original_settings


dit_runners = _load_real_dit_runners()


@pytest.fixture(autouse=True)
def _no_telemetry_client(monkeypatch):
    """``log_execution_time`` wraps every ``run()`` and reports to the
    telemetry client in a ``finally``. Not what is under test, and it may hold
    a real queue depending on collection order."""
    import utils.decorators

    monkeypatch.setattr(utils.decorators, "get_telemetry_client", Mock())


class SectionStart(SimpleNamespace):
    def __init__(self, name):
        super().__init__(name=name)


class SectionEnd(SimpleNamespace):
    def __init__(self, name):
        super().__init__(name=name)


class FakeTensor:
    """A decoded (B, T, H, W, C) video, without allocating one."""

    def __init__(self, *shape):
        self.shape = shape


class StubPipeline:
    """Replays the sections the real tt_dit video pipelines emit."""

    def __init__(self, frames):
        self.frames = frames
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        on_event = kwargs.get("on_event")
        if on_event is not None:
            for name in ("encoder", "denoising", "vae"):
                on_event(SectionStart(name))
                on_event(SectionEnd(name))
        return self.frames


# Every video runner whose pipeline accepts ``on_event``: the Wan2.2 family
# (the I2V variants inherit ``WanPipeline.__call__``) plus Mochi. Deliberately
# not the Prodia distills, LTX or MiniMax-H3, which take no ``on_event``.
WIRED_RUNNERS = [
    "TTWan22Runner",
    "TTWan22I2VRunner",
    "TTWan22I2VAniSoraRunner",
    "TTWan22I2VDistillRunner",
    "TTWan22I2VLoRARunner",
    "TTWan22I2VLightningRunner",
    "TTMochi1Runner",
]


def build_runner(class_name, model_type, warming_up=False):
    """Construct a runner without touching a device or the base __init__."""
    cls = getattr(dit_runners, class_name)
    runner = cls.__new__(cls)
    runner.device_id = "0"
    runner._warming_up = warming_up
    runner.logger = Mock()
    runner.settings = SimpleNamespace(model_runner=model_type)
    runner.resolution = SimpleNamespace(width=832, height=480)
    runner.export_in_runner = False
    runner.pipeline = StubPipeline(FakeTensor(1, 81, 480, 832, 3))
    # The I2V runners decode a conditioning image through ImageManager; not
    # what is under test here.
    runner._build_image_prompt = Mock(return_value=[])
    return runner


def a_request():
    return SimpleNamespace(
        prompt="a cat",
        negative_prompt="",
        num_inference_steps=4,
        seed=0,
        image_prompts=[],
    )


def sample(name, model_type, resolution="832x480"):
    return REGISTRY.get_sample_value(
        name,
        dict(model_type=model_type, device_id="0", resolution=resolution),
    )


@pytest.mark.parametrize("class_name", WIRED_RUNNERS)
class TestRunnerWiring:
    def test_passes_a_recorder_and_exports_the_decode(self, class_name):
        model_type = f"wiring-{class_name}"
        runner = build_runner(class_name, model_type)

        runner.run([a_request()])

        recorder = runner.pipeline.kwargs.get("on_event")
        assert isinstance(recorder, dit_runners.VideoStageRecorder), (
            f"{class_name} did not pass an on_event recorder to its pipeline"
        )
        assert (
            sample("tt_media_server_video_denoise_duration_seconds_count", model_type)
            == 1
        )
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count", model_type
            )
            == 1
        )
        assert sample("tt_media_server_video_vae_frames_total", model_type) == 81
        assert sample("tt_media_server_video_vae_pixels_total", model_type) == (
            81 * 832 * 480
        )

    def test_warmup_is_not_recorded(self, class_name):
        model_type = f"wiring-warmup-{class_name}"
        runner = build_runner(class_name, model_type, warming_up=True)

        runner.run([a_request()])

        assert runner.pipeline.kwargs.get("on_event") is None
        assert (
            sample("tt_media_server_video_denoise_duration_seconds_count", model_type)
            is None
        )
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count", model_type
            )
            is None
        )


class TestMochiFlushSurvivesConversion:
    """Mochi stacks the pipeline's PIL output into an array after the decode.

    That conversion is outside the pipeline call, so a failure there used to
    swallow a decode that had already happened and been timed.
    """

    def test_a_failed_frame_stack_still_records_the_decode(self):
        model_type = "wiring-mochi-stack-failure"
        runner = build_runner("TTMochi1Runner", model_type)
        # np.stack iterates each "video"; a bare object is not iterable.
        runner.pipeline = StubPipeline([object()])

        with pytest.raises(TypeError):
            runner.run([a_request()])

        # Shape is unprobeable, so the label falls back to unknown and the
        # throughput counters stay empty -- but the timing is not lost.
        assert (
            sample(
                "tt_media_server_video_vae_decode_duration_seconds_count",
                model_type,
                resolution="unknown",
            )
            == 1
        )
        assert (
            sample(
                "tt_media_server_video_vae_frames_total",
                model_type,
                resolution="unknown",
            )
            is None
        )


class TestResolutionLabel:
    def test_reads_a_runner_resolution(self):
        assert (
            dit_runners._resolution_label(SimpleNamespace(width=1280, height=720))
            == "1280x720"
        )

    def test_no_resolution_means_no_fallback_label(self):
        assert dit_runners._resolution_label(None) is None
        assert dit_runners._resolution_label(SimpleNamespace()) is None

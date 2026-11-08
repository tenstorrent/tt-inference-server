# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""What the generations API's runners pass their tt_dit pipelines, and the
warmup requests they build. The pipelines are stubs; the runner module is
loaded against fake tt-metal modules, as in test_video_stage_metrics_wiring."""

import importlib.util
import pathlib
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.flux3_image_klein_request import Flux3ImageKleinRequest
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.ideogram4_request import Ideogram4Request
from domain.minimax_h3_request import (
    MiniMaxH3FL2VARequest,
    MiniMaxH3Ref2VARequest,
)
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import (
    WAN_NEGATIVE_PROMPT,
    WAN_SIZES,
    Wan22i2vA14bRequest,
)
from PIL import Image

_RUNNERS = Path(__file__).resolve().parent.parent / "tt_model_runners"

_TT_METAL_MODULES = (
    "models",
    "models.common",
    "models.common.utility_functions",
    "models.tt_dit",
    "models.tt_dit.pipelines",
    "models.tt_dit.pipelines.flux1",
    "models.tt_dit.pipelines.flux1.pipeline_flux1",
    "models.tt_dit.pipelines.flux1.pipeline_flux1_kontext",
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


def _load_generation_runners():
    settings = SimpleNamespace(
        model_runner="tt-wan2.2-i2v",
        model_service="generation",
        device_mesh_shape=(2, 4),
        enable_telemetry=False,
    )
    settings_module = ModuleType("config.settings")
    settings_module.settings = settings
    settings_module.get_settings = lambda: settings

    replaced = {
        name: sys.modules.get(name)
        for name in (
            *_TT_METAL_MODULES,
            "ttnn",
            "config.settings",
            "tt_model_runners.dit_runners",
        )
    }
    logger_module = sys.modules.get("utils.logger")
    patch_logger = logger_module is not None and not hasattr(
        logger_module, "log_exception_chain"
    )
    for name in _TT_METAL_MODULES:
        sys.modules.setdefault(name, Mock())
    sys.modules.setdefault("ttnn", Mock())
    sys.modules["config.settings"] = settings_module
    if patch_logger:
        logger_module.log_exception_chain = Mock()
    try:
        modules = {}
        for name in ("dit_runners", "generation_runners"):
            spec = importlib.util.spec_from_file_location(
                f"tt_model_runners.{name}", _RUNNERS / f"{name}.py"
            )
            modules[name] = importlib.util.module_from_spec(spec)
            sys.modules[f"tt_model_runners.{name}"] = modules[name]
            spec.loader.exec_module(modules[name])
        return modules["generation_runners"]
    finally:
        for name, module in replaced.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("tt_model_runners.generation_runners", None)
        if patch_logger:
            del logger_module.log_exception_chain


runners = _load_generation_runners()


class _Pipeline:
    def __init__(self):
        self.calls = []
        self.resolutions = [
            size for sizes in WAN_SIZES.values() for size in sizes.values()
        ]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return ["output"]

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        pathlib.Path(kwargs["output_path"]).write_bytes(b"mp4")
        return kwargs["output_path"]


def _runner(class_name: str, warming_up: bool = False):
    cls = getattr(runners, class_name)
    runner = cls.__new__(cls)
    runner.device_id = "0"
    runner._warming_up = warming_up
    runner._image_stage_recorder = lambda: None
    runner._video_stage_recorder = lambda resolution: None
    runner.logger = Mock()
    runner.settings = SimpleNamespace(model_runner="tt-wan2.2-i2v")
    runner.pipeline = _Pipeline()
    return runner


def _resolved(request, width, height, files=None, seed=5, prompt=None):
    request._width, request._height, request._seed = width, height, seed
    request._files = files or {}
    request._prompt = prompt if prompt is not None else request.prompt
    return request


def test_wan_passes_both_frames_as_they_are(monkeypatch):
    monkeypatch.setattr(
        runners, "ImagePrompt", lambda image, frame_pos: (image.size, frame_pos)
    )
    runner = _runner("TTWan22I2VA14BRunner")
    # The runner decodes them; the pipeline crops and resizes them.
    first, last = _png(2000, 720), _png(640, 360)
    request = _resolved(
        Wan22i2vA14bRequest(
            prompt="a boat",
            first_frame="https://x/a.png",
            last_frame="https://x/b.png",
            aspect_ratio="16:9",
            size="720p",
        ),
        1280,
        720,
        {"first_frame": [first], "last_frame": [last]},
    )
    assert runner.run([request]) == ["output"]
    (call,) = runner.pipeline.calls
    assert call.pop("image_prompt") == [((2000, 720), 0), ((640, 360), 80)]
    assert call == {
        "prompts": ["a boat"],
        "negative_prompts": [WAN_NEGATIVE_PROMPT],
        "num_inference_steps": 40,
        "guidance_scale": 3.5,
        "guidance_scale_2": 3.5,
        "flow_shift": 5.0,
        "seed": 5,
        "height": 720,
        "width": 1280,
        "traced": True,
        "on_event": None,
    }


def test_wan_warmup_covers_every_combination():
    runner = _runner("TTWan22I2VA14BRunner", warming_up=True)
    requests = runner.warmup_requests()
    # 4 sizes, with and without the last frame.
    assert len(requests) == 8
    for request in requests:
        runner.pipeline.calls.clear()
        runner.run([request])
        (call,) = runner.pipeline.calls
        assert (call["width"], call["height"]) == (request._width, request._height)
        assert call["num_inference_steps"] == 2


def test_wan_deep_canary_runs_a_short_warmup_request():
    from config.constants import CANARY_DEEP_TASK_ID

    # The multi-host lockstep runner and its followers run it outside warmup.
    runner = _runner("TTWan22I2VA14BRunner")
    request = runner._build_warmup_video_request()
    assert isinstance(request, Wan22i2vA14bRequest)
    assert request._task_id == CANARY_DEEP_TASK_ID
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert (call["width"], call["height"]) == (request._width, request._height)
    assert call["num_inference_steps"] == 2


@pytest.mark.parametrize(
    "mesh_shape, inherited, expected",
    [
        ((4, 8), {"trace_region_size": 125_000_000}, 512 * 1024 * 1024),
        ((4, 32), {"trace_region_size": 125_000_000}, 256 * 1024 * 1024),
        # No region reserved: the traces are allocated as they are captured.
        ((2, 2), {}, None),
    ],
)
def test_wan_trace_region_fits_every_bucket(
    monkeypatch, mesh_shape, inherited, expected
):
    runner = _runner("TTWan22I2VA14BRunner")
    runner.settings.device_mesh_shape = mesh_shape
    monkeypatch.setitem(
        runners.TTWan22I2VRunner.get_pipeline_device_params.__globals__,
        "_wan22_dit_device_params",
        lambda shape: {"fabric_config": "ring", **inherited},
    )
    params = runner.get_pipeline_device_params()
    assert params["fabric_config"] == "ring"
    assert params.get("trace_region_size") == expected


def test_wan_warmup_conditions_the_last_frame(monkeypatch):
    monkeypatch.setattr(
        runners, "ImagePrompt", lambda image, frame_pos: (image, frame_pos)
    )
    runner = _runner("TTWan22I2VA14BRunner", warming_up=True)
    runner.pipeline.resolutions = list(WAN_SIZES["480p"].values())
    requests = runner.warmup_requests()
    assert {(r._width, r._height) for r in requests} == set(WAN_SIZES["480p"].values())
    frame_positions = set()
    for request in requests:
        runner.pipeline.calls.clear()
        runner.run([request])
        (call,) = runner.pipeline.calls
        frame_positions.add(tuple(pos for _, pos in call["image_prompt"]))
    assert frame_positions == {(0,), (0, 80)}


@pytest.mark.parametrize("with_images", [False, True])
def test_klein_edits_only_with_images(with_images):
    runner = _runner("TTFlux2Klein4BRunner")
    images = {"images": [Image.new("RGB", (8, 8))]} if with_images else {}
    request = _resolved(Flux2klein4bRequest(prompt="a cat"), 1024, 1024)
    # The service decodes FLUX.2's images: the runner gets the images themselves.
    request._images = images
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["images"] == (images["images"] if with_images else None)
    assert call["num_inference_steps"] == 4


def test_klein_pipeline_warms_itself_up(monkeypatch):
    module = Mock()
    monkeypatch.setitem(
        sys.modules, "models.tt_dit.pipelines.flux2.pipeline_flux2", module
    )
    runner = _runner("TTFlux2Klein4BRunner")
    runner.ttnn_device = Mock()
    runner.settings = SimpleNamespace(
        model_weights_path="black-forest-labs/FLUX.2-klein-4B"
    )
    runner.create_pipeline()
    assert module.Flux2Pipeline.create_pipeline.call_args.kwargs["trace_warmup"] is True
    assert runner.warmup_requests() == []


class _Flux3Pipeline:
    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return SimpleNamespace(
            image=Image.new("RGB", (request.width, request.height)),
            prompt=f"a detailed caption of {request.prompt}",
        )


@pytest.fixture
def flux3_runner(monkeypatch):
    pipeline_module = ModuleType("pipeline_flux3_klein")
    pipeline_module.Flux3KleinRequest = SimpleNamespace
    monkeypatch.setitem(
        sys.modules,
        "models.tt_dit.pipelines.flux3_klein.pipeline_flux3_klein",
        pipeline_module,
    )
    runner = _runner("TTFlux3ImageKleinRunner")
    runner.pipeline = _Flux3Pipeline()
    return runner


@pytest.mark.parametrize("with_image", [False, True])
def test_flux3_passes_the_reference_as_it_is(flux3_runner, with_image):
    images = {"images": [Image.new("RGB", (3000, 1000))]} if with_image else {}
    request = _resolved(Flux3ImageKleinRequest(prompt="a cat"), 2048, 2048)
    # The service decodes FLUX.3's reference: the runner gets the image itself.
    request._images = images
    (image,) = flux3_runner.run([request])
    (call,) = flux3_runner.pipeline.calls
    assert (call.prompt, call.width, call.height, call.seed) == ("a cat", 2048, 2048, 5)
    assert call.steps == 16
    assert call.images == tuple(images.get("images", []))
    assert image.size == (2048, 2048)
    assert image.info["prompt"] == "a detailed caption of a cat"


def test_flux3_warmup_covers_every_size_with_and_without_a_reference(flux3_runner):
    flux3_runner._warming_up = True
    for request in flux3_runner.warmup_requests():
        flux3_runner.run([request])
    calls = flux3_runner.pipeline.calls
    assert {(call.width, call.height, len(call.images)) for call in calls} == {
        (side, side, refs) for side in (1024, 2048, 4096) for refs in (0, 1)
    }
    assert {call.steps for call in calls} == {2}


@pytest.mark.parametrize("warming_up", [False, True])
def test_ltx_writes_the_served_shape(monkeypatch, tmp_path, warming_up):
    monkeypatch.setenv("TT_GENERATIONS_DIR", str(tmp_path))
    runner = _runner("TTLtx23DistilledRunner", warming_up=warming_up)
    if warming_up:
        (request,) = runner.warmup_requests()
    else:
        request = _resolved(
            Ltx23DistilledRequest(prompt="a violinist"), 1920, 1088, seed=5
        )
    (path,) = runner.run([request])
    (call,) = runner.pipeline.calls
    assert call == {
        "prompt": request._prompt,
        "output_path": path,
        "num_frames": 153,
        "height": 1088,
        "width": 1920,
        "seed": request._seed,
        "fps": 25,
    }
    # Into the job's directory; warmup only captures the traces, and its
    # video is not kept.
    assert pathlib.Path(path).parent == tmp_path / request._task_id
    assert pathlib.Path(path).exists() is not warming_up
    assert pathlib.Path(path).parent.exists() is not warming_up


def test_qwen_passes_the_input_images():
    runner = _runner("TTQwenImageEdit2511Runner")
    images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    request = _resolved(
        QwenImageEdit2511Request(
            prompt="add a hat", negative_prompt="blurry", images=["https://x/a.png"] * 2
        ),
        1376,
        768,
    )
    # The service decodes Qwen's images: the runner gets the images themselves.
    request._images = {"images": images}
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["image"] is images
    assert (call["width"], call["height"], call["cfg_scale"]) == (1376, 768, 4.0)
    assert call["negative_prompts"] == ["blurry"]
    assert call["num_inference_steps"] == 40


def test_qwen_pipeline_warms_itself_up():
    assert _runner("TTQwenImageEdit2511Runner").warmup_requests() == []


def test_qwen_pipeline_runs_with_dbcache(monkeypatch):
    module = Mock()
    monkeypatch.setitem(
        sys.modules, "models.tt_dit.pipelines.qwenimage.pipeline_qwenimage", module
    )
    runner = _runner("TTQwenImageEdit2511Runner")
    runner.ttnn_device = SimpleNamespace(shape=(4, 8))
    runner.settings = SimpleNamespace(model_weights_path="Qwen/Qwen-Image-Edit-2511")
    assert runner.create_pipeline() is module.QwenImagePipeline.return_value
    kwargs = module.QwenImagePipelineConfig.default.call_args.kwargs
    assert kwargs == {
        "mesh_shape": (4, 8),
        "checkpoint_name": "Qwen/Qwen-Image-Edit-2511",
        "dbcache": module.DEFAULT_DBCACHE,
    }
    module.QwenImagePipeline.assert_called_once_with(
        device=runner.ttnn_device,
        config=module.QwenImagePipelineConfig.default.return_value,
    )


class _H3Pipeline:
    """Returns planar yuv420 frames and a stereo soundtrack, as the device
    stitch does."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        audio = np.zeros((2, 4800), dtype=np.float32)
        return SimpleNamespace(
            video_format="yuv420",
            video=np.zeros((kwargs["num_frames"], 12, 8), dtype=np.uint8),
            audio=[SimpleNamespace(numpy=lambda: audio)],
            sampling_rate=48000,
            fps=24,
        )


def _h3_runner(class_name: str):
    runner = _runner(class_name)
    runner.pipeline = _H3Pipeline()
    runner.settings = SimpleNamespace(
        model_runner="tt-minimax-h3-fl2va",
        model_weights_path=None,
        device_mesh_shape=(4, 8),
    )
    runner.ttnn_device = Mock()
    return runner


def _png(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def h3_exports(monkeypatch, tmp_path):
    """The (frames, audio, sampling_rate, kwargs) of each MP4 the runner
    exports, which it writes under ``tmp_path/encoded``."""
    monkeypatch.setenv("TT_GENERATIONS_DIR", str(tmp_path / "generations"))
    exports = []

    class _VideoManager:
        def export_to_mp4_with_audio(self, frames, audio, sampling_rate, **kwargs):
            exports.append((frames, audio, sampling_rate, kwargs))
            path = tmp_path / "encoded" / f"{len(exports)}.mp4"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"mp4")
            return str(path)

    monkeypatch.setattr(runners, "VideoManager", _VideoManager)
    return exports


def test_h3_passes_the_shape_and_writes_the_video_into_the_job(h3_exports, tmp_path):
    runner = _h3_runner("TTMiniMaxH3Runner")
    request = _resolved(
        MiniMaxH3FL2VARequest(prompt="waves", aspect_ratio="9:16", duration=6),
        768,
        1344,
    )
    (path,) = runner.run([request])
    (call,) = runner.pipeline.calls
    assert call == {
        "prompt": "waves",
        "num_frames": 158,
        "aspect_ratio": (9, 16),
        "num_inference_steps": 50,
        "seed": 5,
    }
    assert path == str(tmp_path / "generations" / request._task_id / "h3.mp4")
    assert pathlib.Path(path).read_bytes() == b"mp4"
    ((frames, audio, sampling_rate, kwargs),) = h3_exports
    assert frames.shape == (158, 12, 8)
    assert (audio.shape, sampling_rate) == ((2, 4800), 48000)
    assert kwargs == {"fps": 24, "pixel_format": "yuv420p"}


def test_h3_leaves_the_shape_of_an_adaptive_request_to_the_keyframe(h3_exports):
    runner = _h3_runner("TTMiniMaxH3Runner")
    request = _resolved(
        MiniMaxH3FL2VARequest(prompt="waves", first_frame="https://x/a.png"), 768, 1344
    )
    request._files = {"first_frame": [_png(32, 64)]}
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert "aspect_ratio" not in call


def test_h3_exports_nothing_where_no_job_keeps_the_video(h3_exports):
    runner = _h3_runner("TTMiniMaxH3Runner")
    # The deep canary, and the other ranks of a multi-host mesh.
    assert runner.run([runner._build_warmup_video_request()]) == [None]
    runner.export_in_runner = False
    request = _resolved(MiniMaxH3FL2VARequest(prompt="waves"), 1344, 768)
    assert runner.run([request]) == [None]
    assert h3_exports == []


@pytest.mark.parametrize(
    "keyframes, expected",
    [
        ({}, {}),
        ({"first_frame": [(64, 32)]}, {"image": (64, 32), "last_image": None}),
        ({"last_frame": [(32, 64)]}, {"image": None, "last_image": (32, 64)}),
        (
            {"first_frame": [(64, 32)], "last_frame": [(32, 64)]},
            {"image": (64, 32), "last_image": (32, 64)},
        ),
    ],
)
def test_h3_decodes_the_keyframes(keyframes, expected):
    runner = _h3_runner("TTMiniMaxH3TurboRunner")
    request = _resolved(MiniMaxH3FL2VARequest(prompt="waves"), 1344, 768)
    request._files = {
        name: [_png(*size) for size in sizes] for name, sizes in keyframes.items()
    }
    inputs = runner._inputs(request)
    assert {
        name: image if image is None else image.size for name, image in inputs.items()
    } == expected


def test_h3_ref2va_packs_images_then_videos_then_audio(monkeypatch):
    packing = ModuleType("packing_ref2va")
    packing.MiniMaxH3Reference = lambda **kwargs: kwargs
    packing.reference_from_video_file = lambda path: {
        "video": pathlib.Path(path).read_bytes()
    }
    packing.decode_reference_audio = lambda path: (
        pathlib.Path(path).read_bytes(),
        16000,
    )
    monkeypatch.setitem(
        sys.modules, "models.tt_dit.pipelines.minimax_h3.packing_ref2va", packing
    )
    runner = _h3_runner("TTMiniMaxH3Ref2VARunner")
    request = _resolved(
        MiniMaxH3Ref2VARequest(prompt="a song", images=["https://x/a.png"]), 1344, 768
    )
    request._files = {
        "images": [_png(48, 64)],
        "videos": [b"v0", b"v1"],
        "audios": [b"a0"],
    }
    (image, *clips) = runner._inputs(request)["references"]
    assert image["image"].size == (48, 64)
    assert clips == [
        {"video": b"v0"},
        {"video": b"v1"},
        {"audio": b"a0", "sample_rate": 16000},
    ]


def test_h3_undecodable_input_fails_only_the_request(monkeypatch, h3_exports):
    # On a multi-host mesh any other error ends the job on every rank.
    packing = ModuleType("packing_ref2va")
    packing.MiniMaxH3Reference = lambda **kwargs: kwargs

    def truncated(path):
        raise EOFError("end of stream")

    packing.reference_from_video_file = truncated
    packing.decode_reference_audio = Mock()
    monkeypatch.setitem(
        sys.modules, "models.tt_dit.pipelines.minimax_h3.packing_ref2va", packing
    )
    runner = _h3_runner("TTMiniMaxH3Ref2VARunner")
    request = _resolved(
        MiniMaxH3Ref2VARequest(prompt="a song", videos=["https://x/a.mp4"]), 1344, 768
    )
    request._files = {"videos": [b"v0"]}
    with pytest.raises(ValueError, match="could not be decoded: end of stream"):
        runner.run([request])
    assert runner.pipeline.calls == []


@pytest.mark.parametrize(
    "class_name, inputs",
    [
        ("TTMiniMaxH3Runner", {}),
        ("TTMiniMaxH3Ref2VARunner", {"images": [(768, 768)]}),
    ],
)
def test_h3_deep_canary_is_the_shortest_video(class_name, inputs):
    runner = _h3_runner(class_name)
    request = runner._build_warmup_video_request()
    assert (request.frames, request.aspect_ratio) == (107, "16:9")
    assert request._prompt
    assert {
        name: [runner._images(request, name)[0].size] for name in request._files
    } == inputs


# The Turbo runners, with the base runner each extends, its pipeline task and
# the (video, audio) shift its adapter was distilled at.
H3_TURBO_RUNNERS = [
    ("TTMiniMaxH3TurboRunner", "TTMiniMaxH3Runner", "t2va", (6.0, 3.0)),
    ("TTMiniMaxH3TurboRef2VARunner", "TTMiniMaxH3Ref2VARunner", "ref2va", (12.0, 3.0)),
]


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = tmp_path / "turbo.safetensors"
    path.write_bytes(b"")
    monkeypatch.setenv("MINIMAX_H3_LORA_PATH", str(path))
    monkeypatch.delenv("MINIMAX_H3_VIDEO_SHIFT", raising=False)
    monkeypatch.delenv("MINIMAX_H3_AUDIO_SHIFT", raising=False)
    return str(path)


@pytest.mark.parametrize("name, base, task, shifts", H3_TURBO_RUNNERS)
def test_h3_turbo_keeps_the_base_task_and_runs_five_grid_points(
    name, base, task, shifts
):
    cls, base_cls = getattr(runners, name), getattr(runners, base)
    assert issubclass(cls, base_cls)
    assert cls.pipeline_task == task
    assert cls.dit_fsdp == base_cls.dit_fsdp
    assert cls.num_inference_steps == 5


@pytest.mark.parametrize("name, base, task, shifts", H3_TURBO_RUNNERS)
def test_h3_turbo_pipeline_gets_the_adapter_and_its_shift(
    name, base, task, shifts, adapter, monkeypatch
):
    turbo = Mock()
    monkeypatch.setattr(
        runners._MiniMaxH3TurboMixin, "_pipeline_factory", lambda self: turbo
    )
    _h3_runner(name).create_pipeline()
    kwargs = turbo.create_pipeline.call_args.kwargs
    assert kwargs["task"] == task
    assert kwargs["lora_path"] == adapter
    assert (kwargs["video_shift"], kwargs["audio_shift"]) == shifts


@pytest.mark.parametrize("name, base, task, shifts", H3_TURBO_RUNNERS)
def test_h3_turbo_shift_env_overrides_the_default(
    name, base, task, shifts, adapter, monkeypatch
):
    monkeypatch.setenv("MINIMAX_H3_VIDEO_SHIFT", "12")
    monkeypatch.setenv("MINIMAX_H3_AUDIO_SHIFT", "2.5")
    kwargs = _h3_runner(name)._create_pipeline_kwargs()
    assert (kwargs["video_shift"], kwargs["audio_shift"]) == (12.0, 2.5)


@pytest.mark.parametrize("name, base, task, shifts", H3_TURBO_RUNNERS)
@pytest.mark.parametrize("lora_path", [None, "/nonexistent/turbo.safetensors"])
def test_h3_turbo_refuses_to_start_without_an_adapter_file(
    name, base, task, shifts, lora_path, monkeypatch
):
    if lora_path is None:
        monkeypatch.delenv("MINIMAX_H3_LORA_PATH", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_H3_LORA_PATH", lora_path)
    with pytest.raises(ValueError, match="MINIMAX_H3_LORA_PATH"):
        _h3_runner(name)._create_pipeline_kwargs()


@pytest.mark.parametrize(
    "class_name, l1_small_size",
    [
        ("TTMiniMaxH3Runner", None),
        ("TTMiniMaxH3Ref2VARunner", 16384),
        ("TTMiniMaxH3TurboRef2VARunner", 16384),
    ],
)
def test_h3_ref2va_keeps_its_l1_pool(class_name, l1_small_size, monkeypatch):
    captured = {}

    def fake_params(mesh_shape, *, l1_small_size=None):
        captured["l1_small_size"] = l1_small_size
        return {}

    monkeypatch.setattr(runners, "_minimax_h3_device_params", fake_params)
    _h3_runner(class_name).get_pipeline_device_params()
    assert captured["l1_small_size"] == l1_small_size


def test_ideogram_gets_the_caption_and_preset():
    runner = _runner("TTIdeogram4Runner")
    request = Ideogram4Request(
        structured_prompt={
            "compositional_deconstruction": {"background": "b", "elements": []}
        },
        rendering_speed="quality",
    )
    request = _resolved(request, 1024, 1024, prompt=request.model_prompt())
    runner.run([request])
    (call,) = runner.pipeline.calls
    assert call["prompts"] == [
        '{"compositional_deconstruction":{"background":"b","elements":[]}}'
    ]
    assert call["preset"] == "V4_QUALITY_48"


def test_ideogram_warmup_covers_every_size():
    # 4 sizes; one preset compiles all three.
    runner = _runner("TTIdeogram4Runner", warming_up=True)
    requests = runner.warmup_requests()
    assert len(requests) == 4
    for request in requests:
        runner.pipeline.calls.clear()
        runner.run([request])
        (call,) = runner.pipeline.calls
        assert (call["width"], call["height"]) == (request._width, request._height)


@pytest.mark.parametrize(
    ("mesh_shape", "fabric"), [((4, 8), "FABRIC_1D_RING"), ((2, 4), "FABRIC_1D")]
)
def test_ideogram_fabric_follows_the_pipeline_preset(mesh_shape, fabric):
    runner = _runner("TTIdeogram4Runner")
    runner.settings.device_mesh_shape = mesh_shape
    params = runner.get_pipeline_device_params()
    assert params["fabric_config"] is getattr(runners.ttnn.FabricConfig, fabric)


def test_ideogram_pipeline_gets_the_weights_path_as_is(monkeypatch):
    # The pipeline resolves a repo id itself, from tenstorrent/tt-metal#63894 on.
    pipeline = ModuleType("pipeline_ideogram4")
    pipeline.Ideogram4Pipeline = Mock()
    monkeypatch.setitem(
        sys.modules, "models.tt_dit.pipelines.ideogram4.pipeline_ideogram4", pipeline
    )
    runner = _runner("TTIdeogram4Runner")
    runner.settings.model_weights_path = "ideogram-ai/ideogram-4-fp8"
    runner.ttnn_device = Mock()
    runner.create_pipeline()
    kwargs = pipeline.Ideogram4Pipeline.create_pipeline.call_args.kwargs
    assert kwargs["weights_dir"] == "ideogram-ai/ideogram-4-fp8"

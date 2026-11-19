# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free checks of cpp_server/python/cpp_dit_adapters, the only Python of
the C++ video worker. tt_dit pipelines are replaced by fakes, so this runs
anywhere with numpy, PIL and pytest.

The rest of the runner layer is C++ and is tested by build/dit_runner_test.

Run: python -m pytest -q tests/video/python   (from cpp_server)
"""

import base64
import io
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

CPP_SERVER = Path(__file__).resolve().parents[3]
MEDIA_SERVER = CPP_SERVER.parent
sys.path.insert(0, str(CPP_SERVER / "python"))

import cpp_dit_adapters as adapters  # noqa: E402
from cpp_dit_adapters import _common  # noqa: E402


def png_b64(size=(8, 6), color=(255, 0, 0)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fake_module(monkeypatch, dotted, **attrs):
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = sys.modules[dotted]
    for key, value in attrs.items():
        monkeypatch.setattr(module, key, value, raising=False)
    return module


class FakePipeline:
    """Records the calls; create_pipeline returns an instance of it."""

    created = []

    def __init__(self, result=None, **kwargs):
        self.result = result
        self.kwargs = kwargs
        self.calls = []

    @classmethod
    def create_pipeline(cls, *args, **kwargs):
        pipe = cls(result=cls.result_for_create, **kwargs)
        cls.created.append((args, kwargs, pipe))
        return pipe

    result_for_create = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def pipeline_class(result):
    return type("P", (FakePipeline,), {"result_for_create": result, "created": []})


FRAMES = np.zeros((1, 2, 4, 6, 3), np.uint8)


# --------------------------------------------------------------------------
# the registry and its match with the C++ models
# --------------------------------------------------------------------------


def cpp_adapter_names():
    """Adapter names the C++ models ask for."""
    models = (CPP_SERVER / "include/runtime/dit/dit_models.hpp").read_text()
    names = set(re.findall(r'adapterName\(\) const override \{ return "(\w+)"; \}', models))
    names |= set(re.findall(r'adapter_\{"(\w+)"\}', models))
    registry = (CPP_SERVER / "src/runtime/dit/dit_registry.cpp").read_text()
    names |= set(re.findall(r'"(sd35|flux1|motif|qwen_image)"\);', registry))
    names |= set(
        re.findall(
            r'"(flux_kontext)"',
            (CPP_SERVER / "src/runtime/dit/models_image.cpp").read_text(),
        )
    )
    return names


def test_every_adapter_the_cpp_models_name_exists():
    wanted = cpp_adapter_names()
    assert {"wan_t2v", "minimax", "ltx", "mochi", "flux_kontext", "sd35"} <= wanted
    assert wanted <= set(adapters.adapter_names()), wanted - set(adapters.adapter_names())


def test_every_adapter_is_used_by_a_cpp_model():
    assert set(adapters.adapter_names()) <= cpp_adapter_names()


def test_unknown_adapter_and_double_registration():
    with pytest.raises(ValueError, match="no adapter"):
        adapters.create("nope", None, {})
    with pytest.raises(ValueError, match="already registered"):
        adapters.register("wan_t2v")(lambda mesh, o: None)


def test_cpp_runner_list_and_meshes_match_the_python_server():
    """build/dit_runner_test dumps the C++ mesh table; compare it with the
    Python server's config (skipped when either side is not available)."""
    dump = CPP_SERVER / "build" / "dit_meshes.json"
    if not dump.exists():
        pytest.skip("run build/dit_runner_test first")
    sys.path.insert(0, str(MEDIA_SERVER))
    try:
        from config.constants import ModelConfigs, ModelRunners
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Python server config not importable here: {e}")
    finally:
        sys.path.remove(str(MEDIA_SERVER))
    source = (MEDIA_SERVER / "tt_model_runners" / "dit_runners.py").read_text()
    start = source.index("dit_runner_log_map")
    log_map = source[start : source.index("}", start)]
    python_runners = {
        ModelRunners[e].value
        for e in re.findall(r"ModelRunners\.(\w+)\.value", log_map)
        if e != "SP_RUNNER"  # drives an external process, not a tt_dit pipeline
    }
    cpp = json.loads(dump.read_text())
    assert set(cpp) == python_runners
    expected = {}
    for (runner, device), cfg in ModelConfigs.items():
        if runner.value in python_runners:
            expected.setdefault(runner.value, {})[device.value] = list(cfg["device_mesh_shape"])
    assert cpp == expected


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


def test_frames_result_normalizes_what_pipelines_return():
    out = _common.frames_result(FRAMES)
    assert out["kind"] == "frames" and out["frames"].dtype == np.uint8
    assert _common.frames_result([FRAMES[0]])["frames"].shape == (2, 4, 6, 3)
    ints = _common.frames_result(np.zeros((2, 2, 2, 3), np.int16))["frames"]
    assert ints.dtype == np.float32 and ints.flags["C_CONTIGUOUS"]
    from PIL import Image

    pil = [[Image.new("RGB", (6, 4)) for _ in range(3)]]
    assert _common.frames_result(pil)["frames"].shape == (1, 3, 4, 6, 3)
    with pytest.raises(ValueError):
        _common.frames_result([])


def test_images_result_is_png_bytes():
    from PIL import Image

    out = _common.images_result([Image.new("RGB", (4, 4)), np.zeros((1, 4, 6, 3), np.float32)])
    assert out["kind"] == "images"
    assert Image.open(io.BytesIO(out["images"][0])).size == (4, 4)
    assert Image.open(io.BytesIO(out["images"][1])).size == (6, 4)
    assert _common.path_result(["/a.mp4"]) == {"kind": "path", "path": "/a.mp4"}


def test_base64_decoding():
    assert _common.decode_base64("aGVsbG8") == b"hello"  # padding restored
    assert _common.decode_base64("data:text/plain;base64,aGVsbG8=") == b"hello"
    with pytest.raises(ValueError, match="URLs"):
        _common.decode_base64("https://example.com/a.png")
    assert _common.decode_image(png_b64(), size=(4, 2)).size == (4, 2)


# --------------------------------------------------------------------------
# Wan2.2
# --------------------------------------------------------------------------

WAN_OPTIONS = {"checkpoint": "ckpt", "height": 480, "width": 832, "num_frames": 81}


def test_wan_t2v_creates_and_calls_the_pipeline_with_the_cpp_arguments(monkeypatch):
    P = pipeline_class(FRAMES)
    fake_module(monkeypatch, "models.tt_dit.pipelines.wan.pipeline_wan", WanPipeline=P)
    adapter = adapters.create("wan_t2v", "MESH", WAN_OPTIONS)
    assert P.created[0][1] == {
        "mesh_device": "MESH", "checkpoint_name": "ckpt",
        "height": 480, "width": 832, "num_frames": 81,
    }  # fmt: skip
    args = {"prompts": ["a"], "num_inference_steps": 4, "seed": 1, "traced": False}
    result = adapter.call(args)
    assert P.created[0][2].calls[0][1] == args
    assert result["kind"] == "frames" and result["frames"].shape == (1, 2, 4, 6, 3)


@dataclass
class ImagePrompt:
    image: object
    frame_pos: int


def test_wan_i2v_turns_image_prompts_into_pipeline_objects(monkeypatch):
    P = pipeline_class(FRAMES)
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.wan.pipeline_wan_i2v",
        WanPipelineI2V=P, ImagePrompt=ImagePrompt,
    )  # fmt: skip
    adapter = adapters.create("wan_i2v", "MESH", WAN_OPTIONS)
    assert "checkpoint_name" not in P.created[0][1]  # the I2V pipeline has its own
    adapter.call(
        {"prompts": ["a"], "image_prompts": [{"image": png_b64(), "frame_pos": -1}]}
    )
    kwargs = P.created[0][2].calls[0][1]
    assert "image_prompts" not in kwargs
    (prompt,) = kwargs["image_prompt"]
    assert prompt.frame_pos == -1 and prompt.image.size == (8, 6)


def test_wan_lora_builds_the_config_and_downloads_missing_adapters(monkeypatch):
    seen = {}

    class Lora(FakePipeline):
        def __init__(self, device, config, lora_high, lora_low):
            super().__init__(FRAMES)
            seen.update(device=device, config=config, high=lora_high, low=lora_low)

        @staticmethod
        def default_config(**kwargs):
            seen["config_kwargs"] = kwargs
            return "CONFIG"

    fake_module(
        monkeypatch, "models.tt_dit.experimental.pipelines.pipeline_wan_lora",
        WanPipelineI2VLora=Lora,
    )  # fmt: skip
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.wan.pipeline_wan_i2v", ImagePrompt=ImagePrompt
    )
    downloads = []
    fake_module(
        monkeypatch, "huggingface_hub",
        hf_hub_download=lambda repo_id, filename: downloads.append(filename) or f"/dl/{filename}",
    )  # fmt: skip
    options = {
        **WAN_OPTIONS, "lora_high": "/mine/high", "cfg_enabled": False,
        "boundary_ratio": 0.875, "download_repo": "r", "download_high": "h", "download_low": "l",
    }  # fmt: skip
    adapters.create("wan_lora", "MESH", options)
    assert seen["high"] == "/mine/high" and seen["low"] == "/dl/l" and downloads == ["l"]
    assert seen["config_kwargs"]["cfg_enabled"] is False
    assert seen["config_kwargs"]["config_overrides"] == {"boundary_ratio": 0.875}
    assert seen["config"] == "CONFIG" and seen["device"] == "MESH"


def test_wan_prodia_i2v_resizes_images_to_the_output_size(monkeypatch):
    created = {}

    def create_i2v_pipeline(mesh, weights_dir, **kwargs):
        created.update(mesh=mesh, weights_dir=weights_dir, **kwargs)
        return pipe

    pipe = FakePipeline(FRAMES)
    fake_module(monkeypatch, "pipelines.pipeline_i2v", create_i2v_pipeline=create_i2v_pipeline)
    adapter = adapters.create("wan_prodia_i2v", "MESH", {**WAN_OPTIONS, "checkpoint": "w"})
    assert created["weights_dir"] == "w"
    adapter.call(
        {
            "prompt": "x", "height": 480, "width": 832, "num_frames": 81, "seed": 0,
            "traced": True, "image_prompts": [{"image": png_b64(), "frame_pos": 5}],
        }
    )  # fmt: skip
    ((image, pos),) = pipe.calls[0][1]["image"]
    assert pos == 5 and image.size == (832, 480)


# --------------------------------------------------------------------------
# Mochi, LTX, MiniMax, images
# --------------------------------------------------------------------------


def test_mochi_stacks_pil_frames(monkeypatch):
    from PIL import Image

    P = pipeline_class([[Image.new("RGB", (6, 4)) for _ in range(3)]])
    fake_module(monkeypatch, "models.tt_dit.pipelines.mochi.pipeline_mochi", MochiPipeline=P)
    result = adapters.create("mochi", "MESH", {"checkpoint": "m"}).call({"prompts": ["x"]})
    assert result["frames"].shape == (1, 3, 4, 6, 3)


def test_ltx_returns_the_path_the_pipeline_wrote(monkeypatch):
    calls = []

    class Ltx:
        @staticmethod
        def create_pipeline(**kwargs):
            calls.append(kwargs)
            return Ltx()

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return kwargs["output_path"]

    fake_module(monkeypatch, "models.tt_dit.pipelines.ltx.pipeline_ltx_distilled", LTXDistilledPipeline=Ltx)
    fake_module(monkeypatch, "models.tt_dit.utils.ltx", default_ltx_gemma=lambda: "/gemma")
    fake_module(monkeypatch, "ttnn", Topology=types.SimpleNamespace(Ring="Ring"))
    options = {"checkpoint": "c", "traced": True, "num_frames": 153, "height": 1088, "width": 1920, "fps": 25.0}
    adapter = adapters.create("ltx", "MESH", options)
    assert calls[0]["traced"] is True and calls[0]["gemma_path"] == "/gemma"
    assert calls[0]["topology"] == "Ring"
    result = adapter.call({"prompt": "x", "output_path": "/o/a.mp4", "seed": 1})
    assert result == {"kind": "path", "path": "/o/a.mp4"}
    assert calls[1] == ("x", {"output_path": "/o/a.mp4", "seed": 1})


def minimax_output(video_format="rgb_float"):
    import torch

    video = torch.zeros(1, 3, 2, 4, 6) if video_format == "rgb_float" else np.zeros((2, 6, 4), np.uint8)
    return types.SimpleNamespace(
        video=video, audio=torch.zeros(1, 2, 100), sampling_rate=16000,
        fps=24.0, video_format=video_format,
    )  # fmt: skip


def test_minimax_maps_the_cpp_call_to_the_pipeline(monkeypatch):
    P = pipeline_class(minimax_output())
    fake_module(monkeypatch, "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3", MiniMaxH3Pipeline=P)
    refs = {}
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.minimax_h3.packing_ref2va",
        MiniMaxH3Reference=lambda **kw: refs.setdefault("made", []).append(kw) or kw,
        decode_reference_audio=lambda p: ("wave", 16000),
        reference_from_video_file=lambda p: "video-ref",
    )  # fmt: skip
    options = {
        "pipeline": "h3", "task": "ref2va", "weights_dir": None, "dit_fsdp": True,
        "trace_denoise": None, "bucket_denoise": None,
    }  # fmt: skip
    adapter = adapters.create("minimax", "MESH", options)
    assert P.created[0][1]["task"] == "ref2va" and "lora_path" not in P.created[0][1]
    result = adapter.call(
        {
            "prompt": "x", "num_frames": 124, "aspect_ratio": [16, 9], "height": None,
            "width": None, "num_inference_steps": 50, "seed": 3,
            "first_image": png_b64(), "references": {"images": [png_b64()]},
        }
    )  # fmt: skip
    args, kwargs = P.created[0][2].calls[0]
    assert args == ("x",) and kwargs["aspect_ratio"] == (16, 9)
    assert kwargs["image"].size == (8, 6) and kwargs["last_image"] is None
    assert kwargs["references"][0]["image"].size == (8, 6)
    assert result["kind"] == "av" and result["frames"].shape == (2, 4, 6, 3)
    assert result["frames"].dtype == np.float32 and result["audio"].shape == (2, 100)
    assert result["sampling_rate"] == 16000 and result["pixel_format"] == "rgb24"


def test_minimax_variants_and_planar_video(monkeypatch):
    P = pipeline_class(minimax_output("yuv420"))
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3_turbo",
        MiniMaxH3TurboPipeline=P,
    )  # fmt: skip
    options = {
        "pipeline": "turbo", "task": "fl2va", "weights_dir": "/w", "dit_fsdp": False,
        "trace_denoise": False, "bucket_denoise": None, "lora_path": "/l",
        "video_shift": 6.0, "audio_shift": 3.0,
    }  # fmt: skip
    adapter = adapters.create("minimax", "MESH", options)
    assert P.created[0][1]["video_shift"] == 6.0 and P.created[0][1]["lora_path"] == "/l"
    result = adapter.call({"prompt": "x", "aspect_ratio": [16, 9]})
    assert result["pixel_format"] == "yuv420p" and result["frames"].dtype == np.uint8


def test_minimax_preset(monkeypatch):
    fake_module(monkeypatch, "ttnn", Topology=types.SimpleNamespace(Ring="Ring"))
    seen = []
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3",
        resolve_mesh_preset=lambda shape: seen.append(shape) or {"topology": "Ring", "trace_denoise": True},
    )  # fmt: skip
    assert adapters.minimax_preset(4, 32) == {"ring": True, "trace_denoise": True}
    assert seen == [(4, 32)]


def test_image_adapters_return_png_bytes(monkeypatch):
    from PIL import Image

    pipe = types.SimpleNamespace(calls=[])
    pipe.run_single_prompt = lambda **kw: pipe.calls.append(kw) or [Image.new("RGB", (4, 4))]
    P = type("F", (), {"create_pipeline": staticmethod(lambda **kw: pipe)})
    fake_module(monkeypatch, "models.tt_dit.pipelines.flux1.pipeline_flux1", Flux1Pipeline=P)
    result = adapters.create("flux1", "MESH", {"checkpoint": "f"}).call(
        {"prompt": "x", "negative_prompt": None, "num_inference_steps": 20, "seed": 9}
    )
    assert pipe.calls[0]["seed"] == 9
    assert Image.open(io.BytesIO(result["images"][0])).size == (4, 4)


def test_kontext_resizes_the_edit_image_and_falls_back_to_the_pipeline_size(monkeypatch):
    from PIL import Image

    class Kontext(FakePipeline):
        _width, _height = 640, 320

    Kontext.result_for_create = [Image.new("RGB", (4, 4))]
    Kontext.created = []
    fake_module(
        monkeypatch, "models.tt_dit.pipelines.flux1.pipeline_flux1_kontext",
        Flux1KontextPipeline=Kontext,
    )  # fmt: skip
    adapter = adapters.create("flux_kontext", "MESH", {"checkpoint": "k", "lora_path": "/l", "lora_scale": 0.5})
    assert Kontext.created[0][1]["lora_scale"] == 0.5
    adapter.call({"prompts": ["x"], "width": None, "height": None, "image": png_b64()})
    kwargs = Kontext.created[0][2].calls[0][1]
    assert (kwargs["width"], kwargs["height"]) == (640, 320)
    assert kwargs["image"].size == (640, 320)
    adapter.call({"prompts": ["x"], "width": 512, "height": 256, "image": None})
    assert Kontext.created[0][2].calls[1][1]["image"] is None

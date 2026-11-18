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
    return set(
        re.findall(r'adapterName\(\) const override \{ return "(\w+)"; \}', models)
    )


def test_every_adapter_the_cpp_models_name_exists():
    wanted = cpp_adapter_names()
    assert wanted == {"wan_t2v", "wan_i2v"}
    assert wanted <= set(adapters.adapter_names()), wanted - set(
        adapters.adapter_names()
    )


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
    # The C++ server runs the Wan2.2 runners only (the rest of the Python
    # runners are not ported here); each of them must be a Python runner.
    assert set(cpp) == {"tt-wan2.2", "tt-wan2.2-i2v"}
    assert set(cpp) <= python_runners
    expected = {}
    for (runner, device), cfg in ModelConfigs.items():
        if runner.value in cpp:
            expected.setdefault(runner.value, {})[device.value] = list(
                cfg["device_mesh_shape"]
            )
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

    out = _common.images_result(
        [Image.new("RGB", (4, 4)), np.zeros((1, 4, 6, 3), np.float32)]
    )
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


# --------------------------------------------------------------------------
# Mochi, LTX, MiniMax, images
# --------------------------------------------------------------------------

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The TT lmms-eval plugin: WAV parts, response handling, the subprocess env."""

import base64
import io
import re
import sys
import wave
from types import SimpleNamespace

import numpy as np

from llm_module.lmms_plugins import PLUGIN_PACKAGE, PLUGIN_ROOT, lmms_plugin_env

sys.path.insert(0, str(PLUGIN_ROOT))  # what lmms_plugin_env does for the subprocess

from tt_lmms_plugins.audio_parts import (  # noqa: E402
    audio_content_part,
    is_audio_visual,
    wav_bytes_from_array,
)
from tt_lmms_plugins.models import AVAILABLE_MODELS  # noqa: E402
from tt_lmms_plugins.responses import (  # noqa: E402
    parse_enable_thinking,
    response_text,
    thread_cap,
)


def _decode(part):
    raw = base64.b64decode(part["input_audio"]["data"])
    with wave.open(io.BytesIO(raw), "rb") as w:
        return (
            w.getnchannels(),
            w.getframerate(),
            w.getsampwidth(),
            w.readframes(w.getnframes()),
        )


def test_hf_audio_dict_becomes_a_16_bit_mono_wav_part():
    t = np.arange(1600) / 16000.0
    visual = {
        "array": (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        "sampling_rate": 16000,
    }
    part = audio_content_part(visual)
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    channels, rate, width, frames = _decode(part)
    assert (channels, rate, width) == (1, 16000, 2)
    pcm = np.frombuffer(frames, dtype="<i2")
    assert len(pcm) == 1600
    assert abs(int(pcm.max()) - int(0.5 * 32767)) <= 1


def test_stereo_arrays_are_averaged_and_clipped():
    stereo = np.stack([np.full(100, 2.0), np.full(100, -2.0)]).astype(np.float32)
    pcm = np.frombuffer(wav_bytes_from_array(stereo, 8000)[44:], dtype="<i2")
    assert len(pcm) == 100 and int(pcm.max()) == 0
    pcm = np.frombuffer(wav_bytes_from_array(np.full(10, 3.0), 8000)[44:], dtype="<i2")
    assert int(pcm.min()) == 32767


def test_wav_path_is_sent_as_is(tmp_path):
    raw = wav_bytes_from_array(np.zeros(16), 16000)
    p = tmp_path / "clip.wav"
    p.write_bytes(raw)
    assert base64.b64decode(audio_content_part(str(p))["input_audio"]["data"]) == raw


def test_only_audio_visuals_are_recognised():
    assert is_audio_visual({"array": np.zeros(4), "sampling_rate": 16000})
    assert is_audio_visual("/x/clip.FLAC")
    assert not is_audio_visual("/x/photo.png")
    assert not is_audio_visual({"path": "clip.wav"})
    assert not is_audio_visual(None)


def test_plugin_registry_names_the_class_in_its_module():
    assert AVAILABLE_MODELS == {"openai_compatible_tt": "OpenAICompatibleTT"}
    src = (PLUGIN_ROOT / "tt_lmms_plugins/models/openai_compatible_tt.py").read_text()
    assert re.search(
        r'@register_model\("openai_compatible_tt"\)\nclass OpenAICompatibleTT\(', src
    )
    # The upstream 4096 clamp and the fixed 32-thread cap are gone.
    assert 'gen_kwargs["max_new_tokens"] = 4096' not in src
    assert "min(len(tasks_to_run), 32)" not in src
    # Replies are put back in the requests' original order before returning
    # (upstream returns them in the collator's sorted order: shuffled scores).
    body = src[src.index("def generate_until") :]
    assert "res = re_ords.get_original(res)" in body
    assert body.index("res = re_ords.get_original(res)") < body.index("return res")


def test_response_text_is_empty_for_an_unclosed_thought():
    ok = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="B"))]
    )
    unclosed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning_content="...")
            )
        ]
    )
    assert response_text(ok) == "B"
    assert response_text(unclosed) == ""
    assert response_text(SimpleNamespace(choices=[])) == ""


def test_thread_cap_follows_num_concurrent_and_the_batch():
    assert thread_cap(10, None) == 10
    assert thread_cap(100, None) == 32
    assert thread_cap(100, 128) == 100
    assert thread_cap(200, "128") == 128
    assert thread_cap(0, 4) == 1


def test_plugin_env_registers_the_package_once_and_adds_the_plugin_root():
    env = lmms_plugin_env(
        {"PYTHONPATH": "/else", "LMMS_EVAL_PLUGINS": "other"}, "/plug"
    )
    assert env["LMMS_EVAL_PLUGINS"] == f"other,{PLUGIN_PACKAGE}"
    assert env["PYTHONPATH"].split(":") == ["/plug", "/else"]
    again = lmms_plugin_env(env, "/plug")
    assert again["LMMS_EVAL_PLUGINS"] == env["LMMS_EVAL_PLUGINS"]
    assert again["PYTHONPATH"] == env["PYTHONPATH"]
    bare = lmms_plugin_env({})
    assert bare == {"LMMS_EVAL_PLUGINS": PLUGIN_PACKAGE, "PYTHONPATH": str(PLUGIN_ROOT)}
    assert (PLUGIN_ROOT / PLUGIN_PACKAGE / "models" / "__init__.py").is_file()
    # lmms-eval's CLI resolves <plugin>.tasks for every plugin and crashes without it.
    assert (PLUGIN_ROOT / PLUGIN_PACKAGE / "tasks" / "__init__.py").is_file()


def test_enable_thinking_model_arg_parses_to_a_tri_state():
    assert parse_enable_thinking(None) is None
    assert parse_enable_thinking("") is None
    assert parse_enable_thinking("false") is False
    assert parse_enable_thinking("0") is False
    assert parse_enable_thinking("True") is True
    assert parse_enable_thinking(True) is True
    import pytest

    with pytest.raises(ValueError):
        parse_enable_thinking("maybe")

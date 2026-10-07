# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free unit tests for the Qwen3-TTS runner (tt_model_runners/qwen3_tts_runner.py).

torch, ttnn and the tt-metal model package are not used: conftest stubs torch/ttnn, and
the model API (``_tts_api``) is patched where a test needs it.
"""

import asyncio
import base64
import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from config.constants import SupportedModels
from domain.text_to_speech_request import TextToSpeechRequest

# conftest replaces every runner module with a stub class; import the real one and put
# the stub back so later test files see what they expect. soundfile lives in the audio
# venv, so stub it when this venv lacks it (only _synthesize writes WAVs).
_RUNNER = "tt_model_runners.qwen3_tts_runner"
_stub = sys.modules.pop(_RUNNER, None)
_sf_missing = importlib.util.find_spec("soundfile") is None
if _sf_missing:
    sys.modules["soundfile"] = MagicMock()
try:
    # conftest's utils.logger stub lacks log_exception_chain (used only on fabric errors)
    with patch.object(
        sys.modules["utils.logger"], "log_exception_chain", MagicMock(), create=True
    ):
        qr = importlib.import_module(_RUNNER)
finally:
    if _stub is not None:
        sys.modules[_RUNNER] = _stub
    if _sf_missing:
        sys.modules.pop("soundfile", None)

HF_17 = SupportedModels.QWEN3_TTS_1_7B.value
HF_06 = SupportedModels.QWEN3_TTS_0_6B.value


def _settings(mesh=(1, 1), weights=None, galaxy=False):
    return SimpleNamespace(
        device_mesh_shape=mesh,
        model_weights_path=weights,
        is_galaxy=galaxy,
        use_dynamic_batcher=False,
        max_batch_size=1,
    )


def _bare_runner(mesh=(1, 1), weights=None):
    """A runner without running __init__ (no env setup, no device)."""
    r = qr.TTQwen3TTSRunner.__new__(qr.TTQwen3TTSRunner)
    r.device_id = "0"
    r.logger = MagicMock()
    r.settings = _settings(mesh, weights)
    r.is_tensor_parallel = mesh[0] * mesh[1] > 1
    r.voice_prompts = None
    r._ref_states = {}
    r.model = None
    r.ctx = None
    return r


def _init_runner(mesh=(1, 1), galaxy=False, weights=None):
    s = _settings(mesh, weights, galaxy)
    with patch(
        "tt_model_runners.base_device_runner.get_settings", return_value=s
    ), patch.object(qr, "settings", s):
        return qr.TTQwen3TTSRunner("-1")


class TestLooksJapanese:
    @pytest.mark.parametrize("text", ["こんにちは", "テスト", "日本語"])
    def test_japanese(self, text):
        assert qr._looks_japanese(text)

    @pytest.mark.parametrize("text", ["Hello", "Grüß Gott", "", "123"])
    def test_not_japanese(self, text):
        assert not qr._looks_japanese(text)


class TestResolveHfId:
    @pytest.mark.parametrize(
        "weights,expected",
        [
            (None, HF_17),
            (HF_17, HF_17),
            (HF_06, HF_06),
            ("/mnt/weights/Qwen3-TTS-12Hz-0.6B-Base", HF_06),
            ("/mnt/weights/Qwen3-TTS-12Hz-1.7B-Base", HF_17),
            ("/mnt/weights/something-else", HF_17),
        ],
    )
    def test_resolve(self, weights, expected):
        assert _bare_runner(weights=weights)._resolve_hf_id() == expected


class TestInitEnvironment:
    def test_n150_pops_throttle_and_disables_fabric(self):
        with patch.dict(
            os.environ, {"TT_MM_THROTTLE_PERF": "5", "TT_METAL_CACHE": "/x"}
        ):
            os.environ.pop("TT_METAL_FABRIC_DISABLE", None)
            r = _init_runner()
            assert "TT_MM_THROTTLE_PERF" not in os.environ
            assert "TT_METAL_CACHE" not in os.environ
            assert os.environ.get("TT_METAL_FABRIC_DISABLE") == "1"
            assert r.is_tensor_parallel is False
            assert r.hf_id == HF_17

    def test_n300_tensor_parallel_keeps_fabric(self):
        with patch.dict(os.environ, {"TT_METAL_FABRIC_DISABLE": "1"}):
            r = _init_runner(mesh=(1, 2))
            assert r.is_tensor_parallel is True
            assert "TT_METAL_FABRIC_DISABLE" not in os.environ

    def test_galaxy_single_chip_leaves_fabric_alone(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("TT_METAL_FABRIC_DISABLE", None)
            _init_runner(galaxy=True)
            assert "TT_METAL_FABRIC_DISABLE" not in os.environ

    def test_device_decode_defaults(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("TT_QWEN3_DEVICE_DECODE", None)
            os.environ.pop("TT_QWEN3_DEVICE_DECODE_MODE", None)
            r = _init_runner()
            assert r._use_device_decode is True
            assert r._decode_continue is True

    def test_device_decode_opt_out(self):
        with patch.dict(
            os.environ,
            {"TT_QWEN3_DEVICE_DECODE": "0", "TT_QWEN3_DEVICE_DECODE_MODE": "full"},
        ):
            r = _init_runner()
            assert r._use_device_decode is False
            assert r._decode_continue is False


class TestDeviceParams:
    def test_n150(self):
        p = _bare_runner().get_pipeline_device_params()
        assert p == {
            "l1_small_size": qr.Qwen3TTSConstants.L1_SMALL_SIZE,
            "trace_region_size": qr.Qwen3TTSConstants.TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }

    def test_n300_adds_fabric(self):
        p = _bare_runner(mesh=(1, 2)).get_pipeline_device_params()
        assert "fabric_config" in p
        assert p["num_command_queues"] == 2


class TestLanguage:
    def _req(self, text="Hello", speaker_id=None):
        return TextToSpeechRequest(text=text, speaker_id=speaker_id)

    def test_default_english(self):
        with patch.dict(os.environ, {"QWEN3_TTS_LANGUAGE": ""}):
            assert _bare_runner()._language_for(self._req()) == "english"

    def test_japanese_text(self):
        with patch.dict(os.environ, {"QWEN3_TTS_LANGUAGE": ""}):
            assert _bare_runner()._language_for(self._req("こんにちは")) == "japanese"

    def test_env_overrides_detection(self):
        with patch.dict(os.environ, {"QWEN3_TTS_LANGUAGE": "German"}):
            assert _bare_runner()._language_for(self._req("こんにちは")) == "german"

    def test_speaker_language_wins(self):
        with patch.dict(os.environ, {"QWEN3_TTS_LANGUAGE": "german"}):
            req = self._req("Hello", speaker_id="French")
            assert _bare_runner()._language_for(req) == "french"

    def test_unknown_env_ignored(self):
        with patch.dict(os.environ, {"QWEN3_TTS_LANGUAGE": "klingon"}):
            assert _bare_runner()._language_for(self._req()) == "english"


class TestResolveVoice:
    def _runner_with_prompts(self, prompts):
        r = _bare_runner()
        r.voice_prompts = MagicMock()
        r.voice_prompts.get.side_effect = prompts.get
        r.voice_prompts.list_available.return_value = sorted(prompts)
        return r

    def test_preset_voice(self):
        jim = SimpleNamespace(ref_codes="codes", ref_text="ref", audio_data="audio")
        r = self._runner_with_prompts({"jim": jim})
        with patch.object(qr, "_tts_api", return_value=MagicMock()):
            out = r._resolve_voice(TextToSpeechRequest(text="hi", speaker_id="jim"))
        assert out == ("codes", "ref", "audio", "jim")

    def test_default_and_language_speaker_map_to_default_voice(self):
        jim = SimpleNamespace(ref_codes="c", ref_text="t", audio_data="a")
        r = self._runner_with_prompts({qr.DEFAULT_VOICE_ID: jim})
        with patch.object(qr, "_tts_api", return_value=MagicMock()):
            for sid in (None, "english", "japanese"):
                out = r._resolve_voice(TextToSpeechRequest(text="hi", speaker_id=sid))
                assert out[3] == qr.DEFAULT_VOICE_ID

    def test_unknown_voice_lists_available(self):
        r = self._runner_with_prompts({"jim": object()})
        with patch.object(qr, "_tts_api", return_value=MagicMock()):
            with pytest.raises(ValueError, match="Unknown voice_id='nobody'.*jim"):
                r._resolve_voice(TextToSpeechRequest(text="hi", speaker_id="nobody"))

    def test_adhoc_clone_decodes_payload_and_removes_temp_file(self):
        api = MagicMock()
        seen = {}

        def encode(path, main_weights=None):
            with open(path, "rb") as f:
                seen["bytes"] = f.read()
            seen["path"] = path
            return "ref_codes", "audio_data"

        api.encode_reference_audio.side_effect = encode
        req = TextToSpeechRequest(
            text="hi",
            voice_clone_audio=base64.b64encode(b"RIFFfake").decode(),
            voice_clone_text="clone transcript",
        )
        with patch.object(qr, "_tts_api", return_value=api):
            out = _bare_runner()._resolve_voice(req)
        assert out == ("ref_codes", "clone transcript", "audio_data", "<adhoc>")
        assert seen["bytes"] == b"RIFFfake"
        assert not os.path.exists(seen["path"])

    def test_clone_audio_without_text_falls_back_to_speaker(self):
        jim = SimpleNamespace(ref_codes="c", ref_text="t", audio_data="a")
        r = self._runner_with_prompts({"jim": jim})
        req = TextToSpeechRequest(text="hi", voice_clone_audio="AAAA")
        with patch.object(qr, "_tts_api", return_value=MagicMock()):
            assert r._resolve_voice(req)[3] == "jim"


class TestRefStateCache:
    def test_adhoc_is_not_cached(self):
        api = MagicMock()
        r = _bare_runner()
        assert r._ref_state_for(api, "<adhoc>", "codes") is None
        api.prepare_icl_decoder_state.assert_not_called()

    def test_preset_state_computed_once(self):
        api = MagicMock()
        api.prepare_icl_decoder_state.return_value = "state"
        r = _bare_runner()
        r.decoder_weights = "w"
        with patch.object(qr.torch, "equal", side_effect=lambda a, b: a == b):
            assert r._ref_state_for(api, "jim", "codes") == "state"
            assert r._ref_state_for(api, "jim", "codes") == "state"
            api.prepare_icl_decoder_state.assert_called_once_with("codes", "w")
            # different reference codes for the same voice id: recompute
            r._ref_state_for(api, "jim", "other")
            assert api.prepare_icl_decoder_state.call_count == 2


class TestRunGuards:
    def test_empty_list(self):
        with pytest.raises(ValueError, match="Empty request list"):
            asyncio.run(_bare_runner()._run_async([]))

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_text(self, text):
        with pytest.raises(ValueError, match="Text cannot be empty"):
            asyncio.run(_bare_runner()._run_async([TextToSpeechRequest(text=text)]))

    def test_batch_processes_first_only(self):
        r = _bare_runner()
        r._synthesize = MagicMock(return_value="resp")
        reqs = [TextToSpeechRequest(text="one"), TextToSpeechRequest(text="two")]
        assert asyncio.run(r._run_async(reqs)) == "resp"
        r._synthesize.assert_called_once_with(reqs[0])
        r.logger.warning.assert_called_once()

    def test_synthesize_before_warmup(self):
        with patch.object(qr, "_tts_api", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="warmup"):
                _bare_runner()._synthesize(TextToSpeechRequest(text="hi"))

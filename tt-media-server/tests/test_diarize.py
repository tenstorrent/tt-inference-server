# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import types
from pathlib import Path

import pytest

from utils.diarize import (
    AudioSegment,
    DiarizationProcessor,
    DiarizeCLI,
    DiarizeResponse,
    DiarizeService,
    OperationMode,
    ProcessorFactory,
    TorchLoadPatcher,
    VADProcessor,
)


# ---------------------------------------------------------------------------
# OperationMode
# ---------------------------------------------------------------------------


def test_operation_mode_values_lists_all_modes():
    assert set(OperationMode.values()) == {"diarize", "vad"}


def test_operation_mode_round_trip_through_string():
    assert OperationMode("diarize") is OperationMode.DIARIZE
    assert OperationMode("vad") is OperationMode.VAD


def test_operation_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        OperationMode("transcribe")


# ---------------------------------------------------------------------------
# AudioSegment / DiarizeResponse
# ---------------------------------------------------------------------------


def test_audio_segment_to_dict_omits_speaker_when_none():
    seg = AudioSegment(start=0.0, end=1.5)
    assert seg.to_dict() == {"start": 0.0, "end": 1.5}


def test_audio_segment_to_dict_includes_speaker_when_set():
    seg = AudioSegment(start=0.0, end=1.5, speaker="SPEAKER_00")
    assert seg.to_dict() == {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}


def test_diarize_response_default_is_success():
    resp = DiarizeResponse()
    assert resp.status == "success"
    assert resp.error is None
    assert resp.segments == []


def test_diarize_response_error_response_helper():
    resp = DiarizeResponse.error_response("kaboom")
    assert resp.status == "error"
    assert resp.error == "kaboom"
    assert resp.segments == []


def test_diarize_response_save_writes_json(tmp_path: Path):
    seg = AudioSegment(start=0.0, end=2.0, speaker="SPEAKER_01")
    resp = DiarizeResponse(segments=[seg])
    out = tmp_path / "result.json"
    resp.save(out)

    payload = json.loads(out.read_text())
    assert payload["status"] == "success"
    assert payload["error"] is None
    assert payload["segments"] == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]


# ---------------------------------------------------------------------------
# TorchLoadPatcher (formerly _installLegacyTorchLoadDefault)
# ---------------------------------------------------------------------------


def _fake_torch_module():
    """Build a torch-shaped namespace whose `.load` records every call."""
    call_log: list[tuple[tuple, dict]] = []

    def fake_load(*args, **kwargs):
        call_log.append((args, dict(kwargs)))
        return "loaded"

    return types.SimpleNamespace(load=fake_load), call_log


def test_torch_load_patcher_replaces_load():
    fake_torch, _ = _fake_torch_module()
    original_load = fake_torch.load
    TorchLoadPatcher(fake_torch).install()
    assert fake_torch.load is not original_load


def test_torch_load_patcher_forces_false_when_kwarg_missing():
    fake_torch, call_log = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    fake_torch.load("/tmp/model.pt")
    assert call_log[0][1] == {"weights_only": False}


def test_torch_load_patcher_forces_false_when_kwarg_is_none():
    # The real scenario hit by lightning_fabric._load: it forwards
    # weights_only=None explicitly, which a plain dict.setdefault() would miss.
    fake_torch, call_log = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    fake_torch.load("/tmp/model.pt", weights_only=None)
    assert call_log[0][1] == {"weights_only": False}


def test_torch_load_patcher_respects_explicit_true():
    fake_torch, call_log = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    fake_torch.load("/tmp/model.pt", weights_only=True)
    assert call_log[0][1] == {"weights_only": True}


def test_torch_load_patcher_respects_explicit_false():
    fake_torch, call_log = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    fake_torch.load("/tmp/model.pt", weights_only=False)
    assert call_log[0][1] == {"weights_only": False}


def test_torch_load_patcher_forwards_positional_and_extra_kwargs():
    fake_torch, call_log = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    fake_torch.load("/tmp/model.pt", "cpu", map_location="cpu")
    assert call_log[0][0] == ("/tmp/model.pt", "cpu")
    assert call_log[0][1] == {"map_location": "cpu", "weights_only": False}


def test_torch_load_patcher_preserves_return_value():
    fake_torch, _ = _fake_torch_module()
    TorchLoadPatcher(fake_torch).install()
    assert fake_torch.load("/tmp/model.pt") == "loaded"


# ---------------------------------------------------------------------------
# ProcessorFactory
# ---------------------------------------------------------------------------


def test_processor_factory_builds_diarization_processor():
    factory = ProcessorFactory(model_name="pyannote/test", hf_token="t")
    processor = factory.create(OperationMode.DIARIZE)
    assert isinstance(processor, DiarizationProcessor)


def test_processor_factory_builds_vad_processor():
    factory = ProcessorFactory(model_name="pyannote/test", hf_token=None)
    processor = factory.create(OperationMode.VAD)
    assert isinstance(processor, VADProcessor)


# ---------------------------------------------------------------------------
# DiarizeCLI argparse wiring
# ---------------------------------------------------------------------------


def test_diarize_cli_builds_service_with_parsed_args(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    audio = tmp_path / "audio.npy"
    output = tmp_path / "out.json"
    cli = DiarizeCLI(
        argv=[
            "--audio",
            str(audio),
            "--output",
            str(output),
            "--mode",
            "diarize",
            "--model-name",
            "pyannote/custom",
            "--hf-token",
            "my-token",
        ]
    )
    service = cli.build_service()
    assert isinstance(service, DiarizeService)
    assert service._audio_file == audio
    assert service._output_file == output
    assert service._mode is OperationMode.DIARIZE
    # Factory was built with the parsed values
    assert service._processor_factory._model_name == "pyannote/custom"
    assert service._processor_factory._hf_token == "my-token"


def test_diarize_cli_falls_back_to_env_hf_token(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    cli = DiarizeCLI(
        argv=[
            "--audio",
            str(tmp_path / "audio.npy"),
            "--output",
            str(tmp_path / "out.json"),
            "--mode",
            "vad",
        ]
    )
    service = cli.build_service()
    assert service._processor_factory._hf_token == "env-token"


def test_diarize_cli_rejects_unknown_mode(tmp_path: Path):
    cli = DiarizeCLI(
        argv=[
            "--audio",
            str(tmp_path / "audio.npy"),
            "--output",
            str(tmp_path / "out.json"),
            "--mode",
            "transcribe",
        ]
    )
    with pytest.raises(SystemExit):
        cli.build_service()


# ---------------------------------------------------------------------------
# DiarizeService error path (does not require torch / whisperx)
# ---------------------------------------------------------------------------


def test_diarize_service_writes_error_response_when_audio_load_fails(
    tmp_path: Path,
):
    output = tmp_path / "result.json"
    factory = ProcessorFactory(model_name="pyannote/test", hf_token=None)
    service = DiarizeService(
        audio_file=tmp_path / "missing.npy",
        output_file=output,
        mode=OperationMode.VAD,
        processor_factory=factory,
    )
    exit_code = service.run()
    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["status"] == "error"
    assert payload["error"]  # non-empty error message


def test_diarize_service_writes_success_response_with_stubbed_processor(
    tmp_path: Path, monkeypatch
):
    import numpy as np

    audio_path = tmp_path / "audio.npy"
    np.save(audio_path, np.zeros(16, dtype=np.float32))

    expected = [AudioSegment(start=0.0, end=1.0, speaker="SPEAKER_00")]

    class StubProcessor:
        def process(self, audio_array):
            assert audio_array.dtype == np.float32
            return expected

    class StubFactory:
        def create(self, mode):
            assert mode is OperationMode.DIARIZE
            return StubProcessor()

    output = tmp_path / "result.json"
    service = DiarizeService(
        audio_file=audio_path,
        output_file=output,
        mode=OperationMode.DIARIZE,
        processor_factory=StubFactory(),
    )
    assert service.run() == 0

    payload = json.loads(output.read_text())
    assert payload["status"] == "success"
    assert payload["segments"] == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

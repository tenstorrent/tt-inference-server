# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import types
from pathlib import Path

import pytest

from utils.diarize import (
    AudioProcessor,
    AudioSegment,
    DiarizationProcessor,
    DiarizeCLI,
    DiarizeResponse,
    DiarizeServer,
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


# ---------------------------------------------------------------------------
# DiarizeServer (persistent worker mode)
# ---------------------------------------------------------------------------


class _StubProcessor(AudioProcessor):
    """Records process() calls; returns a canned segment list per mode."""

    def __init__(self, mode: OperationMode, segments: list[AudioSegment]):
        self.mode = mode
        self._segments = segments
        self.calls: list[int] = []

    def process(self, audio_array):
        self.calls.append(len(audio_array))
        return list(self._segments)


class _StubFactory:
    """Factory that hands out a StubProcessor per create() call and records
    how many times each mode has been created. With lazy processor
    construction we expect exactly one create() per mode that receives
    traffic — and zero for modes that never do."""

    def __init__(self, segments_by_mode: dict[OperationMode, list[AudioSegment]]):
        self._segments = segments_by_mode
        self.built: dict[OperationMode, _StubProcessor] = {}
        self.create_counts: dict[OperationMode, int] = {}

    def create(self, mode: OperationMode) -> _StubProcessor:
        self.create_counts[mode] = self.create_counts.get(mode, 0) + 1
        processor = _StubProcessor(mode, self._segments[mode])
        self.built[mode] = processor
        return processor


def _make_audio_file(tmp_path: Path, n: int = 32) -> Path:
    import numpy as np

    audio_path = tmp_path / "audio.npy"
    np.save(audio_path, np.zeros(n, dtype=np.float32))
    return audio_path


def _run_server_with(stdin_text: str, factory) -> tuple[int, list[dict], str]:
    """Drive DiarizeServer with a scripted stdin, return (exit, responses, stderr)."""
    import io

    stdout = io.StringIO()
    stderr = io.StringIO()
    server = DiarizeServer(
        processor_factory=factory,
        stdin=io.StringIO(stdin_text),
        stdout=stdout,
        stderr=stderr,
    )
    exit_code = server.serve()
    responses = [
        json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()
    ]
    return exit_code, responses, stderr.getvalue()


def test_diarize_server_emits_ready_without_building_any_processor(tmp_path: Path):
    """With lazy construction, ready comes back before the factory is ever
    called. This makes worker startup fast; the model load happens on the
    first real request (typically the CpuWorkloadHandler warmup task)."""
    factory = _StubFactory(
        {OperationMode.DIARIZE: [], OperationMode.VAD: []},
    )
    # Empty stdin → EOF immediately → server just emits ready and exits.
    exit_code, responses, _ = _run_server_with("", factory)

    assert exit_code == 0
    assert responses == [{"status": "ready"}]
    # Zero processors built: no traffic, no work.
    assert factory.built == {}
    assert factory.create_counts == {}


def test_diarize_server_handles_diarize_request_end_to_end(tmp_path: Path):
    expected = [AudioSegment(start=0.0, end=1.0, speaker="SPEAKER_00")]
    factory = _StubFactory(
        {OperationMode.DIARIZE: expected, OperationMode.VAD: []},
    )
    audio_path = _make_audio_file(tmp_path)
    request = {"id": "abc-123", "mode": "diarize", "audio_path": str(audio_path)}

    exit_code, responses, _ = _run_server_with(json.dumps(request) + "\n", factory)

    assert exit_code == 0
    assert len(responses) == 2
    assert responses[0] == {"status": "ready"}
    assert responses[1] == {
        "id": "abc-123",
        "status": "success",
        "error": None,
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
    }
    assert factory.built[OperationMode.DIARIZE].calls == [32]


def test_diarize_server_processes_multiple_requests_reusing_same_processor(
    tmp_path: Path,
):
    """A single VADProcessor instance should be built on first request and
    then serve every subsequent VAD request from the same cached instance.
    A mode with zero traffic (diarize here) should never be built at all."""
    factory = _StubFactory(
        {OperationMode.DIARIZE: [], OperationMode.VAD: [AudioSegment(0.0, 0.5)]},
    )
    audio_path = _make_audio_file(tmp_path, n=8)
    stdin = "".join(
        json.dumps({"id": f"r{i}", "mode": "vad", "audio_path": str(audio_path)}) + "\n"
        for i in range(3)
    )
    exit_code, responses, _ = _run_server_with(stdin, factory)

    assert exit_code == 0
    # 1 ready + 3 request responses
    assert len(responses) == 4
    ids = [r["id"] for r in responses[1:]]
    assert ids == ["r0", "r1", "r2"]
    # Exactly one construction for the mode that got traffic; none for the
    # mode that didn't.
    assert factory.create_counts == {OperationMode.VAD: 1}
    assert factory.built[OperationMode.VAD].calls == [8, 8, 8]


def test_diarize_server_returns_error_for_invalid_mode(tmp_path: Path):
    factory = _StubFactory({OperationMode.DIARIZE: [], OperationMode.VAD: []})
    request = {"id": "x", "mode": "transcribe", "audio_path": "/tmp/whatever.npy"}

    _, responses, _ = _run_server_with(json.dumps(request) + "\n", factory)

    assert responses[-1]["status"] == "error"
    assert responses[-1]["id"] == "x"
    assert "invalid mode" in responses[-1]["error"]


def test_diarize_server_returns_error_when_audio_path_missing(tmp_path: Path):
    factory = _StubFactory({OperationMode.DIARIZE: [], OperationMode.VAD: []})
    request = {"id": "y", "mode": "vad"}  # audio_path missing

    _, responses, _ = _run_server_with(json.dumps(request) + "\n", factory)

    assert responses[-1] == {
        "id": "y",
        "status": "error",
        "error": "missing audio_path",
        "segments": [],
    }


def test_diarize_server_returns_error_when_audio_file_unreadable(tmp_path: Path):
    factory = _StubFactory({OperationMode.DIARIZE: [], OperationMode.VAD: []})
    request = {"id": "z", "mode": "vad", "audio_path": str(tmp_path / "nope.npy")}

    _, responses, _ = _run_server_with(json.dumps(request) + "\n", factory)

    assert responses[-1]["status"] == "error"
    assert responses[-1]["id"] == "z"
    assert responses[-1]["error"]  # some numpy IOError


def test_diarize_server_skips_blank_lines_and_reports_malformed_json(tmp_path: Path):
    factory = _StubFactory({OperationMode.DIARIZE: [], OperationMode.VAD: []})
    audio_path = _make_audio_file(tmp_path)
    good = json.dumps({"id": "g", "mode": "vad", "audio_path": str(audio_path)})
    stdin = "\n\n{not valid json\n" + good + "\n"

    _, responses, _ = _run_server_with(stdin, factory)

    # ready + malformed-error + good response
    statuses = [r.get("status") for r in responses]
    assert statuses == ["ready", "error", "success"]
    assert "malformed request" in responses[1]["error"]


def test_diarize_server_returns_error_when_factory_fails_on_first_request(
    tmp_path: Path,
):
    """With lazy construction, a broken factory only surfaces when the first
    matching request arrives — the server stays alive and returns an error
    payload instead of crashing. The caller's next request has a chance to
    succeed (e.g. if the failure was transient)."""
    audio_path = _make_audio_file(tmp_path)
    request = {"id": "boom", "mode": "diarize", "audio_path": str(audio_path)}

    class ExplodingFactory:
        def __init__(self):
            self.calls = 0

        def create(self, mode):
            self.calls += 1
            raise RuntimeError("cannot download model")

    factory = ExplodingFactory()
    exit_code, responses, _ = _run_server_with(json.dumps(request) + "\n", factory)

    assert exit_code == 0  # server keeps running to serve future requests
    assert responses[0] == {"status": "ready"}
    assert responses[1] == {
        "id": "boom",
        "status": "error",
        "error": "cannot download model",
        "segments": [],
    }
    assert factory.calls == 1


# ---------------------------------------------------------------------------
# DiarizeCLI --serve wiring
# ---------------------------------------------------------------------------


def test_diarize_cli_build_serve_produces_diarize_server(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cli = DiarizeCLI(argv=["--serve", "--model-name", "pyannote/x", "--hf-token", "t"])
    server = cli.build_serve()
    assert isinstance(server, DiarizeServer)
    # Factory captured our args
    assert server._factory._model_name == "pyannote/x"
    assert server._factory._hf_token == "t"


def test_diarize_cli_build_serve_falls_back_to_env_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-token")
    cli = DiarizeCLI(argv=["--serve"])
    server = cli.build_serve()
    assert server._factory._hf_token == "env-token"


def test_diarize_cli_build_service_rejects_serve_flag():
    cli = DiarizeCLI(argv=["--serve"])
    with pytest.raises(SystemExit):
        cli.build_service()


def test_diarize_cli_build_serve_rejects_missing_serve_flag(tmp_path: Path):
    cli = DiarizeCLI(
        argv=[
            "--audio",
            str(tmp_path / "a.npy"),
            "--output",
            str(tmp_path / "o.json"),
            "--mode",
            "vad",
        ]
    )
    with pytest.raises(SystemExit):
        cli.build_serve()


def test_diarize_cli_build_service_reports_missing_one_shot_args():
    cli = DiarizeCLI(argv=[])
    with pytest.raises(SystemExit):
        cli.build_service()

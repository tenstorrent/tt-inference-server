# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import io
import wave
from unittest.mock import patch

import numpy as np
import pytest

# audio_manager no longer imports torch/whisperx at module load time; those
# packages live in the separate audio_venv and are only invoked via the
# subprocess client. We can import freely without any module-load workaround.
from utils.audio_manager import AudioManager, AudioVenvClient


class DummySettings:
    allow_audio_preprocessing = False
    default_sample_rate = 16000
    max_audio_size_bytes = 1000000
    max_audio_duration_seconds = 60
    max_audio_duration_with_preprocessing_seconds = 120
    audio_chunk_duration_seconds = 10
    model_service = "AUDIO"
    preprocessing_model_weights_path = None


def generate_dummy_wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 320)  # 160 samples
    return buffer.getvalue()


@patch("utils.audio_manager.settings", new=DummySettings())
def test_to_audio_array_base64():
    manager = AudioManager()
    wav_bytes = generate_dummy_wav_bytes()
    import base64

    b64 = base64.b64encode(wav_bytes).decode()
    arr, duration = manager.to_audio_array(b64, False)
    assert isinstance(arr, np.ndarray)
    assert duration > 0


@patch("utils.audio_manager.settings", new=DummySettings())
def test_to_audio_array_bytes():
    manager = AudioManager()
    wav_bytes = generate_dummy_wav_bytes()
    arr, duration = manager.to_audio_array(wav_bytes, False)
    assert isinstance(arr, np.ndarray)
    assert duration > 0


@patch("utils.audio_manager.settings", new=DummySettings())
def test_to_audio_array_invalid_type():
    manager = AudioManager()
    with pytest.raises(ValueError):
        manager.to_audio_array(123, False)


@patch("utils.audio_manager.settings", new=DummySettings())
def test_validate_file_size():
    manager = AudioManager()
    manager._validate_file_size(b"\x00" * 100)
    with pytest.raises(ValueError):
        manager._validate_file_size(b"\x00" * (DummySettings.max_audio_size_bytes + 1))


@patch("utils.audio_manager.settings", new=DummySettings())
def test_validate_and_truncate_duration():
    manager = AudioManager()
    arr = np.zeros(DummySettings.default_sample_rate * 70, dtype=np.float32)
    truncated, duration = manager._validate_and_truncate_duration(arr, False)
    assert duration == DummySettings.max_audio_duration_seconds
    assert (
        len(truncated)
        == DummySettings.default_sample_rate * DummySettings.max_audio_duration_seconds
    )


@patch("utils.audio_manager.settings", new=DummySettings())
def test_normalize_speaker_ids():
    manager = AudioManager()
    segments = [
        {"start": 0, "end": 1, "speaker": "A"},
        {"start": 1, "end": 2, "speaker": "B"},
        {"start": 2, "end": 3, "speaker": "A"},
    ]
    norm = manager._normalize_speaker_ids(segments)
    assert all("speaker" in s for s in norm)
    assert norm[0]["speaker"] == "SPEAKER_00"
    assert norm[1]["speaker"] == "SPEAKER_01"
    assert norm[2]["speaker"] == "SPEAKER_00"


# ---------------------------------------------------------------------------
# AudioVenvClient
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Minimal logger stand-in that just records messages so tests can assert."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg):
        self.errors.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


def test_audio_venv_client_assert_available_raises_when_missing(tmp_path):
    client = AudioVenvClient(
        logger=_RecordingLogger(),
        python_executable=str(tmp_path / "does-not-exist"),
        script_path=tmp_path / "does-not-exist.py",
    )
    assert client.is_available() is False
    with pytest.raises(FileNotFoundError):
        client.assert_available()


def test_audio_venv_client_run_returns_segments_on_success(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    fake_python = tmp_path / "python"
    fake_script = tmp_path / "diarize.py"
    fake_python.write_text("")
    fake_script.write_text("")

    expected_segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    def fake_run(cmd, capture_output, text, timeout):
        # cmd should contain --audio, --output, --mode, and our optional flags
        assert "--audio" in cmd
        assert "--output" in cmd
        assert "--mode" in cmd
        assert "diarize" in cmd
        assert "--model-name" in cmd
        assert "pyannote/test" in cmd
        assert "--hf-token" in cmd
        assert "secret" in cmd
        # Simulate the subprocess writing its JSON response to the output path
        output_idx = cmd.index("--output") + 1
        import json as _json

        with open(cmd[output_idx], "w") as f:
            _json.dump({"status": "success", "segments": expected_segments}, f)

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr("utils.audio_manager.subprocess.run", fake_run)

    client = AudioVenvClient(
        logger=logger,
        python_executable=str(fake_python),
        script_path=fake_script,
    )
    segments = client.run(
        mode="diarize",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=10,
        model_name="pyannote/test",
        hf_token="secret",
    )
    assert segments == expected_segments
    assert logger.errors == []


def test_audio_venv_client_run_returns_none_on_nonzero_exit(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    fake_python = tmp_path / "python"
    fake_script = tmp_path / "diarize.py"
    fake_python.write_text("")
    fake_script.write_text("")

    def fake_run(cmd, capture_output, text, timeout):
        class _Result:
            returncode = 1
            stderr = "boom"

        return _Result()

    monkeypatch.setattr("utils.audio_manager.subprocess.run", fake_run)

    client = AudioVenvClient(
        logger=logger,
        python_executable=str(fake_python),
        script_path=fake_script,
    )
    segments = client.run(
        mode="vad",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=10,
    )
    assert segments is None
    assert any("boom" in e for e in logger.errors)


def test_audio_venv_client_run_returns_none_on_timeout(tmp_path, monkeypatch):
    import subprocess as _subprocess

    logger = _RecordingLogger()
    fake_python = tmp_path / "python"
    fake_script = tmp_path / "diarize.py"
    fake_python.write_text("")
    fake_script.write_text("")

    def fake_run(cmd, capture_output, text, timeout):
        raise _subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr("utils.audio_manager.subprocess.run", fake_run)

    client = AudioVenvClient(
        logger=logger,
        python_executable=str(fake_python),
        script_path=fake_script,
    )
    segments = client.run(
        mode="diarize",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=1,
    )
    assert segments is None
    assert any("timed out" in e for e in logger.errors)


def test_audio_venv_client_run_returns_none_on_error_payload(tmp_path, monkeypatch):
    logger = _RecordingLogger()
    fake_python = tmp_path / "python"
    fake_script = tmp_path / "diarize.py"
    fake_python.write_text("")
    fake_script.write_text("")

    def fake_run(cmd, capture_output, text, timeout):
        output_idx = cmd.index("--output") + 1
        import json as _json

        with open(cmd[output_idx], "w") as f:
            _json.dump({"status": "error", "error": "model not found"}, f)

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr("utils.audio_manager.subprocess.run", fake_run)

    client = AudioVenvClient(
        logger=logger,
        python_executable=str(fake_python),
        script_path=fake_script,
    )
    segments = client.run(
        mode="diarize",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=10,
    )
    assert segments is None
    assert any("model not found" in e for e in logger.errors)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import io
import json
import os
import wave
from unittest.mock import patch

import numpy as np
import pytest

# audio_manager no longer imports torch/whisperx at module load time; those
# packages live in the separate audio_venv and are only invoked through the
# persistent worker subprocess. We can import freely without any workaround.
from utils.audio_manager import AudioManager, AudioVenvWorker


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
# AudioVenvWorker
#
# AudioVenvWorker talks to a long-lived `diarize.py --serve` subprocess over
# stdin/stdout. To test its I/O + respawn behaviour without spinning up an
# actual audio venv we inject a fake Popen backed by a real `os.pipe()` for
# stdout (so `selectors.select()` can see the fd become readable) and a plain
# `StringIO` for stdin.
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Minimal logger stand-in that records messages so tests can assert."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg):
        self.errors.append(str(msg))

    def warning(self, msg):
        self.warnings.append(str(msg))

    def info(self, msg):
        self.infos.append(str(msg))


class _FakePopen:
    """Real-pipe backed fake of `subprocess.Popen`.

    Tests drive it via :meth:`emit` (write one JSON line to the readable side
    of stdout) and :meth:`emit_eof` (simulate the worker closing stdout).
    """

    def __init__(self):
        read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "r", buffering=1, encoding="utf-8")
        self.stdin = io.StringIO()
        self.stderr = io.StringIO()
        self.returncode = None
        self._write_fd_closed = False

    def emit(self, payload: dict) -> None:
        os.write(self._write_fd, (json.dumps(payload) + "\n").encode())

    def emit_line(self, line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        os.write(self._write_fd, line.encode())

    def emit_eof(self) -> None:
        if not self._write_fd_closed:
            os.close(self._write_fd)
            self._write_fd_closed = True

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15
        self.emit_eof()

    def kill(self):
        if self.returncode is None:
            self.returncode = -9
        self.emit_eof()

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def cleanup(self) -> None:
        try:
            self.emit_eof()
        except OSError:
            pass
        try:
            self.stdout.close()
        except Exception:
            pass


class _PopenFactory:
    """Records every spawn and lets the test control what each fake worker
    emits on start. Used as the `popen_factory` argument to AudioVenvWorker."""

    def __init__(self):
        self.spawned: list[_FakePopen] = []
        self.commands: list[list[str]] = []
        self._on_spawn = lambda proc: proc.emit({"status": "ready"})

    def set_on_spawn(self, fn):
        self._on_spawn = fn

    def __call__(self, cmd, **kwargs):
        proc = _FakePopen()
        self.spawned.append(proc)
        self.commands.append(cmd)
        self._on_spawn(proc)
        return proc

    def cleanup(self) -> None:
        for proc in self.spawned:
            proc.cleanup()


def _touch(path):
    path.write_text("")
    return path


@pytest.fixture
def worker_env(tmp_path):
    """Yields (logger, popen_factory, python_path, script_path).
    Cleans up all fake popen pipes at the end regardless of test outcome."""
    logger = _RecordingLogger()
    factory = _PopenFactory()
    python_path = _touch(tmp_path / "python")
    script_path = _touch(tmp_path / "diarize.py")
    try:
        yield logger, factory, str(python_path), script_path
    finally:
        factory.cleanup()


def _make_worker(env, *, model_name="pyannote/test", hf_token="secret"):
    logger, factory, python_path, script_path = env
    return AudioVenvWorker(
        logger=logger,
        model_name=model_name,
        hf_token=hf_token,
        python_executable=python_path,
        script_path=script_path,
        popen_factory=factory,
    )


# ---------- availability / spawn -------------------------------------------


def test_audio_venv_worker_assert_available_raises_when_python_missing(tmp_path):
    worker = AudioVenvWorker(
        logger=_RecordingLogger(),
        python_executable=str(tmp_path / "does-not-exist"),
        script_path=tmp_path / "does-not-exist.py",
    )
    assert worker.is_available() is False
    with pytest.raises(FileNotFoundError):
        worker.assert_available()


def test_start_spawns_with_expected_command_and_reads_ready(worker_env):
    worker = _make_worker(worker_env)
    worker.start()

    _, factory, python_path, script_path = worker_env
    assert len(factory.spawned) == 1
    cmd = factory.commands[0]
    assert cmd[:3] == [python_path, str(script_path), "--serve"]
    assert "--model-name" in cmd and "pyannote/test" in cmd
    assert "--hf-token" in cmd and "secret" in cmd
    assert worker.is_running() is True


def test_start_is_idempotent_when_worker_already_running(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    worker.start()  # second call must be a no-op

    _, factory, *_ = worker_env
    assert len(factory.spawned) == 1


def test_start_raises_when_ready_never_comes(worker_env, monkeypatch):
    logger, factory, *_ = worker_env
    # Override on_spawn to write nothing, and shrink the ready timeout so the
    # test doesn't hang.
    factory.set_on_spawn(lambda _proc: None)
    monkeypatch.setattr(AudioVenvWorker, "_READY_TIMEOUT_SECONDS", 0.1)

    worker = _make_worker(worker_env)
    with pytest.raises(RuntimeError, match="did not signal ready"):
        worker.start()
    assert worker.is_running() is False


def test_start_raises_on_unexpected_ready_payload(worker_env):
    _, factory, *_ = worker_env
    factory.set_on_spawn(lambda proc: proc.emit({"status": "boom"}))

    worker = _make_worker(worker_env)
    with pytest.raises(RuntimeError, match="unexpected ready payload"):
        worker.start()


def test_start_omits_model_flags_when_not_configured(worker_env):
    logger, factory, python_path, script_path = worker_env
    worker = AudioVenvWorker(
        logger=logger,
        model_name=None,
        hf_token=None,
        python_executable=python_path,
        script_path=script_path,
        popen_factory=factory,
    )
    worker.start()

    cmd = factory.commands[0]
    assert "--model-name" not in cmd
    assert "--hf-token" not in cmd


# ---------- run: happy path -------------------------------------------------


def _last_request_from_stdin(proc: _FakePopen) -> dict:
    lines = [line for line in proc.stdin.getvalue().splitlines() if line.strip()]
    assert lines, "no request was written to worker stdin"
    return json.loads(lines[-1])


def test_run_returns_segments_on_success(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]

    expected = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    # Set up the "server side": echo back a success payload with the same id
    # that AudioVenvWorker sends. We do this before calling run() by
    # intercepting stdin — but the id is generated inside run(). Simpler:
    # emit an "answer" that matches whatever id gets sent, by reading stdin
    # inside a side-effect. Since our fake stdin is a StringIO written to
    # synchronously, we can pre-arrange: patch stdin.write to react.
    original_write = proc.stdin.write

    def reactive_write(line):
        result = original_write(line)
        proc.stdin.flush()
        req = json.loads(line)
        proc.emit(
            {
                "id": req["id"],
                "status": "success",
                "error": None,
                "segments": expected,
            }
        )
        return result

    proc.stdin.write = reactive_write

    segments = worker.run(
        mode="diarize",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=5,
    )
    assert segments == expected

    sent = _last_request_from_stdin(proc)
    assert sent["mode"] == "diarize"
    assert sent["audio_path"].endswith(".npy")
    assert "id" in sent


def test_run_returns_none_on_error_payload(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]

    original_write = proc.stdin.write

    def reactive_write(line):
        result = original_write(line)
        req = json.loads(line)
        proc.emit(
            {
                "id": req["id"],
                "status": "error",
                "error": "model not found",
                "segments": [],
            }
        )
        return result

    proc.stdin.write = reactive_write

    segments = worker.run(
        mode="vad",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=5,
    )
    assert segments is None
    logger = worker_env[0]
    assert any("model not found" in e for e in logger.errors)


def test_run_returns_none_on_id_mismatch_and_drops_worker(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]

    original_write = proc.stdin.write

    def reactive_write(line):
        result = original_write(line)
        proc.emit(
            {
                "id": "wrong-id",
                "status": "success",
                "error": None,
                "segments": [{"start": 0.0, "end": 1.0}],
            }
        )
        return result

    proc.stdin.write = reactive_write

    logger = worker_env[0]
    segments = worker.run(
        mode="vad",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=5,
    )
    assert segments is None
    assert any("id mismatch" in e for e in logger.errors)
    assert worker.is_running() is False


def test_run_returns_none_on_timeout_and_terminates_worker(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]
    # Do NOT wire a reactive_write, so the worker "never responds".

    logger = worker_env[0]
    segments = worker.run(
        mode="vad",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=0.1,
    )
    assert segments is None
    assert any("timed out" in e for e in logger.errors)
    # After timeout the worker must be gone so the next call respawns.
    assert worker.is_running() is False
    assert proc.returncode is not None  # was terminated


def test_run_returns_none_on_worker_eof(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]

    original_write = proc.stdin.write

    def reactive_write(line):
        result = original_write(line)
        proc.emit_eof()  # worker died mid-request
        return result

    proc.stdin.write = reactive_write

    logger = worker_env[0]
    segments = worker.run(
        mode="vad",
        audio_array=np.zeros(16, dtype=np.float32),
        timeout_seconds=5,
    )
    assert segments is None
    assert any("closed stdout unexpectedly" in e for e in logger.errors)
    assert worker.is_running() is False


# ---------- respawn --------------------------------------------------------


def test_run_respawns_worker_after_it_dies(worker_env):
    """If the worker died between requests, the next run() call must spin up
    a fresh subprocess automatically."""
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    first = factory.spawned[0]

    # Simulate death outside of any request.
    first.terminate()
    assert worker.is_running() is False

    # Next request should transparently spawn a second Popen. Wire a
    # reactive server on the *next* spawn so we can capture the successful
    # response.
    def on_next_spawn(proc):
        proc.emit({"status": "ready"})
        original_write = proc.stdin.write

        def reactive_write(line):
            result = original_write(line)
            req = json.loads(line)
            proc.emit(
                {
                    "id": req["id"],
                    "status": "success",
                    "error": None,
                    "segments": [{"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}],
                }
            )
            return result

        proc.stdin.write = reactive_write

    factory.set_on_spawn(on_next_spawn)

    segments = worker.run(
        mode="diarize",
        audio_array=np.zeros(8, dtype=np.float32),
        timeout_seconds=5,
    )

    assert segments == [{"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert len(factory.spawned) == 2  # original + respawn
    assert worker.is_running() is True


def test_run_returns_none_when_respawn_itself_fails(worker_env, monkeypatch):
    """If the respawn attempt itself can't complete, run() surfaces None
    instead of raising, and logs the failure."""
    logger, factory, *_ = worker_env
    monkeypatch.setattr(AudioVenvWorker, "_READY_TIMEOUT_SECONDS", 0.05)
    factory.set_on_spawn(lambda _proc: None)  # never signal ready

    worker = _make_worker(worker_env)
    segments = worker.run(
        mode="vad",
        audio_array=np.zeros(4, dtype=np.float32),
        timeout_seconds=5,
    )
    assert segments is None
    assert any("Cannot start audio worker" in e for e in logger.errors)


# ---------- close ----------------------------------------------------------


def test_close_shuts_down_the_worker_cleanly(worker_env):
    worker = _make_worker(worker_env)
    worker.start()
    _, factory, *_ = worker_env
    proc = factory.spawned[0]

    worker.close()

    assert worker.is_running() is False
    # stdin was closed, which is the EOF signal the real server watches for.
    assert proc.stdin.closed is True


def test_close_is_a_noop_when_worker_never_started(worker_env):
    worker = _make_worker(worker_env)
    # Must not raise, must not spawn anything.
    worker.close()
    _, factory, *_ = worker_env
    assert factory.spawned == []

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from config.constants import ModelServices
from device_workers.encode_stage import EncodeStage
from utils.video_manager import VideoAudioResult

_TIMEOUT_S = 5.0


@pytest.fixture
def video_manager():
    manager = MagicMock()
    manager.export_to_mp4.side_effect = lambda frames: (
        f"/tmp/{int(frames[0, 0, 0, 0])}.mp4"
    )
    manager.export_to_mp4_with_audio.return_value = "/tmp/av.mp4"
    with patch("utils.video_manager.VideoManager", return_value=manager):
        yield manager


def _frames(tag: int):
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames[0, 0, 0, 0] = tag
    return frames


def _drain(q, n):
    return [q.get(timeout=_TIMEOUT_S) for _ in range(n)]


class TestEncodeStage:
    def test_results_in_submission_order(self, video_manager):
        results, errors = queue.Queue(), queue.Queue()
        stage = EncodeStage("w0", results, errors)
        for tag in (1, 2, 3):
            stage.submit(f"t{tag}", _frames(tag))
        stage.close()
        assert _drain(results, 3) == [
            ("w0", "t1", "/tmp/1.mp4"),
            ("w0", "t2", "/tmp/2.mp4"),
            ("w0", "t3", "/tmp/3.mp4"),
        ]
        assert errors.empty()

    def test_audio_result_is_muxed(self, video_manager):
        results = queue.Queue()
        stage = EncodeStage("w0", results, queue.Queue())
        audio_result = VideoAudioResult(
            _frames(0), np.zeros(10), 48000, fps=24, pixel_format="yuv420p"
        )
        stage.submit("t", audio_result)
        stage.close()
        assert results.get(timeout=_TIMEOUT_S) == ("w0", "t", "/tmp/av.mp4")
        args, kwargs = video_manager.export_to_mp4_with_audio.call_args
        assert args[2] == 48000
        assert kwargs == {"fps": 24, "pixel_format": "yuv420p"}

    def test_existing_path_passes_through(self, video_manager):
        results = queue.Queue()
        stage = EncodeStage("w0", results, queue.Queue())
        stage.submit("t", "/already/exported.mp4")
        stage.close()
        assert results.get(timeout=_TIMEOUT_S) == ("w0", "t", "/already/exported.mp4")
        video_manager.export_to_mp4.assert_not_called()

    def test_encode_error_goes_to_error_queue_and_stage_continues(self, video_manager):
        video_manager.export_to_mp4.side_effect = [
            RuntimeError("ffmpeg died"),
            "/ok.mp4",
        ]
        results, errors = queue.Queue(), queue.Queue()
        stage = EncodeStage("w0", results, errors)
        stage.submit("bad", _frames(1))
        stage.submit("good", _frames(2))
        stage.close()
        worker_id, task_id, message = errors.get(timeout=_TIMEOUT_S)
        assert (worker_id, task_id) == ("w0", "bad")
        assert "ffmpeg died" in message
        assert results.get(timeout=_TIMEOUT_S) == ("w0", "good", "/ok.mp4")

    def test_close_drains_queued_jobs(self, video_manager):
        gate = threading.Event()

        def slow_export(frames):
            gate.wait(_TIMEOUT_S)
            return "/slow.mp4"

        video_manager.export_to_mp4.side_effect = slow_export
        results = queue.Queue()
        stage = EncodeStage("w0", results, queue.Queue())
        for tag in range(3):
            stage.submit(f"t{tag}", _frames(tag))
        threading.Timer(0.1, gate.set).start()
        stage.close()
        assert [r[1] for r in _drain(results, 3)] == ["t0", "t1", "t2"]

    def test_submit_blocks_when_encoder_is_behind(self, video_manager):
        gate = threading.Event()
        video_manager.export_to_mp4.side_effect = lambda frames: gate.wait(_TIMEOUT_S)
        stage = EncodeStage("w0", queue.Queue(), queue.Queue())
        # Taken by the encoder thread.
        stage.submit("t0", _frames(0))
        stage.submit("t1", _frames(1))
        # Queue now full.
        stage.submit("t2", _frames(2))
        blocked = threading.Thread(target=stage.submit, args=("t3", _frames(3)))
        blocked.start()
        blocked.join(0.2)
        assert blocked.is_alive()
        gate.set()
        blocked.join(_TIMEOUT_S)
        assert not blocked.is_alive()
        stage.close()


class TestForRunner:
    def _settings(self, service=ModelServices.VIDEO.value, enabled=True):
        return SimpleNamespace(model_service=service, video_async_encode=enabled)

    def test_video_runner_gets_stage_and_export_turned_off(self):
        runner = SimpleNamespace(export_in_runner=True)
        stage = EncodeStage.for_runner(
            runner, self._settings(), "w0", queue.Queue(), queue.Queue()
        )
        assert stage is not None
        assert runner.export_in_runner is False
        stage.close()

    def test_disabled(self):
        runner = SimpleNamespace(export_in_runner=True)
        assert (
            EncodeStage.for_runner(
                runner, self._settings(enabled=False), "w0", None, None
            )
            is None
        )
        assert runner.export_in_runner is True

    def test_not_video(self):
        runner = SimpleNamespace(export_in_runner=True)
        settings = self._settings(service=ModelServices.IMAGE.value)
        assert EncodeStage.for_runner(runner, settings, "w0", None, None) is None
        assert runner.export_in_runner is True

    def test_normalize_single_audio_result(self):
        audio_result = VideoAudioResult(_frames(0), np.zeros(1), 16000)
        assert EncodeStage.normalize_responses(audio_result) == [audio_result]
        assert EncodeStage.normalize_responses(["/a.mp4"]) == ["/a.mp4"]

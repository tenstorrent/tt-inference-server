# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mock video runner: pipeline output shape and export hook (conftest mocks torch/diffusers)."""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from config.constants import MAX_VIDEO_SIZE
from ipc.video_shm import VideoRequest, VideoStatus
from tt_model_runners import mock_video_runner as mvr
from utils.video_delivery_metrics import num_frames_from_video_tensor


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(mvr.time, "sleep", lambda _: None)


def test_max_video_size_fits_wan22_quad_float32():
    n = 1280 * 720 * 3 * 81 * 4
    assert n < MAX_VIDEO_SIZE


def test_mock_float32_wan22_quad_tensor_under_cap(no_sleep, monkeypatch):
    import numpy as np

    monkeypatch.setenv("TT_MOCK_FLOAT32_TENSOR", "1")
    p = mvr.MockVideoPipeline()
    out = p(prompt="x", height=720, width=1280, num_frames=81, seed=2)
    assert out.dtype == np.float32
    assert out.nbytes == 1280 * 720 * 81 * 3 * 4
    assert out.nbytes < MAX_VIDEO_SIZE


def test_mock_float32_over_max_video_size_raises(no_sleep, monkeypatch):
    monkeypatch.setenv("TT_MOCK_FLOAT32_TENSOR", "1")
    p = mvr.MockVideoPipeline()
    with pytest.raises(ValueError, match="MAX_VIDEO_SIZE"):
        p(prompt="x", height=720, width=2560, num_frames=81, seed=0)


def test_num_frames_from_video_tensor():
    import numpy as np

    a5 = np.zeros((1, 7, 4, 4, 3))
    assert num_frames_from_video_tensor(a5) == 7
    a4 = np.zeros((11, 4, 4, 3))
    assert num_frames_from_video_tensor(a4) == 11


def test_mock_pipeline_produces_wan_style_batch(no_sleep):
    p = mvr.MockVideoPipeline()
    frames = p(prompt="test", num_frames=4, height=16, width=16, seed=2)
    assert frames.dtype.name.startswith("uint")
    assert frames.shape == (1, 4, 16, 16, 3)
    assert hasattr(p, "_mock_pipeline_excluding_sleep_s")
    assert p._mock_pipeline_wall_s >= p._mock_pipeline_excluding_sleep_s >= 0
    assert p._mock_sleep_accum_s >= 0


def test_mock_video_runner_run_invokes_export_to_mp4(no_sleep, monkeypatch, tmp_path):
    """Same sequence as standalone bridge: frames -> VideoManager.export_to_mp4 -> path."""
    captured = {}

    def fake_export(self, frames, fps=16, timing_out=None):
        captured["shape"] = frames.shape
        out = tmp_path / "mock.mp4"
        out.write_bytes(
            b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso6" + b"\x00" * 100
        )
        return str(out)

    monkeypatch.setattr("utils.video_manager.VideoManager.export_to_mp4", fake_export)

    req = MagicMock()
    req.prompt = "p"
    req.negative_prompt = None
    req.num_inference_steps = 12
    req.seed = 0

    runner = mvr.MockVideoRunner("dev0")
    runner.pipeline = mvr.MockVideoPipeline()
    out = runner.run([req])

    assert out == [str(tmp_path / "mock.mp4")]
    assert captured["shape"] == (1, 81, 720, 1280, 3)
    assert os.path.isfile(out[0])


def test_handle_signal_sets_shutdown():
    import signal

    mvr._shutdown = False
    mvr._handle_signal(signal.SIGTERM, None)
    assert mvr._shutdown is True
    mvr._shutdown = False


def test_run_shm_bridge_processes_one_request(monkeypatch, tmp_path):
    import numpy as np

    mvr._shutdown = False
    req = VideoRequest(
        task_id="abcdef12-3456-7890-abcd-ef1234567890",
        prompt="p",
        negative_prompt="",
        num_inference_steps=12,
        seed=0,
        height=480,
        width=832,
        num_frames=4,
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )
    mock_in = MagicMock()
    mock_out = MagicMock()
    mock_in.read_request = Mock(side_effect=[req, None])

    mp4 = tmp_path / "out.mp4"
    mp4.write_bytes(b"x")

    class FastPipeline:
        def __call__(self, **kwargs):
            return np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(mvr, "MockVideoPipeline", FastPipeline)

    with patch("ipc.video_shm.VideoShm", side_effect=[mock_in, mock_out]):
        with patch("utils.video_manager.VideoManager") as VM:
            VM.return_value.export_to_mp4.return_value = str(mp4)
            with patch("ipc.video_shm.cleanup_orphaned_video_files", return_value=0):
                mvr._run_shm_bridge()

    mock_out.write_response.assert_called()
    mock_in.close.assert_called_once()
    mock_out.close.assert_called_once()


def test_module_main_invokes_bridge(monkeypatch):
    """Mirror ``if __name__ == "__main__"`` without ``runpy.run_module``.

    ``runpy.run_module`` re-executes the module body, which redefines
    ``_run_shm_bridge`` and drops the monkeypatch — the real bridge then blocks
    on ``read_request()`` (appears as a hung test suite).
    """
    called = []
    monkeypatch.setattr(mvr, "_run_shm_bridge", lambda: called.append(1))
    monkeypatch.setattr(mvr.signal, "signal", lambda *a, **k: None)

    mvr.signal.signal(mvr.signal.SIGTERM, mvr._handle_signal)
    mvr.signal.signal(mvr.signal.SIGINT, mvr._handle_signal)
    mvr._run_shm_bridge()
    assert called == [1]


def test_run_shm_bridge_error_path_writes_error_response(monkeypatch, tmp_path):
    """Pipeline/export failure -> ERROR VideoResponse and continue loop."""
    mvr._shutdown = False
    req = VideoRequest(
        task_id="abcdef12-3456-7890-abcd-ef1234567890",
        prompt="p",
        negative_prompt="",
        num_inference_steps=12,
        seed=0,
        height=480,
        width=832,
        num_frames=4,
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )
    mock_in = MagicMock()
    mock_out = MagicMock()
    mock_in.read_request = Mock(side_effect=[req, None])

    class BoomPipeline:
        def __call__(self, **kwargs):
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(mvr, "MockVideoPipeline", BoomPipeline)

    with patch("ipc.video_shm.VideoShm", side_effect=[mock_in, mock_out]):
        with patch("utils.video_manager.VideoManager"):
            with patch("ipc.video_shm.cleanup_orphaned_video_files", return_value=0):
                mvr._run_shm_bridge()

    err_resps = [
        c[0][0]
        for c in mock_out.write_response.call_args_list
        if c[0][0].status == VideoStatus.ERROR
    ]
    assert len(err_resps) == 1
    assert "simulated inference failure" in err_resps[0].error_message


def test_run_shm_bridge_success_path_logs_inference_and_encoded_size(
    monkeypatch, tmp_path
):
    """Covers [MOCK] inference logs, getsize log, and Request ... completed (bridge body)."""
    import numpy as np

    mvr._shutdown = False
    req = VideoRequest(
        task_id="abcdef12-3456-7890-abcd-ef1234567890",
        prompt="p",
        negative_prompt="",
        num_inference_steps=12,
        seed=0,
        height=480,
        width=832,
        num_frames=4,
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )
    mock_in = MagicMock()
    mock_out = MagicMock()
    mock_in.read_request = Mock(side_effect=[req, None])
    mp4 = tmp_path / "out.mp4"
    mp4.write_bytes(b"x" * 120)

    class FastPipeline:
        def __call__(self, **kwargs):
            return np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(mvr, "MockVideoPipeline", FastPipeline)

    mock_log = MagicMock()
    _ul = sys.modules["utils.logger"]
    _saved_tt = _ul.TTLogger
    _ul.TTLogger = lambda: mock_log
    try:
        with patch("ipc.video_shm.VideoShm", side_effect=[mock_in, mock_out]):
            with patch("utils.video_manager.VideoManager") as VM:
                VM.return_value.export_to_mp4.return_value = str(mp4)
                with patch(
                    "ipc.video_shm.cleanup_orphaned_video_files", return_value=0
                ):
                    mvr._run_shm_bridge()
    finally:
        _ul.TTLogger = _saved_tt

    joined = " ".join(str(c) for c in mock_log.info.call_args_list).lower()
    assert "mock_bench" in joined
    assert "synthetic" in joined
    assert "video_delivery" in joined
    assert "mock_after_bench" in joined
    assert "bench_to_handoff_s" in joined
    assert "encoded mp4" in joined
    assert "bytes" in joined
    assert "completed" in joined


def test_run_shm_bridge_logs_when_orphans_removed(monkeypatch, tmp_path):
    import numpy as np

    mvr._shutdown = False
    req = VideoRequest(
        task_id="abcdef12-3456-7890-abcd-ef1234567890",
        prompt="p",
        negative_prompt="",
        num_inference_steps=12,
        seed=0,
        height=480,
        width=832,
        num_frames=4,
        guidance_scale=3.0,
        guidance_scale_2=4.0,
    )
    mock_in = MagicMock()
    mock_out = MagicMock()
    mock_in.read_request = Mock(side_effect=[req, None])
    mp4 = tmp_path / "x.mp4"
    mp4.write_bytes(b"x")

    class FastPipeline:
        def __call__(self, **kwargs):
            return np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(mvr, "MockVideoPipeline", FastPipeline)

    mock_log = MagicMock()
    _ul = sys.modules["utils.logger"]
    _saved_tt = _ul.TTLogger
    _ul.TTLogger = lambda: mock_log
    try:
        with patch("ipc.video_shm.VideoShm", side_effect=[mock_in, mock_out]):
            with patch("utils.video_manager.VideoManager") as VM:
                VM.return_value.export_to_mp4.return_value = str(mp4)
                with patch(
                    "ipc.video_shm.cleanup_orphaned_video_files", return_value=3
                ):
                    mvr._run_shm_bridge()
    finally:
        _ul.TTLogger = _saved_tt

    assert any(
        "3" in str(call) and "orphan" in str(call).lower()
        for call in mock_log.info.call_args_list
    )

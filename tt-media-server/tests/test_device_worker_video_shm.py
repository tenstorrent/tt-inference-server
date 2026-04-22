# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from ipc.video_shm import VideoRequest, VideoResponse, VideoStatus
from tt_model_runners.sp_runner import SPRunner

_mock_settings = MagicMock()
_mock_settings.device_mesh_shape = (1, 1)
_mock_settings.use_dynamic_batcher = False
# Concrete numeric required: ``SPRunner._read_response_for`` does
# ``time.monotonic() + timeout_s`` and compares ``remaining <= 0``; a
# default MagicMock attribute breaks that arithmetic with a TypeError.
_mock_settings.video_request_timeout_seconds = 60.0


@pytest.fixture(autouse=True)
def _patch_base_runner():
    """Ensure BaseDeviceRunner.__init__ uses controlled mocks regardless of import order."""
    with patch(
        "tt_model_runners.base_device_runner.get_settings",
        return_value=_mock_settings,
    ), patch("tt_model_runners.base_device_runner.setup_runner_environment"), patch(
        "tt_model_runners.base_device_runner.TTLogger",
        return_value=MagicMock(),
    ):
        yield


class MockVideoGenerateRequest:
    def __init__(
        self,
        task_id="test-task-id-000000000000000000",
        prompt="a sunset on a beach",
        negative_prompt="blurry",
        num_inference_steps=20,
        seed=42,
    ):
        self._task_id = task_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.seed = seed


def _touch_mp4_file() -> str:
    """Create an empty file; simulates coordinator output (mp4 path in SHM)."""
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="tt_video_test_")
    os.close(fd)
    return path


def _write_video_file(video) -> str:
    """Pickle *video* to a file and return the path (legacy / error-path tests).

    Uses :func:`tempfile.mkstemp` (short-lived OS temp). Production video IPC uses
    RAM-backed paths under ``TT_VIDEO_FILE_DIR`` with a separate TTL policy — not this helper.
    """
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="tt_video_test_")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(video, fh)
    return path


def _install_shm_factory(MockVideoShm):
    """Wire the patched ``VideoShm`` class to return mode-specific mocks.

    Also pins ``queue_depth`` to 0 on the output mock so ``SPRunner.set_device``'s
    startup ``_drain_stale_responses`` is a no-op — otherwise ``range(MagicMock())``
    would iterate and the drain would consume (and unlink) whatever
    ``read_response.return_value`` the test installed for the real call.
    """
    mock_input = MagicMock()
    mock_output = MagicMock()
    mock_output.queue_depth.return_value = 0

    def mock_video_shm_factory(*args, **kwargs):
        if kwargs.get("mode") == "input":
            return mock_input
        return mock_output

    MockVideoShm.side_effect = mock_video_shm_factory
    return mock_input, mock_output


class TestSPRunnerRequestConversion:
    """Test that SPRunner.run() correctly converts requests and sends to SHM."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_basic_conversion(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        file_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")
        assert runner.run([req]) == [file_path]

        written_req = mock_input.write_request.call_args[0][0]
        assert isinstance(written_req, VideoRequest)
        assert written_req.task_id == "tid"
        assert written_req.prompt == "a sunset on a beach"
        assert written_req.negative_prompt == "blurry"
        assert written_req.num_inference_steps == 20
        assert written_req.seed == 42
        assert written_req.height == 480
        assert written_req.width == 832
        assert written_req.num_frames == 81

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_none_negative_prompt_becomes_empty(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        file_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid", negative_prompt=None)
        assert runner.run([req]) == [file_path]

        written_req = mock_input.write_request.call_args[0][0]
        assert written_req.negative_prompt == ""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_none_seed_becomes_zero(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        file_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid", seed=None)
        assert runner.run([req]) == [file_path]

        written_req = mock_input.write_request.call_args[0][0]
        assert written_req.seed == 0


class TestSPRunnerResponseHandling:
    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_success_response_returns_mp4_path(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        file_path = _touch_mp4_file()
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")
        out = runner.run([req])

        assert out == [file_path]
        assert isinstance(out[0], str)
        assert os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_error_response_raises(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.ERROR, "", "inference failed"
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")

        with pytest.raises(RuntimeError, match="error"):
            runner.run([req])

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_timeout_returns_none_raises(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mock_output.read_response.return_value = None

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")

        with pytest.raises(TimeoutError, match="REQUEST_TIMEOUT"):
            runner.run([req])


class TestSPRunnerLifecycle:
    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_close_device_sets_shutdown(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        runner = SPRunner("dev0")
        runner.set_device()
        runner.close_device()

        assert runner._shutdown is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_load_weights_is_noop(self, MockVideoShm):
        runner = SPRunner("dev0")
        assert runner.load_weights() is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_warmup_is_noop_and_does_not_touch_shm(self, MockVideoShm):
        """Warmup is owned by the external ``video_runner`` (which warms the DiT
        before opening SHM), so ``SPRunner.warmup`` must return immediately and
        never write a request or read a response. Driving SHM here would also
        force ``num_inference_steps`` below the ``VideoGenerateRequest``
        validator floor (``ge=12``).
        """
        import asyncio

        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        runner = SPRunner("dev0")
        runner.set_device()

        # set_device drains stale responses at startup; reset before the assert
        # so we can prove warmup itself doesn't touch either ring.
        mock_input.reset_mock()
        mock_output.reset_mock()

        result = asyncio.get_event_loop().run_until_complete(runner.warmup())
        assert result is True

        mock_input.write_request.assert_not_called()
        mock_output.read_response.assert_not_called()

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_timeout_during_read_response(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        mock_output.read_response.return_value = None

        runner = SPRunner("dev0")
        runner.set_device()

        req = MockVideoGenerateRequest(task_id="tid")
        with pytest.raises(TimeoutError, match="REQUEST_TIMEOUT"):
            runner.run([req])


class TestSPRunnerFileCleanup:
    """SPRunner no longer reads/deletes the mp4 on success (job layer owns the file)."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_mp4_not_deleted_after_success(self, MockVideoShm):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        file_path = _touch_mp4_file()
        assert os.path.exists(file_path)

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        assert runner.run([MockVideoGenerateRequest(task_id="tid")]) == [file_path]

        assert os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_success_does_not_open_or_parse_file(self, MockVideoShm):
        """Coordinator sends a path; SPRunner returns it without reading bytes."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        fd, file_path = tempfile.mkstemp(suffix=".mp4", prefix="tt_video_corrupt_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"not-a-valid-mp4")

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()

        out = runner.run([MockVideoGenerateRequest(task_id="tid")])
        assert out == [file_path]
        assert os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_error_response_cleans_up_leftover_file(self, MockVideoShm):
        """If runner wrote a file but then reported ERROR, SPRunner cleans up."""
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        fd, file_path = tempfile.mkstemp(suffix=".pkl", prefix="tt_video_err_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"partial")

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.ERROR, file_path, "pipeline error"
        )

        runner = SPRunner("dev0")
        runner.set_device()

        with pytest.raises(RuntimeError, match="error"):
            runner.run([MockVideoGenerateRequest(task_id="tid")])

        assert not os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.cleanup_orphaned_video_files")
    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_close_device_calls_cleanup(self, MockVideoShm, mock_cleanup):
        mock_input, mock_output = _install_shm_factory(MockVideoShm)

        runner = SPRunner("dev0")
        runner.set_device()
        runner.close_device()

        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

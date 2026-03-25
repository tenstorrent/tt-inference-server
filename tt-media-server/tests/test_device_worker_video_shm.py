# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ipc.video_shm import VideoRequest, VideoResponse, VideoStatus
from tt_model_runners.sp_runner import SPRunner

_mock_settings = MagicMock()
_mock_settings.device_mesh_shape = (1, 1)
_mock_settings.use_dynamic_batcher = False


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


def _write_video_file(video) -> str:
    """Pickle *video* to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="tt_video_test_")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(video, fh)
    return path


class TestSPRunnerRequestConversion:
    """Test that SPRunner.run() correctly converts requests and sends to SHM."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_basic_conversion(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        dummy_video = np.zeros((1, 1, 2, 3, 3), dtype=np.uint8)
        file_path = _write_video_file(dummy_video)
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")
        runner.run([req])

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
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        dummy_video = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
        file_path = _write_video_file(dummy_video)
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid", negative_prompt=None)
        runner.run([req])

        written_req = mock_input.write_request.call_args[0][0]
        assert written_req.negative_prompt == ""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_none_seed_becomes_zero(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        dummy_video = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
        file_path = _write_video_file(dummy_video)
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid", seed=None)
        runner.run([req])

        written_req = mock_input.write_request.call_args[0][0]
        assert written_req.seed == 0


class TestSPRunnerResponseHandling:
    H, W, C = 2, 3, 3

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_success_response_returns_frames(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        num_frames = 3
        expected = np.zeros((1, num_frames, self.H, self.W, self.C), dtype=np.uint8)
        expected[0, 0, :, :, :] = 0x01
        expected[0, 2, :, :, :] = 0x03
        file_path = _write_video_file(expected)
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")
        frames = runner.run([req])

        assert isinstance(frames, np.ndarray)
        assert frames.shape == (1, num_frames, self.H, self.W, self.C)
        assert frames[0, 0, 0, 0, 0] == 0x01
        assert frames[0, 2, 0, 0, 0] == 0x03
        assert not os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_error_response_raises(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

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
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        mock_output.read_response.return_value = None

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")

        with pytest.raises(RuntimeError, match="timed out"):
            runner.run([req])


class TestSPRunnerLifecycle:
    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_close_device_sets_shutdown(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        runner = SPRunner("dev0")
        runner.set_device()
        runner.close_device()

        assert runner._shutdown is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_load_weights_is_noop(self, MockVideoShm):
        runner = SPRunner("dev0")
        assert runner.load_weights() is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_warmup_returns_true(self, MockVideoShm):
        import asyncio

        runner = SPRunner("dev0")
        result = asyncio.get_event_loop().run_until_complete(runner.warmup())
        assert result is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_timeout_during_read_response(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        mock_output.read_response.return_value = None

        runner = SPRunner("dev0")
        runner.set_device()

        req = MockVideoGenerateRequest(task_id="tid")
        with pytest.raises(RuntimeError, match="timed out"):
            runner.run([req])


class TestSPRunnerFileCleanup:
    """Verify SPRunner cleans up video files in success, error, and exception paths."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_file_deleted_after_successful_load(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        video = np.ones((1, 2, 3, 3, 3), dtype=np.uint8)
        file_path = _write_video_file(video)
        assert os.path.exists(file_path)

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        runner.run([MockVideoGenerateRequest(task_id="tid")])

        assert not os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_file_deleted_on_corrupted_pickle(self, MockVideoShm):
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        fd, file_path = tempfile.mkstemp(suffix=".pkl", prefix="tt_video_corrupt_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"not-valid-pickle-data")

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, file_path, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()

        with pytest.raises(Exception):
            runner.run([MockVideoGenerateRequest(task_id="tid")])

        assert not os.path.exists(file_path)

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_error_response_cleans_up_leftover_file(self, MockVideoShm):
        """If runner wrote a file but then reported ERROR, SPRunner cleans up."""
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

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
        mock_input = MagicMock()
        mock_output = MagicMock()

        def mock_video_shm_factory(*args, **kwargs):
            if kwargs.get("mode") == "input":
                return mock_input
            return mock_output

        MockVideoShm.side_effect = mock_video_shm_factory

        runner = SPRunner("dev0")
        runner.set_device()
        runner.close_device()

        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

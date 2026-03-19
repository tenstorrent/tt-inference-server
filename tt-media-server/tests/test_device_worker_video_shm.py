# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from unittest.mock import MagicMock, patch

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

        h, w, c, n = 2, 3, 3, 1
        blob = bytes([0x01] * (n * h * w * c))
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, n, h, w, c, blob, ""
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

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, 1, 2, 2, 3, bytes(12), ""
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

        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, 1, 2, 2, 3, bytes(12), ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid", seed=None)
        runner.run([req])

        written_req = mock_input.write_request.call_args[0][0]
        assert written_req.seed == 0


class TestSPRunnerResponseHandling:
    H, W, C = 2, 3, 3
    FRAME_BYTES = H * W * C

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
        blob = (
            b"\x01" * self.FRAME_BYTES
            + b"\x02" * self.FRAME_BYTES
            + b"\x03" * self.FRAME_BYTES
        )
        mock_output.read_response.return_value = VideoResponse(
            "tid", VideoStatus.SUCCESS, num_frames, self.H, self.W, self.C, blob, ""
        )

        runner = SPRunner("dev0")
        runner.set_device()
        req = MockVideoGenerateRequest(task_id="tid")
        frames = runner.run([req])

        assert frames.shape == (1, 3, self.H, self.W, self.C)
        assert frames[0, 0, 0, 0, 0] == 0x01
        assert frames[0, 2, 0, 0, 0] == 0x03

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
            "tid", VideoStatus.ERROR, 0, 0, 0, 0, b"", "inference failed"
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
        mock_input.unlink.assert_called_once()
        mock_input.close.assert_called_once()
        mock_output.unlink.assert_called_once()
        mock_output.close.assert_called_once()

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

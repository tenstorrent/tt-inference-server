# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import sys
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

sys.modules["ttnn"] = Mock()

mock_settings = Mock()
mock_settings.enable_telemetry = False
mock_settings.max_batch_size = 1
mock_settings.model_runner = "sp_runner"
mock_settings.use_dynamic_batcher = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)
mock_settings.default_throttle_level = ""
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=Mock())

from config.constants import SHUTDOWN_SIGNAL
from ipc.video_shm import VideoRequest, VideoResponse, VideoStatus


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


class WorkerExitException(Exception):
    pass


def _create_get_many_side_effect(return_values):
    iterator = iter(return_values)

    def side_effect(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration:
            raise WorkerExitException("Test complete")

    return side_effect


class TestSPRunnerRequestConversion:
    """Test that SPRunner.run() correctly converts requests and sends to SHM."""

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_basic_conversion(self, MockVideoShm):
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

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


class TestDeviceWorkerVideoShm:
    @pytest.fixture
    def mock_queues(self):
        task_queue = Mock()
        result_queue = Mock()
        warmup_signals_queue = Mock()
        warmup_signals_queue._closed = False
        error_queue = Mock()
        return task_queue, result_queue, warmup_signals_queue, error_queue

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_successful_request_processing(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        test_frames = np.ones((1, 2, 4, 4, 3), dtype=np.uint8)
        mock_runner.run.return_value = test_frames

        req = MockVideoGenerateRequest(task_id="tid")
        task_queue.get_many.side_effect = _create_get_many_side_effect([[req]])

        with pytest.raises(WorkerExitException):
            device_worker_video_shm(
                "w0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        mock_runner.run.assert_called_once_with([req])
        result_queue.put.assert_called_once()
        _, task_id, frames = result_queue.put.call_args[0][0]
        assert task_id == "tid"
        np.testing.assert_array_equal(frames, test_frames)

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_runner_error_reports_to_error_queue(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        mock_runner.run.side_effect = RuntimeError("Runner reported error")

        req = MockVideoGenerateRequest(task_id="tid")
        task_queue.get_many.side_effect = _create_get_many_side_effect([[req]])

        with pytest.raises(WorkerExitException):
            device_worker_video_shm(
                "w0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        error_queue.put.assert_called()
        found_error = any(
            "error" in str(c).lower() for c in error_queue.put.call_args_list
        )
        assert found_error

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_shutdown_signal_exits_cleanly(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[SHUTDOWN_SIGNAL]]
        )

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
        )

        mock_runner.close_device.assert_called_once()
        mock_loop.close.assert_called_once()

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_empty_batch_continues(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        test_frames = np.ones((1, 1, 4, 4, 3), dtype=np.uint8)
        mock_runner.run.return_value = test_frames

        req = MockVideoGenerateRequest(task_id="tid")
        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [None, [], [req]]
        )

        with pytest.raises(WorkerExitException):
            device_worker_video_shm(
                "w0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        assert task_queue.get_many.call_count == 4

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_warmup_signal_sent(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[SHUTDOWN_SIGNAL]]
        )

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
        )

        warmup_signals_queue.put.assert_called_once_with("w0", timeout=2.0)

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_init_failure_reports_error(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        mock_init.side_effect = RuntimeError("device init failed")

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
        )

        error_queue.put.assert_called_once()
        _, _, msg = error_queue.put.call_args[0][0]
        assert "device init failed" in msg

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_init_returns_none_runner_exits(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        mock_init.return_value = (None, MagicMock())

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
        )

        task_queue.get_many.assert_not_called()

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_warmup_queue_closed_does_not_crash(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        warmup_signals_queue._closed = True

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[SHUTDOWN_SIGNAL]]
        )

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
        )

        warmup_signals_queue.put.assert_not_called()

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_warmup_queue_none_does_not_crash(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, _, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[SHUTDOWN_SIGNAL]]
        )

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            None,
            error_queue,
        )

        mock_runner.close_device.assert_called_once()

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_multiple_requests_processed(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        frames_a = np.ones((1, 2, 4, 4, 3), dtype=np.uint8)
        frames_b = np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        mock_runner.run.side_effect = [frames_a, frames_b]

        req_a = MockVideoGenerateRequest(task_id="a")
        req_b = MockVideoGenerateRequest(task_id="b")
        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[req_a], [req_b]]
        )

        with pytest.raises(WorkerExitException):
            device_worker_video_shm(
                "w0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        assert mock_runner.run.call_count == 2
        assert result_queue.put.call_count == 2

    @patch("device_workers.device_worker_video_shm.initialize_device_worker")
    def test_result_queue_name_passed(self, mock_init, mock_queues):
        from device_workers.device_worker_video_shm import device_worker_video_shm

        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_runner = MagicMock()
        mock_loop = MagicMock()
        mock_init.return_value = (mock_runner, mock_loop)

        task_queue.get_many.side_effect = _create_get_many_side_effect(
            [[SHUTDOWN_SIGNAL]]
        )

        device_worker_video_shm(
            "w0",
            task_queue,
            result_queue,
            warmup_signals_queue,
            error_queue,
            result_queue_name="my_queue",
        )

        mock_runner.close_device.assert_called_once()


class TestSPRunnerLifecycle:
    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_close_device_sets_shutdown(self, MockVideoShm):
        from tt_model_runners.sp_runner import SPRunner

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
        from tt_model_runners.sp_runner import SPRunner

        runner = SPRunner("dev0")
        assert runner.load_weights() is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_warmup_returns_true(self, MockVideoShm):
        import asyncio
        from tt_model_runners.sp_runner import SPRunner

        runner = SPRunner("dev0")
        result = asyncio.get_event_loop().run_until_complete(runner.warmup())
        assert result is True

    @patch("tt_model_runners.sp_runner.VideoShm")
    def test_timeout_during_read_response(self, MockVideoShm):
        from tt_model_runners.sp_runner import SPRunner

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

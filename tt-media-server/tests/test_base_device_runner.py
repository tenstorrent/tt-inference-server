# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import unittest
from unittest.mock import MagicMock, patch

from tt_model_runners.base_device_runner import BaseDeviceRunner


class ConcreteDeviceRunner(BaseDeviceRunner):
    """Concrete implementation of BaseDeviceRunner for testing"""

    async def warmup(self):
        return True

    def run(self, *args, **kwargs):
        return []


class TestBaseDeviceRunner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.device_id = "0"
        self.num_torch_threads = 1

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_initialization_success(self, mock_get_settings, mock_set_torch_threads):
        """Test successful initialization of BaseDeviceRunner"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        self.assertEqual(runner.device_id, self.device_id)
        self.assertEqual(runner.settings, mock_settings)
        self.assertIsNone(runner.ttnn_device)
        self.assertFalse(runner.is_tensor_parallel)

        # Verify torch thread limits were set
        mock_set_torch_threads.assert_called_once_with(self.num_torch_threads)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_tensor_parallel_mode_enabled(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test tensor parallel mode is enabled when device_mesh_shape[0] > 1"""
        # Mock settings with tensor parallel enabled
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (2, 4)  # First param > 1
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        self.assertTrue(runner.is_tensor_parallel)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_tensor_parallel_mode_disabled(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test tensor parallel mode is disabled when device_mesh_shape[0] == 1"""
        # Mock settings with tensor parallel disabled
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)  # First param == 1
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        self.assertFalse(runner.is_tensor_parallel)

    @patch.dict("os.environ", {}, clear=True)
    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_hf_token_warning_no_token_no_cache(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test warning is logged when HF_TOKEN is not set and no cached models exist"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        # ✅ Create instance first, then patch its logger
        with patch("tt_model_runners.base_device_runner.TTLogger") as mock_logger_class:
            mock_logger_instance = MagicMock()
            mock_logger_class.return_value = mock_logger_instance

            runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

            # Verify warning was called
            mock_logger_instance.warning.assert_called_once()
            warning_msg = mock_logger_instance.warning.call_args[0][0]
            self.assertIn("HF_TOKEN", warning_msg)

    @patch.dict("os.environ", {"HF_TOKEN": "test_token"}, clear=True)
    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_no_hf_token_warning_when_token_provided(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test no warning is logged when HF_TOKEN is set"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        # ✅ Patch TTLogger to capture logger calls
        with patch("tt_model_runners.base_device_runner.TTLogger") as mock_logger_class:
            mock_logger_instance = MagicMock()
            mock_logger_class.return_value = mock_logger_instance

            runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

            # Warning should not be called for HF_TOKEN
            for call in mock_logger_instance.warning.call_args_list:
                self.assertNotIn("HF_TOKEN", str(call))

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_logger_initialization(self, mock_get_settings, mock_set_torch_threads):
        """Test that logger is properly initialized"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        self.assertIsNotNone(runner.logger)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_load_weights_default_implementation(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test that load_weights has default implementation returning False"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        result = runner.load_weights()
        self.assertFalse(result)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_set_device_default_implementation(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test that set_device has default implementation returning empty dict"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        result = runner.set_device()
        self.assertEqual(result, {})

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_close_device_default_implementation(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test that close_device has default implementation returning True"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        result = runner.close_device()
        self.assertTrue(result)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_is_request_batchable_default_implementation(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test that is_request_batchable has default implementation returning True"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        mock_request = MagicMock()
        result = runner.is_request_batchable(mock_request)
        self.assertTrue(result)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_is_request_batchable_with_batch(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test is_request_batchable with batch parameter"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

        mock_request = MagicMock()
        mock_batch = MagicMock()
        result = runner.is_request_batchable(mock_request, mock_batch)
        self.assertTrue(result)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_multiple_torch_threads(self, mock_get_settings, mock_set_torch_threads):
        """Test initialization with multiple torch threads"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        num_threads = 4
        runner = ConcreteDeviceRunner(self.device_id, num_threads)

        mock_set_torch_threads.assert_called_once_with(num_threads)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_device_id_stored_correctly(
        self, mock_get_settings, mock_set_torch_threads
    ):
        """Test that device_id is stored correctly"""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (1, 8)
        mock_get_settings.return_value = mock_settings

        test_device_ids = ["0", "1", "device_0", "(0,0)"]

        for device_id in test_device_ids:
            runner = ConcreteDeviceRunner(device_id, self.num_torch_threads)
            self.assertEqual(runner.device_id, device_id)

    @patch("tt_model_runners.base_device_runner.set_torch_thread_limits")
    @patch("tt_model_runners.base_device_runner.get_settings")
    def test_tensor_parallel_logging(self, mock_get_settings, mock_set_torch_threads):
        """Test that tensor parallel mode is logged correctly"""
        # Mock settings with tensor parallel enabled
        mock_settings = MagicMock()
        mock_settings.device_mesh_shape = (2, 4)
        mock_get_settings.return_value = mock_settings

        # ✅ Patch TTLogger to capture logger calls
        with patch("tt_model_runners.base_device_runner.TTLogger") as mock_logger_class:
            mock_logger_instance = MagicMock()
            mock_logger_class.return_value = mock_logger_instance

            runner = ConcreteDeviceRunner(self.device_id, self.num_torch_threads)

            # Verify info was called with tensor parallel message
            mock_logger_instance.info.assert_called_once()
            log_msg = mock_logger_instance.info.call_args[0][0]
            self.assertIn("Tensor parallel mode enabled", log_msg)
            self.assertIn("2", log_msg)  # mesh shape first param


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)

    unittest.main()

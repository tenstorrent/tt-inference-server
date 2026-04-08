# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import unittest
from unittest.mock import MagicMock, call, patch

from huggingface_hub.utils import LocalEntryNotFoundError
from utils.hugging_face_utils import HuggingFaceUtils


class TestHuggingFaceUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.utils = None

    @patch("utils.hugging_face_utils.TTLogger")
    def test_initialization(self, mock_logger_class):
        """Test HuggingFaceUtils initialization"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        utils = HuggingFaceUtils()

        self.assertIsNotNone(utils.logger)
        mock_logger_class.assert_called_once()

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    def test_download_weights_pipeline_loaded_successfully(
        self, mock_get_device_runner, mock_logger_class
    ):
        """Test download_weights when pipeline weights are loaded successfully"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = True
        mock_get_device_runner.return_value = mock_device_runner

        utils = HuggingFaceUtils()
        utils.download_weights()

        # Verify load_weights was called
        mock_device_runner.load_weights.assert_called_once()

        # Verify no further downloads happened
        mock_logger.info.assert_not_called()

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    def test_download_weights_no_model_specified(
        self, mock_settings, mock_get_device_runner, mock_logger_class
    ):
        """Test download_weights when no model_weights_path is specified"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = None

        utils = HuggingFaceUtils()
        utils.download_weights()

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("No model_weights_path specified", warning_msg)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    def test_download_weights_model_exists_locally(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights when model already exists locally"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "/path/to/model"

        # Mock os.path.exists to return True
        mock_path_exists.return_value = True

        utils = HuggingFaceUtils()
        utils.download_weights()

        # Verify model exists message was logged
        mock_logger.info.assert_called_once()
        info_msg = mock_logger.info.call_args[0][0]
        self.assertIn("Model already exists locally", info_msg)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    def test_download_weights_model_cached(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights when model is already cached"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "meta-llama/Llama-2-7b"

        # Mock os.path.exists to return False (not local)
        mock_path_exists.return_value = False

        utils = HuggingFaceUtils()

        # Mock _are_huggingface_weights_cached to return True
        with patch.object(utils, "_are_huggingface_weights_cached", return_value=True):
            with patch("utils.hugging_face_utils.snapshot_download") as mock_download:
                mock_download.return_value = "/cache/model"

                utils.download_weights()

                # Verify cached message was logged
                self.assertIn(
                    call(
                        "Model meta-llama/Llama-2-7b already cached, skipping download"
                    ),
                    mock_logger.info.call_args_list,
                )

                # Verify snapshot_download was called with local_files_only=True
                mock_download.assert_called_once()
                call_kwargs = mock_download.call_args[1]
                self.assertTrue(call_kwargs.get("local_files_only"))

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    def test_download_weights_model_not_cached_downloads(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights downloads model when not cached"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "meta-llama/Llama-2-7b"

        # Mock os.path.exists to return False
        mock_path_exists.return_value = False

        utils = HuggingFaceUtils()

        # Mock _are_huggingface_weights_cached to return False
        with patch.object(utils, "_are_huggingface_weights_cached", return_value=False):
            with patch("utils.hugging_face_utils.snapshot_download") as mock_download:
                mock_download.return_value = "/downloaded/model"

                utils.download_weights()

                # Verify downloading message was logged
                self.assertIn(
                    call("Downloading weights for model: meta-llama/Llama-2-7b"),
                    mock_logger.info.call_args_list,
                )

                # Verify snapshot_download was called
                mock_download.assert_called_once()

                # Verify success message was logged
                self.assertIn(
                    call("Successfully downloaded model weights to: /downloaded/model"),
                    mock_logger.info.call_args_list,
                )

                # Verify settings was updated
                self.assertEqual(mock_settings.model_weights_path, "/downloaded/model")

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    def test_download_weights_handles_import_error(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights handles ImportError gracefully"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "meta-llama/Llama-2-7b"
        mock_path_exists.return_value = False

        utils = HuggingFaceUtils()

        # Mock snapshot_download to raise ImportError
        with patch("utils.hugging_face_utils.snapshot_download") as mock_download:
            mock_download.side_effect = ImportError("huggingface_hub not installed")

            with self.assertRaises(RuntimeError) as context:
                utils.download_weights()

            self.assertIn("Missing required dependency", str(context.exception))
            mock_logger.error.assert_called_once()

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    def test_download_weights_handles_general_exception(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights handles general exceptions gracefully"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "meta-llama/Llama-2-7b"
        mock_path_exists.return_value = False

        utils = HuggingFaceUtils()

        # Mock snapshot_download to raise exception
        with patch("utils.hugging_face_utils.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Network error")

            # Should not raise, just log and continue
            utils.download_weights()

            # Verify error was logged
            self.assertEqual(mock_logger.error.call_count, 1)

            # Verify warning was logged
            self.assertEqual(mock_logger.warning.call_count, 1)
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Continuing with existing model weights path", warning_msg)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.try_to_load_from_cache")
    def test_are_huggingface_weights_cached_found(
        self, mock_try_load, mock_logger_class
    ):
        """Test _are_huggingface_weights_cached returns True when file is found"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        mock_try_load.return_value = "/cache/path/config.json"

        utils = HuggingFaceUtils()
        result = utils._are_huggingface_weights_cached("meta-llama/Llama-2-7b")

        self.assertTrue(result)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.try_to_load_from_cache")
    def test_are_huggingface_weights_cached_not_found(
        self, mock_try_load, mock_logger_class
    ):
        """Test _are_huggingface_weights_cached returns False when file is not found"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        mock_try_load.side_effect = LocalEntryNotFoundError("Not found")

        utils = HuggingFaceUtils()
        result = utils._are_huggingface_weights_cached("meta-llama/Llama-2-7b")

        self.assertFalse(result)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.try_to_load_from_cache")
    def test_are_huggingface_weights_cached_import_error(
        self, mock_try_load, mock_logger_class
    ):
        """Test _are_huggingface_weights_cached handles ImportError gracefully"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        mock_try_load.side_effect = ImportError("Module not found")

        utils = HuggingFaceUtils()
        result = utils._are_huggingface_weights_cached("meta-llama/Llama-2-7b")

        self.assertFalse(result)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.try_to_load_from_cache")
    def test_are_huggingface_weights_cached_general_exception(
        self, mock_try_load, mock_logger_class
    ):
        """Test _are_huggingface_weights_cached handles general exceptions gracefully"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        mock_try_load.side_effect = Exception("Unknown error")

        utils = HuggingFaceUtils()
        result = utils._are_huggingface_weights_cached("meta-llama/Llama-2-7b")

        self.assertFalse(result)

    @patch("utils.hugging_face_utils.TTLogger")
    @patch("utils.hugging_face_utils.get_device_runner")
    @patch("utils.hugging_face_utils.settings")
    @patch("utils.hugging_face_utils.os.path.exists")
    @patch.dict("os.environ", {"HF_HOME": "/custom/hf/home"})
    def test_download_weights_uses_custom_hf_home(
        self,
        mock_path_exists,
        mock_settings,
        mock_get_device_runner,
        mock_logger_class,
    ):
        """Test download_weights uses custom HF_HOME environment variable"""
        # Mock logger
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger

        # Mock device runner
        mock_device_runner = MagicMock()
        mock_device_runner.load_weights.return_value = False
        mock_get_device_runner.return_value = mock_device_runner

        # Mock settings
        mock_settings.model_weights_path = "meta-llama/Llama-2-7b"
        mock_path_exists.return_value = False

        utils = HuggingFaceUtils()

        with patch.object(utils, "_are_huggingface_weights_cached", return_value=False):
            with patch("utils.hugging_face_utils.snapshot_download") as mock_download:
                mock_download.return_value = "/downloaded/model"

                utils.download_weights()

                # Verify HF_HOME was used
                call_kwargs = mock_download.call_args[1]
                self.assertEqual(call_kwargs.get("cache_dir"), "/custom/hf/home")


if __name__ == "__main__":
    unittest.main()

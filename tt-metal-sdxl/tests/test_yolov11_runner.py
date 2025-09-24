# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock ttnn before importing the runner to avoid import errors
sys.modules["ttnn"] = MagicMock()

from tt_model_runners.yolov11_runner import TTYolov11Runner


class TestTTYolov11Runner:
    @pytest.fixture
    def runner(self):
        with patch("tt_model_runners.yolov11_runner.ttnn"):
            runner = TTYolov11Runner("test_device")
            return runner

    def test_init(self, runner):
        assert runner.device_id == "test_device"
        assert runner.tt_device is None
        assert runner.model is None
        assert runner.resolution == (640, 640)  # YOLOv11 uses 640x640
        assert runner.batch_size == 1

    def test_format_detections(self, runner):
        runner.class_names = ["person", "bicycle", "car"]

        # Test with valid detections (need 7 elements: x1, y1, x2, y2, confidence, _, class_id)
        detections = [
            [0.1, 0.2, 0.3, 0.4, 0.95, 0, 0],  # person with high confidence
            [0.5, 0.6, 0.7, 0.8, 0.85, 0, 2],  # car with good confidence
        ]

        formatted = runner._format_detections(detections)

        assert len(formatted) == 2
        assert formatted[0]["class_name"] == "person"
        assert formatted[0]["confidence"] == 0.95
        assert formatted[0]["class_id"] == 0
        assert formatted[0]["bbox"]["x1"] == 0.1

        assert formatted[1]["class_name"] == "car"
        assert formatted[1]["class_id"] == 2

    def test_format_detections_empty(self, runner):
        formatted = runner._format_detections([])
        assert formatted == []

    def test_format_detections_invalid(self, runner):
        runner.class_names = ["person", "bicycle", "car"]

        # Test with invalid detection (too few elements)
        detections = [
            [0.1, 0.2, 0.3, 0.4],  # missing confidence and class_id
        ]

        formatted = runner._format_detections(detections)
        assert formatted == []

    def test_format_detections_unknown_class(self, runner):
        runner.class_names = ["person", "bicycle", "car"]

        # Test with unknown class ID
        detections = [
            [0.1, 0.2, 0.3, 0.4, 0.95, 0, 99],  # class_id 99 doesn't exist
        ]

        formatted = runner._format_detections(detections)

        assert len(formatted) == 1
        assert formatted[0]["class_name"] == "class_99"  # fallback name
        assert formatted[0]["class_id"] == 99

    def test_device_params(self, runner):
        with patch("tt_model_runners.yolov11_runner.YOLOV11_L1_SMALL_SIZE", 24576):
            params = runner._get_device_params()
            
            assert params["l1_small_size"] == 24576
            assert params["trace_region_size"] == 6434816
            assert params["num_command_queues"] == 2

    def test_create_mesh_shape(self, runner):
        # Test single device
        shape = runner._create_mesh_shape([0])
        assert shape == (1, 1)

        # Test dual device
        shape = runner._create_mesh_shape([0, 1])
        assert shape == (1, 2)

        # Test quad device
        shape = runner._create_mesh_shape([0, 1, 2, 3])
        assert shape == (2, 2)

        # Test 8 devices
        shape = runner._create_mesh_shape([0, 1, 2, 3, 4, 5, 6, 7])
        assert shape == (2, 4)

    @patch("tt_model_runners.yolov11_runner.Path")
    def test_create_model_location_generator(self, mock_path, runner):
        mock_tt_metal_home = MagicMock()
        mock_path.return_value = mock_tt_metal_home
        
        generator = runner._create_model_location_generator(mock_tt_metal_home)
        
        # Test without subdir
        result = generator("model_v1")
        expected_path = mock_tt_metal_home / "models" / "demos" / "yolov11" / "model_v1"
        assert str(result) == str(expected_path)
        
        # Test with subdir
        result = generator("model_v1", "weights")
        expected_path = mock_tt_metal_home / "models" / "demos" / "yolov11" / "weights" / "model_v1"
        assert str(result) == str(expected_path)

    def test_log_device_configuration(self, runner):
        with patch("tt_model_runners.yolov11_runner.settings") as mock_settings:
            mock_settings.max_batch_size = 4
            
            # Should log warning about batch size being forced to 1
            with patch.object(runner.logger, 'warning') as mock_warning:
                runner._log_device_configuration()
                mock_warning.assert_called_once()
                assert "Batch size forced to 1 for YOLOv11" in mock_warning.call_args[0][0]

    def test_log_device_configuration_single_device(self, runner):
        runner.use_single_device = True
        
        with patch.object(runner.logger, 'info') as mock_info:
            runner._log_device_configuration()
            mock_info.assert_called_once()
            assert "YOLOv11 using single device operation" in mock_info.call_args[0][0]

    def test_log_device_configuration_multi_device(self, runner):
        runner.use_single_device = False
        
        with patch.object(runner.logger, 'warning') as mock_warning:
            runner._log_device_configuration()
            mock_warning.assert_called()
            # Should have warning about multi-device operation
            warning_calls = [call[0][0] for call in mock_warning.call_args_list]
            assert any("multi-device operation" in call for call in warning_calls)

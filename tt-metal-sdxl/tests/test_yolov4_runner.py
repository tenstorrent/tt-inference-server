# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock ttnn before importing the runner to avoid import errors
sys.modules["ttnn"] = MagicMock()

from tt_model_runners.yolov4_runner import TTYolov4Runner


class TestTTYolov4Runner:
    @pytest.fixture
    def runner(self):
        with patch("tt_model_runners.yolov4_runner.ttnn"):
            runner = TTYolov4Runner("test_device")
            return runner

    def test_init(self, runner):
        assert runner.device_id == "test_device"
        assert runner.tt_device is None
        assert runner.model is None
        assert runner.resolution == (320, 320)
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

    @patch("tt_model_runners.yolov4_runner.ttnn")
    async def test_load_model_fallback(self, mock_ttnn, runner):
        # Mock device
        mock_device = MagicMock()
        mock_ttnn.get_device_ids.return_value = [0]
        mock_ttnn.open_mesh_device.return_value = mock_device
        mock_device.get_num_devices.return_value = 1

        # Test when performant runner is not available
        with patch("sys.path", []):
            with pytest.raises(ImportError):
                await runner.load_model(None)

    def test_run_inference_no_model(self, runner):
        # Test error when model not loaded
        # Create a valid base64 encoded image with proper data URI format
        import base64
        from PIL import Image
        from io import BytesIO

        # Create a simple test image and encode it properly with data URI format
        img = Image.new("RGB", (320, 320), color="red")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_data = base64.b64encode(buffered.getvalue()).decode()
        # Use proper data URI format that ImageManager expects
        valid_base64 = f"data:image/png;base64,{base64_data}"

        # Should fail when trying to call self.model.run() since model is None
        from tt_model_runners.yolov4_runner import InferenceError

        with pytest.raises(InferenceError) as exc_info:
            runner.run_inference(valid_base64)

        # The error should eventually be about AttributeError when trying to call model.run()
        # But it might fail earlier, so we just check that InferenceError is raised
        assert "Inference failed" in str(exc_info.value)

    @patch("tt_model_runners.yolov4_runner.ttnn")
    def test_close_device(self, mock_ttnn, runner):
        mock_device = MagicMock()
        runner.tt_device = mock_device
        mock_device.get_submeshes.return_value = [MagicMock(), MagicMock()]

        result = runner.close_device(None)

        assert result is True
        assert mock_ttnn.close_mesh_device.call_count >= 1

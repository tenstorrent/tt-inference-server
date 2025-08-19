# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys

# Mock ttnn before importing the runner to avoid import errors
sys.modules['ttnn'] = MagicMock()

import torch
import numpy as np
from PIL import Image
from io import BytesIO

from tt_model_runners.yolov4_runner import TTYolov4Runner


class TestTTYolov4Runner:
    
    @pytest.fixture
    def runner(self):
        with patch('tt_model_runners.yolov4_runner.ttnn'):
            runner = TTYolov4Runner("test_device")
            return runner
    
    @pytest.fixture
    def sample_image_base64(self):
        # Create a simple test image
        img = Image.new('RGB', (320, 320), color='red')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def test_init(self, runner):
        assert runner.device_id == "test_device"
        assert runner.tt_device is None
        assert runner.model is None
        assert runner.resolution == (320, 320)
        assert runner.batch_size == 1
    
    def test_base64_to_pil_image(self, runner, sample_image_base64):
        # Test normal base64 string
        img = runner._base64_to_pil_image(sample_image_base64, target_size=(320, 320), target_mode="RGB")
        assert isinstance(img, Image.Image)
        assert img.size == (320, 320)
        assert img.mode == "RGB"
        
        # Test with data: prefix
        prefixed_base64 = f"data:image/png;base64,{sample_image_base64}"
        img = runner._base64_to_pil_image(prefixed_base64, target_size=(640, 640), target_mode="RGB")
        assert img.size == (640, 640)
    
    def test_get_default_coco_names(self, runner):
        names = runner._get_default_coco_names()
        assert len(names) == 80  # COCO has 80 classes
        assert names[0] == "person"
        assert names[1] == "bicycle"
        assert names[-1] == "toothbrush"
    
    def test_format_detections(self, runner):
        runner.class_names = ["person", "bicycle", "car"]
        
        # Test with valid detections
        detections = [
            [0.1, 0.2, 0.3, 0.4, 0.95, 0],  # person with high confidence
            [0.5, 0.6, 0.7, 0.8, 0.85, 2],  # car with good confidence
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
    
    def test_prepare_image_tensor(self, runner, sample_image_base64):
        tensor = runner._prepare_image_tensor(sample_image_base64)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 320, 320)  # NCHW format
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
    
    def test_nms_cpu(self, runner):
        # Test non-maximum suppression
        boxes = np.array([
            [0.1, 0.1, 0.3, 0.3],
            [0.15, 0.15, 0.35, 0.35],  # Overlapping with first
            [0.5, 0.5, 0.7, 0.7],  # Separate box
        ])
        confs = np.array([0.9, 0.8, 0.95])
        
        keep = runner._nms_cpu(boxes, confs, nms_thresh=0.5)
        
        # Should keep the highest confidence boxes
        assert len(keep) > 0
        assert 2 in keep  # Highest confidence
        assert 0 in keep  # High confidence, different location
    
    @patch('tt_model_runners.yolov4_runner.ttnn')
    async def test_load_model_fallback(self, mock_ttnn, runner):
        # Mock device
        mock_device = MagicMock()
        mock_ttnn.get_device_ids.return_value = [0]
        mock_ttnn.open_mesh_device.return_value = mock_device
        mock_device.get_num_devices.return_value = 1
        
        # Test when performant runner is not available
        with patch('sys.path', []):
            with pytest.raises(ImportError):
                await runner.load_model(None)
    
    def test_run_inference_no_model(self, runner, sample_image_base64):
        # Test error when model not loaded
        with pytest.raises(RuntimeError) as exc_info:
            runner.run_inference(sample_image_base64)
        assert "Model not loaded" in str(exc_info.value)
    
    @patch('tt_model_runners.yolov4_runner.ttnn')
    def test_close_device(self, mock_ttnn, runner):
        mock_device = MagicMock()
        runner.mesh_device = mock_device
        mock_device.get_submeshes.return_value = [MagicMock(), MagicMock()]
        
        result = runner.close_device(None)
        
        assert result is True
        assert mock_ttnn.close_mesh_device.call_count >= 1
    
    def test_post_processing_mock(self, runner):
        # Create mock output
        mock_boxes = torch.randn(1, 6356, 1, 4)
        mock_confs = torch.randn(1, 6356, 80)
        
        output = [mock_boxes, mock_confs]
        
        # Run post-processing
        result = runner._post_processing(
            torch.randn(1, 3, 320, 320),
            conf_thresh=0.5,
            nms_thresh=0.4,
            output=output
        )
        
        assert isinstance(result, list)
        assert len(result) == 1  # One batch

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
import ttnn

from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.logger import TTLogger
from utils.image_manager import ImageManager
from domain.image_search_request import ImageSearchRequest

# make sure tests/__init__.py exists
TT_METAL_HOME = Path(os.environ["TT_METAL_HOME"])
tests_init = TT_METAL_HOME / "tests" / "__init__.py"
tests_init.touch()
# make sure tt-metal/tests is imported ahead of server/tests
sys.path.insert(0, str(TT_METAL_HOME))

# Import YOLOv6L-specific modules
from models.demos.yolov6l.runner.performant_runner import YOLOv6lPerformantRunner
from models.demos.yolov6l.common import YOLOV6L_L1_SMALL_SIZE
from models.demos.yolov6l.tt.common import get_mesh_mappers
from models.demos.yolov6l.demo.demo_utils import postprocess
from models.demos.utils.common_demo_utils import load_coco_class_names

# Constants
DEFAULT_RESOLUTION = (640, 640)  # YOLOv6L uses 640x640
DEFAULT_TRACE_REGION_SIZE = 6434816
DEFAULT_NUM_COMMAND_QUEUES = 2
WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS = 120
DEFAULT_CONFIDENCE_THRESHOLD = 0.01
DEFAULT_NMS_THRESHOLD = 0.6
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 60


class YoloV6LModelError(Exception):
    """Base exception for YOLOv6L model errors"""
    pass


class InferenceError(YoloV6LModelError):
    """Error occurred during model inference"""
    pass


class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds timeout limit"""
    pass


class TTYolov6LRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.tt_device = None
        self.model = None
        self.class_names: List[str] = []
        self.resolution = DEFAULT_RESOLUTION
        self.batch_size = 1  # YOLOv6L requires batch_size=1 for memory stability
        self.use_single_device = getattr(settings, "yolov6l_use_single_device", True)
        self.image_manager = ImageManager(storage_dir="")

        self._log_device_configuration()

    def _log_device_configuration(self):
        """Log device configuration warnings."""
        configured_batch_size = settings.max_batch_size
        if configured_batch_size > 1:
            self.logger.warning(
                f"Batch size forced to 1 for YOLOv6L (configured: {configured_batch_size})"
            )

        if self.use_single_device:
            self.logger.info("YOLOv6L using single device operation")
        else:
            self.logger.warning(
                "YOLOv6L using multi-device operation (may cause memory issues)"
            )

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def get_device(self):
        return self._mesh_device()

    def _get_device_params(self):
        """Get YOLOv6L-specific device parameters."""
        return {
            "l1_small_size": YOLOV6L_L1_SMALL_SIZE,
            "trace_region_size": DEFAULT_TRACE_REGION_SIZE,
            "num_command_queues": DEFAULT_NUM_COMMAND_QUEUES,
        }

    def _create_mesh_shape(self, device_ids):
        """Create appropriate mesh shape based on device configuration."""
        if self.use_single_device:
            return ttnn.MeshShape(1, 1)

        param = len(device_ids)
        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, (
                "Device mesh grid shape should have exactly two elements."
            )
            num_devices_requested = grid_dims[0] * grid_dims[1]
            assert num_devices_requested <= len(device_ids), (
                "Requested more devices than available."
            )
            return ttnn.MeshShape(*grid_dims)
        else:
            num_devices_requested = min(param, len(device_ids))
            return ttnn.MeshShape(1, num_devices_requested)

    def _mesh_device(self):
        """Create mesh device with YOLOv6L-optimized configuration."""
        device_params = self._get_device_params()
        device_ids = ttnn.get_device_ids()
        mesh_shape = self._create_mesh_shape(device_ids)

        updated_device_params = self.get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)

        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(
            mesh_shape=mesh_shape, **updated_device_params
        )

        device_count = mesh_device.get_num_devices()
        device_text = "device" if device_count == 1 else "devices"
        self.logger.info(
            f"Created mesh device with {device_count} {device_text} for YOLOv6L"
        )
        return mesh_device

    def close_device(self, device=None) -> bool:
        """Close mesh device and submeshes."""
        target_device = device or self.tt_device
        if target_device is None:
            return True

        try:
            for submesh in target_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
        except Exception:
            pass

        ttnn.close_mesh_device(target_device)
        return True

    def _create_model_location_generator(self, tt_metal_home: Path):
        """Create model location generator for YOLOv6L weights."""

        def model_location_generator(
            rel_path, model_subdir="", download_if_ci_v2=False
        ):
            if os.environ.get("MODEL_WEIGHTS_PATH"):
                weights_dir = Path(os.environ["MODEL_WEIGHTS_PATH"])
                self.logger.info(f"Using MODEL_WEIGHTS_PATH: {weights_dir}")
                assert weights_dir.exists(), (
                    f"MODEL_WEIGHTS_PATH: {weights_dir} does not exist"
                )
            else:
                weights_dir = (
                    tt_metal_home / "models" / "demos" / "yolov6l" / "tests" / "pcc"
                )
                self.logger.info(f"Using default weights directory: {weights_dir}")
            return weights_dir  # Return Path object

        return model_location_generator

    def _distribute_model(self) -> None:
        """Initialize and distribute YOLOv6L model on device."""
        try:
            tt_metal_home = Path(os.environ["TT_METAL_HOME"])
            os.environ["TT_GH_CI_INFRA"] = "1"

            if str(tt_metal_home) not in sys.path:
                sys.path.insert(0, str(tt_metal_home))

            
            reference_dir = tt_metal_home / "models" / "demos" / "yolov6l" / "reference"
            if str(reference_dir) not in sys.path:
                sys.path.insert(0, str(reference_dir))
                self.logger.info(f"Added yolov6l reference directory to sys.path: {reference_dir}")

            inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(
                self.tt_device
            )
            model_location_generator = self._create_model_location_generator(
                tt_metal_home
            )

            self.model = YOLOv6lPerformantRunner(
                self.tt_device,
                device_batch_size=self.batch_size,
                act_dtype=ttnn.bfloat16,
                weight_dtype=ttnn.bfloat16,
                resolution=self.resolution,
                model_location_generator=model_location_generator,
                mesh_mapper=inputs_mesh_mapper,
                weights_mesh_mapper=weights_mesh_mapper,
                mesh_composer=output_mesh_composer,
            )
            self.logger.info("YOLOv6lPerformantRunner initialized successfully")
        except Exception as e:
            self.logger.error(f"Model distribution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to initialize YOLOv6lPerformantRunner") from e

    def _validate_tt_metal_environment(self):
        """Validate TT_METAL_HOME environment setup."""
        tt_metal_home_str = os.environ.get("TT_METAL_HOME")
        if not tt_metal_home_str:
            raise RuntimeError("TT_METAL_HOME environment variable not set")

        tt_metal_home = Path(tt_metal_home_str)
        if not tt_metal_home.exists():
            raise RuntimeError(f"tt-metal home not found at {tt_metal_home}")

        if str(tt_metal_home) not in sys.path:
            sys.path.insert(0, str(tt_metal_home))

        return tt_metal_home

    async def load_model(self, device) -> bool:
        """Load YOLOv6L model with weights and distribute on device."""
        self.logger.info("Loading YOLOv6L model...")

        self.tt_device = device if device is not None else self.get_device()
        self._validate_tt_metal_environment()

        # Load components
        self.class_names = self._load_class_names()
        self.logger.info(f"Loaded {len(self.class_names)} classes and model weights")

        # Distribute with timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._distribute_model),
                timeout=WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Model distribution timed out after {WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS}s"
            )

        # Warmup
        dummy_image = torch.zeros(self.batch_size, 3, *self.resolution)
        import time
        warmup_start = time.time()
        self.model.run(dummy_image)
        warmup_time = (time.time() - warmup_start) * 1000  # Convert to ms
        self.logger.info(f"YOLOv6L model loaded and warmed up successfully (warmup latency: {warmup_time:.2f}ms)")
        return True

    def _preprocess_image_data(self, image_data_list) -> List[str]:
        """Convert input data to list of base64 strings."""
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]

        processed_data = []
        for item in image_data_list:
            if isinstance(item, ImageSearchRequest):
                processed_data.append(item.prompt)
            elif isinstance(item, str):
                processed_data.append(item)
            else:
                raise ValueError(
                    f"Unsupported type: {type(item)}. Expected str or ImageSearchRequest"
                )
        return processed_data

    def _check_timeout(self, start_time: float, timeout_seconds: int, stage: str):
        """Check if operation has exceeded timeout."""
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise InferenceTimeoutError(f"Timeout after {elapsed:.2f}s during {stage}")

    def run_inference(
        self,
        image_data_list,
        num_inference_steps: int = None,
        timeout_seconds: int = None,
    ):
        """Run inference on image data with timeout handling."""
        timeout_seconds = timeout_seconds or DEFAULT_INFERENCE_TIMEOUT_SECONDS
        image_data_list = self._preprocess_image_data(image_data_list)

        start_time = time.time()
        results = []

        # Process images in batches
        for batch_start in range(0, len(image_data_list), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_data_list))
            batch_images = image_data_list[batch_start:batch_end]
            batch_num = batch_start // self.batch_size + 1

            try:
                self._check_timeout(
                    start_time, timeout_seconds, f"batch {batch_num} preparation"
                )

                # Prepare and run inference
                batch_tensor = self._prepare_batch_tensor(
                    batch_images, len(batch_images)
                )
                remaining_time = max(0, timeout_seconds - (time.time() - start_time))

                if remaining_time <= 0:
                    raise InferenceTimeoutError(
                        f"No time remaining for batch {batch_num}"
                    )

                raw_output = self._run_batch_inference(
                    batch_tensor, remaining_time, batch_num
                )

                self._check_timeout(
                    start_time, timeout_seconds, f"batch {batch_num} inference"
                )

                # Post-process results - YOLOv6L uses different postprocessing
                boxes_batch = self._post_process_yolov6l(
                    raw_output, batch_tensor, batch_images
                )

                self._check_timeout(
                    start_time, timeout_seconds, f"batch {batch_num} post-processing"
                )

                # Format detections
                for boxes in boxes_batch:
                    results.append(self._format_detections(boxes))

            except InferenceTimeoutError:
                raise
            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}")
                raise InferenceError(
                    f"Inference failed on batch {batch_num}: {str(e)}"
                ) from e

        total_time = time.time() - start_time
        num_batches = (len(image_data_list) + self.batch_size - 1) // self.batch_size
        self.logger.info(
            f"Completed {len(image_data_list)} images in {num_batches} batches ({total_time:.3f}s)"
        )
        return results

    def _post_process_yolov6l(self, raw_output, batch_tensor, batch_images):
        """Post-process YOLOv6L output to match expected format."""
        # Convert tt-metal output to torch tensor
        _, _, output_mesh_composer = get_mesh_mappers(self.tt_device)
        preds = ttnn.to_torch(raw_output, dtype=torch.float32, mesh_composer=output_mesh_composer)
        
        # Get original images for postprocessing
        # Convert PIL Images to numpy arrays (matching LoadImages format)
        orig_images = []
        for img_base64 in batch_images:
            orig_img_pil = self.image_manager.base64_to_pil_image(img_base64)
            # Convert PIL Image to numpy array (H, W, C) format
            orig_img = np.array(orig_img_pil)
            orig_images.append(orig_img)
        
        # Use YOLOv6L postprocess function
        # batch[0] should be the list of image paths/base64 strings
        batch_results = postprocess(
            preds,
            batch_tensor,
            orig_images,
            [batch_images],  # batch format: list where first element is list of paths
            self.class_names,
            conf=DEFAULT_CONFIDENCE_THRESHOLD,
            max_det=300
        )
        
        # Convert Results objects to list format compatible with _format_detections
        boxes_batch = []
        for result in batch_results:
            boxes = []
            # result is a dict: {"orig_img": ..., "path": ..., "names": ..., "boxes": {...}}
            # boxes is a dict: {"xyxy": tensor, "conf": tensor, "cls": tensor}
            if isinstance(result, dict) and "boxes" in result and result["boxes"] is not None:
                boxes_dict = result["boxes"]
                xyxy = boxes_dict["xyxy"]  # [N, 4] tensor: x1, y1, x2, y2
                conf = boxes_dict["conf"]   # [N] tensor: confidence
                cls = boxes_dict["cls"]     # [N] tensor: class_id
                
                # Convert to list format [x1, y1, x2, y2, conf, conf, class_id]
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].tolist()
                    boxes.append([x1, y1, x2, y2, float(conf[i].item()), float(conf[i].item()), int(cls[i].item())])
            boxes_batch.append(boxes)
        
        return boxes_batch

    def _run_batch_inference(
        self, batch_tensor: torch.Tensor, timeout: float, batch_num: int
    ):
        """Run inference on batch with timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.run, batch_tensor)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise InferenceTimeoutError(f"Batch {batch_num} inference timeout")

    def _prepare_batch_tensor(
        self, batch_images: List[str], current_batch_size: int
    ) -> torch.Tensor:
        """Prepare batch tensor from base64 image strings."""
        batch_tensors = []

        for image_base64 in batch_images:
            image_tensor = self.image_manager.prepare_image_tensor(
                image_base64, target_size=self.resolution, target_mode="RGB"
            ).squeeze(0)  # Remove batch dimension
            batch_tensors.append(image_tensor)

        batch_tensor = torch.stack(batch_tensors, dim=0)

        # Pad if needed
        if current_batch_size < self.batch_size:
            padding_shape = (self.batch_size - current_batch_size, 3, *self.resolution)
            padding = torch.zeros(padding_shape)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)

        return batch_tensor

    def _load_class_names(self) -> List[str]:
        """Load COCO class names from file."""
        names_path = Path(__file__).resolve().parent / "resources" / "coco.names"

        if not names_path.exists():
            raise FileNotFoundError(f"coco.names not found: {names_path}")

        with names_path.open("r") as fp:
            class_names = [line.rstrip() for line in fp if line.strip()]

        return class_names

    def _format_detections(self, detections: List) -> List[Dict[str, Any]]:
        """Format tt-metal detections into structured output."""
        formatted_detections = []

        for detection in detections:
            if len(detection) < 7:
                continue

            x1, y1, x2, y2, confidence, _, class_id = detection[:7]
            class_id_int = int(class_id)

            class_name = "unknown"
            if self.class_names and class_id_int < len(self.class_names):
                class_name = self.class_names[class_id_int]

            formatted_detections.append(
                {
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "confidence": float(confidence),
                    "class_id": class_id_int,
                    "class_name": class_name,
                }
            )

        return formatted_detections

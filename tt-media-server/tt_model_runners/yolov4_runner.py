# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import subprocess
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any

import torch
import ttnn

from config.settings import settings
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.image_manager import ImageManager
from domain.image_search_request import ImageSearchRequest

from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.demos.yolov4.reference.yolov4 import Yolov4
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE  # 10960
from models.demos.yolov4.common import get_mesh_mappers  # Use models.demos.utils.common_demo_utils for tt-metal commit v0.63+


# Constants
DEFAULT_RESOLUTION = (320, 320)
DEFAULT_TRACE_REGION_SIZE = 6434816
DEFAULT_NUM_COMMAND_QUEUES = 2
WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS = 300
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_NMS_THRESHOLD = 0.4
DEFAULT_INFERENCE_TIMEOUT_SECONDS = 60  # YOLOv4 inference timeout


class YoloV4ModelError(Exception):
    """Base exception for YOLOv4 model errors"""
    pass


class InferenceError(YoloV4ModelError):
    """Error occurred during model inference"""
    pass


class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds timeout limit"""
    pass


class TTYolov4Runner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.tt_device = None
        self.model = None
        self.class_names: List[str] = []
        self.resolution = DEFAULT_RESOLUTION
        # Use configured batch size from settings, with warning for YOLOv4 L1 memory constraints
        configured_batch_size = settings.max_batch_size
        if configured_batch_size > 1:
            self.logger.warning(
                f"Configured batch_size {configured_batch_size} may exceed YOLOv4 L1 memory constraints. "
                f"Recommended to use batch_size=1 for stability."
            )
        self.batch_size = 1  # Force batch size to 1 for YOLOv4 memory stability
        # Device configuration - default to single device for stability
        self.use_single_device = getattr(settings, 'yolov4_use_single_device', True)
        if self.use_single_device:
            self.logger.info("YOLOv4 configured for single device operation (recommended for stability)")
        else:
            self.logger.warning("YOLOv4 configured for multi-device operation (may cause memory issues on some images)")
        # Image processing utility
        self.image_manager = ImageManager(storage_dir="")

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        return self._mesh_device()

    def _mesh_device(self):
        # YOLOv4 specific device parameters - use exact values from tt-metal
        device_params = {
            "l1_small_size": YOLOV4_L1_SMALL_SIZE,
            "trace_region_size": DEFAULT_TRACE_REGION_SIZE,
            "num_command_queues": DEFAULT_NUM_COMMAND_QUEUES
        }
        device_ids = ttnn.get_device_ids()

        # Choose device configuration based on setting
        if self.use_single_device:
            # Single device configuration for memory stability
            num_devices_requested = 1
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)
        else:
            # Multi-device configuration (original behavior)
            param = len(device_ids)  # Default to using all available devices

            if isinstance(param, tuple):
                grid_dims = param
                assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
                num_devices_requested = grid_dims[0] * grid_dims[1]
                if num_devices_requested > len(device_ids):
                    self.logger.info("Requested more devices than available. Test not applicable for machine")
                mesh_shape = ttnn.MeshShape(*grid_dims)
                assert num_devices_requested <= len(device_ids), "Requested more devices than available."
            else:
                num_devices_requested = min(param, len(device_ids))
                mesh_shape = ttnn.MeshShape(1, num_devices_requested)

        updated_device_params = self.get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        device_text = "device" if mesh_device.get_num_devices() == 1 else "devices"
        self.logger.info(f"Mesh device with {mesh_device.get_num_devices()} {device_text} created for YOLOv4")
        return mesh_device

    def get_devices(self):
        device = self._mesh_device()
        device_shape = settings.device_mesh_shape
        return (device, device.create_submeshes(ttnn.MeshShape(*device_shape)))

    def close_device(self, device=None) -> bool:
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

    def _distribute_model(self) -> None:
        """Distribute the YOLOv4 model on the device.
        
        This method initializes the YOLOv4PerformantRunner with the loaded weights
        and distributes it across the available devices.
        """
        try:
            # Resolve tt-metal root path via environment variable
            tt_metal_home = Path(os.environ['TT_METAL_HOME'])

            # Set environment variable to ensure tt-metal uses our model_location_generator
            os.environ['TT_GH_CI_INFRA'] = '1'
            
            # Get mesh mappers for the device
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(self.tt_device)
            
            # Create model location generator that returns the directory containing weights
            def model_location_generator(rel_path, model_subdir="", download_if_ci_v2=False):
                """Return directory path where yolov4.pth weights file is located."""
                # tt-metal expects this to return a directory where it can find yolov4.pth
                weights_dir = tt_metal_home / "models" / "demos" / "yolov4" / "tests" / "pcc"
                return str(weights_dir)
            
            # Initialize performant runner on device with explicit paths
            self.logger.info(f"Initializing model with tt-metal home: {tt_metal_home}")
            # Ensure tt-metal root is on sys.path for imports
            if str(tt_metal_home) not in sys.path:
                sys.path.insert(0, str(tt_metal_home))

            self.model = YOLOv4PerformantRunner(
                self.tt_device,
                device_batch_size=self.batch_size,
                act_dtype=ttnn.bfloat16,
                weight_dtype=ttnn.bfloat16,
                resolution=self.resolution,
                model_location_generator=model_location_generator,
                mesh_mapper=inputs_mesh_mapper,
                mesh_composer=output_mesh_composer
            )
            self.logger.info("Using YOLOv4PerformantRunner from tt-metal")
        except Exception as e:
            self.logger.error(f"Error in model distribution: {e}")
            raise RuntimeError(
                f"Failed to initialize YOLOv4PerformantRunner under {tt_metal_home}"
            ) from e

    async def load_model(self, device) -> bool:
        self.logger.info("Loading YOLOv4 model...")
        
        # Setup device
        if device is None:
            self.tt_device = self.get_device()
        else:
            self.tt_device = device

        # Ensure tt-metal root is importable
        tt_metal_home_str = os.environ.get('TT_METAL_HOME')
        if not tt_metal_home_str:
            raise RuntimeError("TT_METAL_HOME environment variable not set")
        tt_metal_home = Path(tt_metal_home_str)
        if not tt_metal_home.exists():
            raise RuntimeError(f"tt-metal home not found at {tt_metal_home}")
        if str(tt_metal_home) not in sys.path:
            sys.path.insert(0, str(tt_metal_home))

        # Load class names
        self.class_names = self._load_class_names()
        self.logger.info(f"Loaded {len(self.class_names)} class names")

        # Load model weights with HF primary and Google Drive fallback
        self.torch_model = self._load_model_weights()
        self.logger.info("Model weights loaded successfully")

        # Distribute model on device with timeout
        def distribute_block():
            self._distribute_model()

        weights_distribution_timeout = WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS
        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Model distribution timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Exception during model loading: {e}")
            raise

        # Warmup inference
        self.logger.info("Running warmup inference...")
        dummy_image = torch.zeros(self.batch_size, 3, *self.resolution)
        _ = self.model.run(dummy_image)
        self.logger.info("Model warmup completed")

        self.logger.info("YOLOv4 model loaded successfully")
        return True

    def _load_model_weights(self):
        """Load YOLOv4 model weights from Google Drive (official tt-metal weights)."""
        # Use Google Drive to download official tt-metal YOLOv4 weights
        tt_metal_home = Path(os.environ['TT_METAL_HOME'])
        weights_path = tt_metal_home / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4.pth"
        download_script = tt_metal_home / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4_weights_download.sh"
        download_cwd = tt_metal_home
        
        if not weights_path.exists():
            self.logger.info("Downloading YOLOv4 weights from Google Drive...")
            try:
                # Execute the download script if it exists, otherwise use direct download
                if download_script.exists():
                    result = subprocess.run(
                        ["bash", str(download_script)],
                        cwd=str(download_cwd),
                        capture_output=True, 
                        text=True, 
                        timeout=300
                    )
                    if result.returncode != 0:
                        self.logger.warning(f"Download script failed: {result.stderr}")
                        raise RuntimeError("Download script execution failed")
            except Exception as e:
                self.logger.error(f"Failed to download weights: {e}")
                raise RuntimeError(f"Could not download model weights: {e}")

        try:
            torch_dict = torch.load(weights_path, map_location='cpu')
            model = Yolov4()
            model.load_state_dict(dict(zip(model.state_dict().keys(), torch_dict.values())))
            model.eval()
            self.logger.info("Successfully loaded YOLOv4 from official tt-metal weights")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise RuntimeError(f"Could not load model weights: {e}")

    def run_inference(self, image_data_list, timeout_seconds: int = None):
        # Set default timeout if not provided
        if timeout_seconds is None:
            timeout_seconds = DEFAULT_INFERENCE_TIMEOUT_SECONDS

        # Handle single or batch inputs; convert ImageSearchRequest objects to base64 strings
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]
        
        # Convert ImageSearchRequest objects to base64 strings
        processed_image_data = []
        for item in image_data_list:
            if isinstance(item, ImageSearchRequest):
                processed_image_data.append(item.prompt)
            elif isinstance(item, str):
                processed_image_data.append(item)
            else:
                raise ValueError(f"Unsupported image data type: {type(item)}. Expected str or ImageSearchRequest.")
        
        image_data_list = processed_image_data
        start_time = time.time()
        results = []
        
        # Process images in batches
        for batch_start in range(0, len(image_data_list), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_data_list))
            batch_images = image_data_list[batch_start:batch_end]
            current_batch_size = len(batch_images)
            
            try:
                # Check timeout before processing batch
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    raise InferenceTimeoutError(
                        f"Inference timed out after {elapsed_time:.2f}s before processing batch {batch_start//self.batch_size + 1}"
                    )
                
                self.logger.info(f"Processing batch {batch_start//self.batch_size + 1} with {current_batch_size} images (timeout: {timeout_seconds}s)")
                
                # Prepare batch tensor
                batch_tensor = self._prepare_batch_tensor(batch_images, current_batch_size)
                
                # Run inference on the batch with a hard timeout using a separate thread
                remaining_time = max(0, timeout_seconds - (time.time() - start_time))
                if remaining_time == 0:
                    raise InferenceTimeoutError(
                        f"Inference hard-timeout of {timeout_seconds}s reached before starting inference on batch {batch_start//self.batch_size + 1}"
                    )

                inference_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.model.run, batch_tensor)
                    try:
                        raw_output = future.result(timeout=remaining_time)
                    except concurrent.futures.TimeoutError:
                        # Ensure the future is cancelled and raise timeout error
                        future.cancel()
                        raise InferenceTimeoutError(
                            f"Inference hard-timeout of {timeout_seconds}s reached during inference on batch {batch_start//self.batch_size + 1}"
                        )

                inference_time = time.time() - inference_start
                
                # Check timeout after inference
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    raise InferenceTimeoutError(
                        f"Inference timed out after {elapsed_time:.2f}s during inference on batch {batch_start//self.batch_size + 1}"
                    )
                
                self.logger.info(f"Batch inference completed in {inference_time:.3f}s for {current_batch_size} images")
                
                # Post-process the batch output using tt-metal implementation
                boxes_batch = post_processing(
                    batch_tensor, 
                    conf_thresh=DEFAULT_CONFIDENCE_THRESHOLD,
                    nms_thresh=DEFAULT_NMS_THRESHOLD, 
                    output=raw_output
                )
                
                # Check timeout after post-processing
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    raise InferenceTimeoutError(
                        f"Inference timed out after {elapsed_time:.2f}s during post-processing of batch {batch_start//self.batch_size + 1}"
                    )
                
                # Format detections for each image in the batch
                for i, boxes in enumerate(boxes_batch):
                    detections = self._format_detections(boxes)
                    results.append(detections)
                
            except InferenceTimeoutError:
                raise
            except Exception as e:
                batch_num = batch_start // self.batch_size + 1
                self.logger.error(f"Error during inference on batch {batch_num}: {e}")
                raise InferenceError(f"Inference failed on batch {batch_num}: {str(e)}") from e

        total_time = time.time() - start_time
        num_batches = (len(image_data_list) + self.batch_size - 1) // self.batch_size
        self.logger.info(f"Completed inference on {len(image_data_list)} images in {num_batches} batches in {total_time:.3f}s")
        return results
    


    def _prepare_batch_tensor(self, batch_images: List[str], current_batch_size: int) -> torch.Tensor:
        """Prepare batch tensor from list of base64 image strings."""
        batch_tensors = []
        
        # Convert each image to tensor using ImageManager
        for image_base64 in batch_images:
            # Use ImageManager to get tensor without batch dimension
            image_tensor = self.image_manager.prepare_image_tensor(
                image_base64,
                target_size=self.resolution,
                target_mode="RGB"
            )
            # Remove the batch dimension that prepare_image_tensor adds
            image_tensor = image_tensor.squeeze(0)
            batch_tensors.append(image_tensor)
        
        # Stack into batch tensor [batch_size, channels, height, width]
        batch_tensor = torch.stack(batch_tensors, dim=0)
        
        # Pad batch to configured batch_size if necessary
        if current_batch_size < self.batch_size:
            # Create padding with zeros
            padding_size = self.batch_size - current_batch_size
            padding_shape = (padding_size, 3, *self.resolution)
            padding = torch.zeros(padding_shape)
            batch_tensor = torch.cat([batch_tensor, padding], dim=0)
        
        return batch_tensor



    def _load_class_names(self) -> List[str]:
        """Load COCO class names from file."""
        # Load from local resources directory
        names_path = Path(__file__).resolve().parent / "resources" / "coco.names"
        
        if names_path.exists():
            class_names = []
            with names_path.open("r") as fp:
                for line in fp.readlines():
                    line = line.rstrip()
                    if line:  # Skip empty lines
                        class_names.append(line)
            self.logger.info(f"Loaded class names from {names_path}")
            return class_names
        
        raise FileNotFoundError(
            f"coco.names not found at expected location: {names_path}"
        )

    def _format_detections(self, detections: List) -> List[Dict[str, Any]]:
        """Format detections into a structured output."""
        formatted_detections = []
        
        for detection in detections:
            # Each detection from tt-metal post_processing contains [x1, y1, x2, y2, confidence, confidence_duplicate, class_id]
            if len(detection) >= 7:
                x1, y1, x2, y2, confidence, _, class_id = detection[:7]
                
                # Get class name
                class_name = "unknown"
                if self.class_names and int(class_id) < len(self.class_names):
                    class_name = self.class_names[int(class_id)]
                
                formatted_detection = {
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": class_name
                }
                formatted_detections.append(formatted_detection)
        
        return formatted_detections


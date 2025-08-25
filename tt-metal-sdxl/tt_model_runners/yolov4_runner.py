# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import base64
import os
import subprocess
import sys
import time
# Added for hard-timeout implementation
import concurrent.futures
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager

import numpy as np
from PIL import Image
import torch
import ttnn

from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger

from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.demos.yolov4.reference.yolov4 import Yolov4
from models.demos.yolov4.common import get_mesh_mappers
from tests.scripts.common import get_updated_device_params


# Constants
DEFAULT_RESOLUTION = (320, 320)
DEFAULT_BATCH_SIZE = 1
DEFAULT_L1_SMALL_SIZE = 10960
DEFAULT_TRACE_REGION_SIZE = 6434816
DEFAULT_NUM_COMMAND_QUEUES = 2
WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS = 120
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_NMS_THRESHOLD = 0.4
DEFAULT_NMS_THRESHOLD_CPU = 0.5
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


class TTYolov4Runner(DeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.tt_device = None
        self.model = None
        self.class_names: List[str] = []
        self.resolution = DEFAULT_RESOLUTION
        self.batch_size = settings.max_batch_size

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        return self._mesh_device()

    def _mesh_device(self):
        # YOLOv4 specific device parameters
        device_params = {
            "l1_small_size": DEFAULT_L1_SMALL_SIZE,
            "trace_region_size": DEFAULT_TRACE_REGION_SIZE,
            "num_command_queues": DEFAULT_NUM_COMMAND_QUEUES
        }
        device_ids = ttnn.get_device_ids()

        param = len(device_ids)  # Default to using all available devices

        if isinstance(param, tuple):
            grid_dims = param
            assert len(grid_dims) == 2, "Device mesh grid shape should have exactly two elements."
            num_devices_requested = grid_dims[0] * grid_dims[1]
            if num_devices_requested > len(device_ids):
                print("Requested more devices than available. Test not applicable for machine")
            mesh_shape = ttnn.MeshShape(*grid_dims)
            assert num_devices_requested <= len(device_ids), "Requested more devices than available."
        else:
            num_devices_requested = min(param, len(device_ids))
            mesh_shape = ttnn.MeshShape(1, num_devices_requested)

        updated_device_params = get_updated_device_params(device_params)
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

        self.logger.info(f"Mesh device with {mesh_device.get_num_devices()} devices created for YOLOv4")
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

    @contextmanager
    def _temporarily_chdir(self, target_path: Path):
        previous_cwd = os.getcwd()
        os.chdir(str(target_path))
        try:
            yield
        finally:
            os.chdir(previous_cwd)


    def _distribute_model(self) -> None:
        """Distribute the YOLOv4 model on the device.
        
        This method initializes the YOLOv4PerformantRunner with the loaded weights
        and distributes it across the available devices.
        """
        try:
            # Resolve tt-metal root path via environment variable
            tt_metal_path = Path(os.environ['TT_METAL_PATH'])
            
            # Get mesh mappers for the device
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(self.tt_device)
            
            # Create model location generator with explicit path
            def model_location_generator(rel_path):
                """Generate absolute path for model files."""
                return str(tt_metal_path / rel_path)
            
            # Initialize performant runner on device with explicit paths
            self.logger.info(f"Initializing model with tt-metal path: {tt_metal_path}")
            # Ensure tt-metal root is on sys.path for imports
            if str(tt_metal_path) not in sys.path:
                sys.path.insert(0, str(tt_metal_path))

            # Some internal runner utilities rely on relative paths; ensure they resolve by temporarily
            # switching to the tt-metal repository root during initialization
            with self._temporarily_chdir(tt_metal_path):
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
                f"Failed to initialize YOLOv4PerformantRunner under {tt_metal_path}"
            ) from e

    async def load_model(self, device) -> bool:
        self.logger.info("Loading YOLOv4 model...")
        
        # Setup device
        if device is None:
            self.tt_device = self.get_device()
        else:
            self.tt_device = device

        # Ensure tt-metal root is importable
        tt_metal_path_str = os.environ.get('TT_METAL_PATH')
        if not tt_metal_path_str:
            raise RuntimeError("TT_METAL_PATH environment variable not set")
        tt_metal_path = Path(tt_metal_path_str)
        if not tt_metal_path.exists():
            raise RuntimeError(f"tt-metal path not found at {tt_metal_path}")
        if str(tt_metal_path) not in sys.path:
            sys.path.insert(0, str(tt_metal_path))

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
        """Load YOLOv4 model weights with HF primary and Google Drive fallback."""
        # Try HuggingFace first
        try:
            self.logger.info("Attempting to load YOLOv4 from HuggingFace...")
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained("ultralytics/yolov4", trust_remote_code=True)
            model = Yolov4()
            model.load_state_dict(hf_model.state_dict())
            model.eval()
            self.logger.info("Successfully loaded YOLOv4 from HuggingFace")
            return model
        except Exception as hf_error:
            self.logger.warning(f"HuggingFace loading failed: {hf_error}, falling back to Google Drive")
        
        # Fallback to Google Drive download (paths resolved from TT_METAL_PATH)
        tt_metal_path = Path(os.environ['TT_METAL_PATH'])
        weights_path = tt_metal_path / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4.pth"
        download_script = tt_metal_path / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4_weights_download.sh"
        download_cwd = tt_metal_path
        
        if not weights_path.exists():
            self.logger.info("Downloading YOLOv4 weights...")
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
                else:
                    # Direct download if script doesn't exist
                    import gdown
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    gdown.download("https://drive.google.com/uc?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ", str(weights_path))
            except Exception as e:
                self.logger.error(f"Failed to download weights: {e}")
                raise RuntimeError(f"Could not download model weights: {e}")

        try:
            torch_dict = torch.load(weights_path, map_location='cpu')
            model = Yolov4()
            model.load_state_dict(dict(zip(model.state_dict().keys(), torch_dict.values())))
            model.eval()
            self.logger.info("Successfully loaded YOLOv4 from local weights")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise RuntimeError(f"Could not load model weights: {e}")

    def run_inference(self, image_data_list, num_inference_steps: int = None, timeout_seconds: int = None):
        # Set default timeout if not provided
        if timeout_seconds is None:
            timeout_seconds = DEFAULT_INFERENCE_TIMEOUT_SECONDS

        # Handle single or batch inputs; each element expected to be base64-encoded image
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]

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
                
                # Post-process the batch output
                boxes_batch = self._post_processing(
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
    
    def _prepare_image_tensor(self, image_base64: str) -> torch.Tensor:
        """Prepare image tensor from base64 string."""
        pil_image = self._base64_to_pil_image(
            image_base64, 
            target_size=self.resolution, 
            target_mode="RGB"
        )
        image_np = np.array(pil_image)
        
        # Convert to NCHW float tensor in [0,1]
        if len(image_np.shape) == 3:
            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1)
            ).float().div(255.0).unsqueeze(0)
        elif len(image_np.shape) == 4:
            image_tensor = torch.from_numpy(
                image_np.transpose(0, 3, 1, 2)
            ).float().div(255.0)
        else:
            raise ValueError(f"Unexpected image shape: {image_np.shape}")
        
        return image_tensor

    def _prepare_batch_tensor(self, batch_images: List[str], current_batch_size: int) -> torch.Tensor:
        """Prepare batch tensor from list of base64 image strings."""
        batch_tensors = []
        
        # Convert each image to tensor
        for image_base64 in batch_images:
            pil_image = self._base64_to_pil_image(
                image_base64, 
                target_size=self.resolution, 
                target_mode="RGB"
            )
            image_np = np.array(pil_image)
            
            # Convert to CHW float tensor in [0,1] (no batch dimension yet)
            if len(image_np.shape) == 3:
                image_tensor = torch.from_numpy(
                    image_np.transpose(2, 0, 1)
                ).float().div(255.0)
            else:
                raise ValueError(f"Unexpected image shape: {image_np.shape}")
            
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

    def _base64_to_pil_image(self, base64_string, target_size=DEFAULT_RESOLUTION, target_mode="RGB"):
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        if image.mode != target_mode:
            image = image.convert(target_mode)
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

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
            # Each detection contains [x1, y1, x2, y2, confidence, class_id]
            if len(detection) >= 6:
                x1, y1, x2, y2, confidence, class_id = detection[:6]
                
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

    def _post_processing(self, img, conf_thresh, nms_thresh, output):
        """Post-process YOLOv4 output to get bounding boxes."""
        # Handle empty or invalid output
        if not output or len(output) < 2:
            self.logger.warning("Invalid or empty output from model")
            return []
        
        box_array = output[0]
        confs = output[1]

        # Convert to numpy arrays
        if isinstance(box_array, torch.Tensor):
            box_array = box_array.cpu().detach().numpy()
        else:
            box_array = np.array(box_array.to(torch.float32))
            
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().detach().numpy()
        else:
            confs = np.array(confs.to(torch.float32))

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = self._nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )

            bboxes_batch.append(bboxes)

        return bboxes_batch

    def _nms_cpu(self, boxes, confs, nms_thresh=DEFAULT_NMS_THRESHOLD_CPU, min_mode=False):
        if boxes.size == 0:
            return np.array([], dtype=int)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)



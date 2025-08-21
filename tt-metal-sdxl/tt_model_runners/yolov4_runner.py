# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import base64
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
import ttnn

from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner
from utils.logger import TTLogger


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


class TTYolov4Runner(DeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.tt_device = None
        self.model = None
        self.class_names: List[str] = []
        self.resolution = DEFAULT_RESOLUTION
        self.batch_size = DEFAULT_BATCH_SIZE

    def _set_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(fabric_config)

    def _reset_fabric(self, fabric_config):
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def get_device(self):
        return self._mesh_device()

    def _mesh_device(self):
        # Use single-device mesh like other runners
        # YOLOv4 specific small L1 size
        device_params = {
            "l1_small_size": DEFAULT_L1_SMALL_SIZE,
            "trace_region_size": DEFAULT_TRACE_REGION_SIZE,
            "num_command_queues": DEFAULT_NUM_COMMAND_QUEUES
        }
        device_ids = ttnn.get_device_ids()
        num_devices_requested = min(1, len(device_ids))
        mesh_shape = ttnn.MeshShape(1, num_devices_requested)

        # Handle device parameters update if the helper is available
        try:
            from tests.scripts.common import get_updated_device_params
            updated_device_params = get_updated_device_params(device_params)
        except ImportError:
            # Fallback if tests.scripts.common is not available
            updated_device_params = device_params.copy()
        fabric_config = updated_device_params.pop("fabric_config", None)
        self._set_fabric(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
        self.logger.info(f"Mesh device with {mesh_device.get_num_devices()} device created for YOLOv4")
        return mesh_device

    def get_devices(self):
        device = self._mesh_device()
        device_shape = settings.device_mesh_shape
        return (device, device.create_submeshes(ttnn.MeshShape(*device_shape)))

    def close_device(self, device) -> bool:
        if device is None:
            for submesh in self.mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(self.mesh_device)
        else:
            ttnn.close_mesh_device(device)
        return True

    def _distribute_model(self) -> None:
        """Distribute the YOLOv4 model on the device.
        
        This method initializes the YOLOv4PerformantRunner with the loaded weights
        and distributes it across the available devices.
        """
        try:
            # Add tt-metal path to sys.path if needed
            tt_metal_path_str = os.environ.get('TT_METAL_PATH')
            if not tt_metal_path_str:
                raise RuntimeError(
                    "TT_METAL_PATH environment variable not set. "
                    "Please set it to the path of your tt-metal directory."
                )
            tt_metal_path = Path(tt_metal_path_str)
            if tt_metal_path.exists() and str(tt_metal_path) not in sys.path:
                sys.path.insert(0, str(tt_metal_path))
            
            from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
            from models.demos.yolov4.common import get_mesh_mappers
            
            # Get mesh mappers for the device
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(self.tt_device)
            
            # Create model location generator with explicit path
            def model_location_generator(rel_path):
                """Generate absolute path for model files."""
                return str(tt_metal_path / rel_path)
            
            # Initialize performant runner on device with explicit paths
            self.logger.info(f"Initializing model with tt-metal path: {tt_metal_path}")
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
            raise ImportError(
                "Could not import YOLOv4PerformantRunner from tt-metal. "
                "Ensure tt-metal is installed and properly configured."
            ) from e

    def _get_weights_from_hf_cache(self) -> Path:
        """Get YOLOv4 weights from HuggingFace cache, downloading if necessary.
        
        Returns:
            Path to the cached weights file
        """
        try:
            # Install huggingface_hub if needed
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                self.logger.info("huggingface_hub not found. Attempting to install...")
                
                # Try to install with proper error handling
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=60  # 60 second timeout for installation
                    )
                    
                    if result.returncode != 0:
                        self.logger.error(f"pip install failed with return code {result.returncode}")
                        if result.stderr:
                            self.logger.error(f"Error output: {result.stderr}")
                        raise RuntimeError(
                            "Failed to install huggingface_hub. Please install manually with:\n"
                            "  pip install huggingface_hub"
                        )
                    
                    # Try importing again after installation
                    from huggingface_hub import hf_hub_download
                    self.logger.info("Successfully installed huggingface_hub")
                    
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "pip install timed out. Please check your network connection and install manually:\n"
                        "  pip install huggingface_hub"
                    )
                except FileNotFoundError:
                    raise RuntimeError(
                        "pip not found. Please ensure pip is installed and available in your Python environment."
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to install huggingface_hub: {e}\n"
                        "Please install manually with: pip install huggingface_hub"
                    )
            
            # Download from HuggingFace using HF cache
            repo_id = settings.model_weights_path or "homohapiens/darknet-yolov4"
            filename = "yolov4.pth"
            
            self.logger.info(f"Loading YOLOv4 weights from HuggingFace cache (repo: {repo_id})")
            
            # Use HF cache directory (respects HF_HOME environment variable)
            # This will automatically handle caching and reuse across servers
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=None,  # Use default HF cache directory
                local_dir_use_symlinks=True  # Use symlinks for efficiency
            )
            
            cached_path = Path(cached_path)
            if cached_path.exists():
                self.logger.info(f"Weights loaded from HF cache: {cached_path}")
            else:
                raise FileNotFoundError(f"Failed to get weights from HF cache: {cached_path}")
            
            return cached_path
            
        except RuntimeError:
            # Re-raise RuntimeError with our custom messages
            raise
        except Exception as e:
            self.logger.error(f"Failed to get weights from HuggingFace: {e}")
            raise

    async def load_model(self, device) -> bool:
        self.logger.info("Loading YOLOv4 model...")

        # Resolve device
        self.tt_device = device or self.get_device()

        # Load class names
        try:
            self.class_names = self._load_class_names()
            self.logger.info(f"Loaded {len(self.class_names)} class names")
        except Exception as e:
            self.logger.warning(f"Failed to load class names: {e}")
            # Use default COCO class names if file not found
            self.class_names = self._get_default_coco_names()

        # Add tt-metal path to sys.path and set up paths
        tt_metal_path_str = os.environ.get('TT_METAL_PATH')
        if not tt_metal_path_str:
            raise RuntimeError(
                "TT_METAL_PATH environment variable not set. "
                "Please set it to the path of your tt-metal directory."
            )
        tt_metal_path = Path(tt_metal_path_str)
        if not tt_metal_path.exists():
            raise RuntimeError(f"tt-metal path not found at {tt_metal_path}")

        if str(tt_metal_path) not in sys.path:
            sys.path.insert(0, str(tt_metal_path))

        # Load model weights from HuggingFace cache
        try:
            from models.demos.yolov4.common import load_torch_model
            
            # Get weights from HF cache (downloads if necessary)
            weights_path = self._get_weights_from_hf_cache()
            
            # Set tt-metal path for model initialization
            os.environ['TT_METAL_PATH'] = str(tt_metal_path)
            
            # Load the model with cached weights
            self.logger.info(f"Loading model from cached weights: {weights_path}")
            self.torch_model = load_torch_model(str(weights_path))
            self.logger.info("Model weights loaded successfully from HF cache")
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}")
            raise

        # Distribute the model on device
        weights_distribution_timeout = WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS

        try:
            await asyncio.wait_for(asyncio.to_thread(self._distribute_model), timeout=weights_distribution_timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Model distribution timed out after {weights_distribution_timeout} seconds")
            raise
        except Exception as e:
            self.logger.error(f"Exception during model loading: {e}")
            raise

        # Perform warmup inference
        try:
            self.logger.info("Running warmup inference...")
            dummy_image = torch.zeros(self.batch_size, 3, *self.resolution)
            _ = self.model.run(dummy_image)
            self.logger.info("Warmup completed successfully")
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}")
            raise

        self.logger.info("YOLOv4 model loaded successfully")
        return True

    def run_inference(self, image_data_list, num_inference_steps: int = None):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle single or batch inputs; each element expected to be base64-encoded image
        if not isinstance(image_data_list, list):
            image_data_list = [image_data_list]

        results = []
        for image_base64 in image_data_list:
            # Convert base64 to image tensor
            image_tensor = self._prepare_image_tensor(image_base64)
            
            # Run inference through the model
            raw_output = self.model.run(image_tensor)
            
            # Post-process the output
            boxes_batch = self._post_processing(
                image_tensor, 
                conf_thresh=DEFAULT_CONFIDENCE_THRESHOLD,
                nms_thresh=DEFAULT_NMS_THRESHOLD, 
                output=raw_output
            )
            
            # Format the output with class names
            detections = self._format_detections(boxes_batch[0] if len(boxes_batch) > 0 else [])
            results.append(detections)

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



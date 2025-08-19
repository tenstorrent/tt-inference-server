# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import base64
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


class TTYolov4Runner(DeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.logger = TTLogger()
        self.tt_device = None
        self.model = None
        self.class_names: List[str] = []
        self.resolution = (320, 320)  # Default resolution
        self.batch_size = 1

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
            "l1_small_size": 10960,
            "trace_region_size": 6434816,
            "num_command_queues": 2
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
        import sys
        import os
        import subprocess
        tt_metal_path = Path(__file__).resolve().parents[3] / "tt-metal"
        if not tt_metal_path.exists():
            raise RuntimeError(f"tt-metal path not found at {tt_metal_path}")

        if str(tt_metal_path) not in sys.path:
            sys.path.insert(0, str(tt_metal_path))

        # Load model weights using tt-metal mechanism
        try:
            from models.demos.yolov4.common import load_torch_model
            self.logger.info("Loading YOLOv4 weights from tt-metal")
            
            # Check if weights need to be downloaded
            weights_path = tt_metal_path / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4.pth"
            if not weights_path.exists():
                self.logger.info("Weights not found, downloading...")
                # Download script path (for reference, but we download directly)
                download_script = tt_metal_path / "models" / "demos" / "yolov4" / "tests" / "pcc" / "yolov4_weights_download.sh"
                
                # Save current working directory
                original_cwd = os.getcwd()
                try:
                    # Change to tt-metal directory for download
                    os.chdir(str(tt_metal_path))
                    self.logger.info(f"Changed working directory to {os.getcwd()}")
                    
                    # Install gdown if needed
                    subprocess.run(["pip3", "install", "gdown"], check=True)
                    
                    # Download weights using gdown directly
                    file_id = "1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ"
                    output_path = "models/demos/yolov4/tests/pcc/yolov4.pth"
                    
                    # Ensure the directory exists
                    output_dir = Path(output_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {output_dir}")
                    
                    self.logger.info("Downloading weights using gdown...")
                    subprocess.run([
                        "python3", "-m", "gdown",
                        f"https://drive.google.com/uc?id={file_id}",
                        "-O", output_path
                    ], check=True)
                    
                    # Verify the file was created
                    if Path(output_path).exists():
                        self.logger.info(f"Downloaded weights to: {Path(output_path).absolute()}")
                    else:
                        self.logger.error(f"Download completed but file not found at: {Path(output_path).absolute()}")
                    
                    if not weights_path.exists():
                        raise FileNotFoundError(f"Failed to download weights to {weights_path}")
                    self.logger.info("Weights downloaded successfully")
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)
                    self.logger.info(f"Restored working directory to {os.getcwd()}")

            # Load the model with explicit path
            os.environ['TT_METAL_PATH'] = str(tt_metal_path)  # Ensure tt-metal path is available
            
            # Change to tt-metal directory for loading to ensure relative paths work
            original_cwd = os.getcwd()
            try:
                os.chdir(str(tt_metal_path))
                self.logger.info(f"Changed working directory to {os.getcwd()} for model loading")
                self.torch_model = load_torch_model(None)  # None uses default path
                self.logger.info("Model weights loaded successfully")
            finally:
                os.chdir(original_cwd)
                self.logger.info(f"Restored working directory to {os.getcwd()}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download weights: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}")
            raise

        def distribute_block():
            try:
                # Add tt-metal path to sys.path if needed
                import sys
                import os
                tt_metal_path = Path(__file__).resolve().parents[3] / "tt-metal"
                if tt_metal_path.exists() and str(tt_metal_path) not in sys.path:
                    sys.path.insert(0, str(tt_metal_path))
                
                from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
                from models.demos.yolov4.common import get_mesh_mappers
                
                # Get mesh mappers for the device
                inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(self.tt_device)
                
                # Change to tt-metal directory for model initialization
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(tt_metal_path))
                    self.logger.info(f"Changed working directory to {os.getcwd()} for model initialization")
                    
                    # Initialize performant runner on device with tt-metal weights
                    self.model = YOLOv4PerformantRunner(
                        self.tt_device,
                        device_batch_size=self.batch_size,
                        act_dtype=ttnn.bfloat16,
                        weight_dtype=ttnn.bfloat16,
                        resolution=self.resolution,
                        model_location_generator=None,  # None uses default path
                        mesh_mapper=inputs_mesh_mapper,
                        mesh_composer=output_mesh_composer
                    )
                    self.logger.info("Using YOLOv4PerformantRunner from tt-metal")
                finally:
                    os.chdir(original_cwd)
                    self.logger.info(f"Restored working directory to {os.getcwd()}")
            except Exception as e:
                self.logger.error(f"Error in distribute block: {e}")
                raise ImportError(
                    "Could not import YOLOv4PerformantRunner from tt-metal. "
                    "Ensure tt-metal is installed and properly configured."
                ) from e

        # 2 minutes to distribute the model on device
        weights_distribution_timeout = 120

        try:
            await asyncio.wait_for(asyncio.to_thread(distribute_block), timeout=weights_distribution_timeout)
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
                conf_thresh=0.3,  # Lower threshold for better detection
                nms_thresh=0.4, 
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

    def _base64_to_pil_image(self, base64_string, target_size=(320, 320), target_mode="RGB"):
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
        # Try multiple locations for coco.names
        possible_paths = [
            # Try tt-metal location
            Path(__file__).resolve().parents[3] / "tt-metal" / "models" / "demos" / "yolov4" / "resources" / "coco.names",
            # Try tt-metal-yolov4 location
            Path(__file__).resolve().parents[2] / "tt-metal-yolov4" / "server" / "coco.names",
            # Try local resources
            Path(__file__).resolve().parent / "resources" / "coco.names",
        ]
        
        for names_path in possible_paths:
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
            f"coco.names not found in any of the expected locations: "
            f"{', '.join(str(p) for p in possible_paths)}"
        )
    
    def _get_default_coco_names(self) -> List[str]:
        """Return default COCO class names if file not found."""
        return [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

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

    def _nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
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



# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Mesh-aware Face Recognition Runner for 8-device parallel inference.

Uses mesh device with data parallelism like yolov8s demo.
"""

import torch
import numpy as np
import ttnn
from loguru import logger


def get_mesh_mappers(device):
    """Get mesh mappers for data parallelism."""
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


class FaceRecognitionMeshRunner:
    """
    Mesh-aware runner for YuNet + SFace face recognition.
    
    Runs on all devices in mesh simultaneously using data parallelism:
    - Input batch is sharded across devices (1 image per device)
    - Weights are replicated to all devices
    - Outputs are concatenated back
    """
    
    def __init__(
        self,
        mesh_device,
        yunet_torch_model,
        sface_torch_model,
        input_height=640,
        input_width=640,
    ):
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.input_height = input_height
        self.input_width = input_width
        
        logger.info(f"Initializing FaceRecognitionMeshRunner on {self.num_devices} devices")
        
        # Get mesh mappers
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = get_mesh_mappers(mesh_device)
        
        # Import model creation functions
        from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model_with_mesh_mapper
        from models.experimental.sface.tt.ttnn_sface import create_sface_model_with_mesh_mapper
        
        # Create TTNN models with mesh mappers
        logger.info("Creating YuNet model on mesh...")
        self.yunet_model = create_yunet_model_with_mesh_mapper(
            mesh_device, 
            yunet_torch_model,
            weights_mesh_mapper=self.weights_mesh_mapper
        )
        
        logger.info("Creating SFace model on mesh...")
        self.sface_model = create_sface_model_with_mesh_mapper(
            mesh_device,
            sface_torch_model,
            weights_mesh_mapper=self.weights_mesh_mapper
        )
        
        logger.info("FaceRecognitionMeshRunner ready!")
    
    def run_yunet(self, batch_tensor):
        """
        Run YuNet on a batch of images.
        
        Args:
            batch_tensor: torch tensor [N, H, W, C] where N = num_devices
        
        Returns:
            YuNet outputs concatenated from all devices
        """
        # Convert to TTNN and shard across devices
        tt_input = ttnn.from_torch(
            batch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
            device=self.mesh_device,
        )
        
        # Run model (executes on all devices in parallel)
        cls_out, box_out, obj_out, kpt_out = self.yunet_model(tt_input)
        
        # Convert outputs back to torch (concatenated from all devices)
        cls_torch = [ttnn.to_torch(c, mesh_composer=self.output_mesh_composer) for c in cls_out]
        box_torch = [ttnn.to_torch(b, mesh_composer=self.output_mesh_composer) for b in box_out]
        obj_torch = [ttnn.to_torch(o, mesh_composer=self.output_mesh_composer) for o in obj_out]
        kpt_torch = [ttnn.to_torch(k, mesh_composer=self.output_mesh_composer) for k in kpt_out]
        
        return cls_torch, box_torch, obj_torch, kpt_torch
    
    def run_sface(self, face_batch):
        """
        Run SFace on a batch of aligned faces.
        
        Args:
            face_batch: torch tensor [N, 112, 112, 3] where N = num_devices
        
        Returns:
            Embeddings [N, 128] concatenated from all devices
        """
        # Convert to TTNN and shard across devices
        tt_input = ttnn.from_torch(
            face_batch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
            device=self.mesh_device,
        )
        
        # Run model
        tt_output = self.sface_model(tt_input)
        
        # Convert back to torch
        embeddings = ttnn.to_torch(tt_output, mesh_composer=self.output_mesh_composer)
        
        return embeddings
    
    def release(self):
        """Release resources."""
        logger.info("Releasing FaceRecognitionMeshRunner resources")

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Shared memory tensor transfer for zero-copy IPC between bridge server and ComfyUI.
"""

import mmap
import struct
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Tuple, Optional
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SharedTensorHandle:
    """Handle for a tensor stored in shared memory."""
    shm_name: str
    shape: Tuple[int, ...]
    dtype: str  # torch dtype as string
    size_bytes: int
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "shm_name": self.shm_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "size_bytes": self.size_bytes
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            shm_name=data["shm_name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            size_bytes=data["size_bytes"]
        )


class TensorBridge:
    """
    Manages shared memory tensor transfer between processes.
    
    Provides zero-copy tensor sharing via shared memory segments.
    """
    
    def __init__(self):
        self._active_segments = {}  # shm_name -> SharedMemory object
        
    def tensor_to_shm(self, tensor: torch.Tensor) -> SharedTensorHandle:
        """
        Transfer a PyTorch tensor to shared memory.
        
        Args:
            tensor: PyTorch tensor to share
            
        Returns:
            SharedTensorHandle with metadata for reconstructing the tensor
        """
        # Ensure tensor is contiguous and on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.contiguous()
        
        # Get tensor metadata
        shape = tensor.shape
        dtype_str = str(tensor.dtype)
        
        # Convert to numpy for shared memory
        np_array = tensor.numpy()
        size_bytes = np_array.nbytes
        
        # Create unique name for this shared memory segment
        shm_name = f"tt_comfy_{uuid.uuid4().hex[:16]}"
        
        try:
            # Create shared memory
            shm = shared_memory.SharedMemory(
                create=True,
                size=size_bytes,
                name=shm_name
            )
            
            # Copy data to shared memory
            shm_array = np.ndarray(
                shape=np_array.shape,
                dtype=np_array.dtype,
                buffer=shm.buf
            )
            np.copyto(shm_array, np_array)
            
            # Store reference to keep it alive
            self._active_segments[shm_name] = shm
            
            logger.debug(f"Created shared tensor: {shm_name}, shape={shape}, dtype={dtype_str}, size={size_bytes}")
            
            return SharedTensorHandle(
                shm_name=shm_name,
                shape=shape,
                dtype=dtype_str,
                size_bytes=size_bytes
            )
            
        except Exception as e:
            logger.error(f"Failed to create shared tensor: {e}")
            # Clean up on failure
            if shm_name in self._active_segments:
                try:
                    self._active_segments[shm_name].close()
                    self._active_segments[shm_name].unlink()
                except:
                    pass
                del self._active_segments[shm_name]
            raise
    
    def shm_to_tensor(self, handle: SharedTensorHandle) -> torch.Tensor:
        """
        Reconstruct a PyTorch tensor from shared memory.
        
        Args:
            handle: SharedTensorHandle with metadata
            
        Returns:
            PyTorch tensor reconstructed from shared memory
        """
        try:
            # Open existing shared memory
            shm = shared_memory.SharedMemory(name=handle.shm_name)
            
            # Parse dtype string back to numpy dtype
            torch_dtype = self._parse_torch_dtype(handle.dtype)
            np_dtype = self._torch_to_numpy_dtype(torch_dtype)
            
            # Create numpy array view of shared memory
            np_array = np.ndarray(
                shape=handle.shape,
                dtype=np_dtype,
                buffer=shm.buf
            )
            
            # Convert to torch tensor (this makes a copy)
            tensor = torch.from_numpy(np_array.copy())
            
            # Close shared memory (don't unlink, creator will do that)
            shm.close()
            
            logger.debug(f"Reconstructed tensor from {handle.shm_name}, shape={handle.shape}")
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to reconstruct tensor from {handle.shm_name}: {e}")
            raise
    
    def release_shm(self, handle: SharedTensorHandle):
        """
        Release a shared memory segment.
        
        Args:
            handle: SharedTensorHandle to release
        """
        shm_name = handle.shm_name
        if shm_name in self._active_segments:
            try:
                self._active_segments[shm_name].close()
                self._active_segments[shm_name].unlink()
                logger.debug(f"Released shared memory: {shm_name}")
            except Exception as e:
                logger.warning(f"Failed to cleanly release {shm_name}: {e}")
            finally:
                del self._active_segments[shm_name]
    
    def cleanup_all(self):
        """Clean up all active shared memory segments."""
        for shm_name in list(self._active_segments.keys()):
            try:
                self._active_segments[shm_name].close()
                self._active_segments[shm_name].unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {shm_name}: {e}")
        self._active_segments.clear()
        logger.info("Cleaned up all shared memory segments")
    
    @staticmethod
    def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
        """Parse torch dtype from string."""
        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float": torch.float32,
            "torch.float64": torch.float64,
            "torch.double": torch.float64,
            "torch.float16": torch.float16,
            "torch.half": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int": torch.int32,
            "torch.int64": torch.int64,
            "torch.long": torch.int64,
            "torch.int16": torch.int16,
            "torch.short": torch.int16,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    @staticmethod
    def _torch_to_numpy_dtype(torch_dtype: torch.dtype) -> np.dtype:
        """Convert torch dtype to numpy dtype."""
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return dtype_map.get(torch_dtype, np.float32)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_all()


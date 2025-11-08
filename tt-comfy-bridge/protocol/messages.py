# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Message protocol definitions for TT-Comfy Bridge IPC.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class OperationType(Enum):
    """Types of operations supported by the bridge."""
    
    # Model management
    INIT_MODEL = "init_model"
    UNLOAD_MODEL = "unload_model"
    
    # Inference operations
    ENCODE_PROMPT = "encode_prompt"
    GENERATE_LATENTS = "generate_latents"
    DENOISE_STEP = "denoise_step"
    DECODE_VAE = "decode_vae"
    FULL_INFERENCE = "full_inference"
    
    # Image-to-image operations
    ENCODE_IMAGE = "encode_image"
    IMG2IMG_INFERENCE = "img2img_inference"
    
    # Utility
    PING = "ping"
    SHUTDOWN = "shutdown"


class MessageStatus(Enum):
    """Status codes for responses."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class Request:
    """Request message structure."""
    operation: str  # OperationType value
    data: Dict[str, Any]
    request_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "data": self.data,
            "request_id": self.request_id
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            operation=data["operation"],
            data=data.get("data", {}),
            request_id=data.get("request_id")
        )


@dataclass
class Response:
    """Response message structure."""
    status: str  # MessageStatus value
    data: Dict[str, Any]
    error: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "request_id": self.request_id
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        return cls(
            status=data["status"],
            data=data.get("data", {}),
            error=data.get("error"),
            request_id=data.get("request_id")
        )
    
    @classmethod
    def success(cls, data: Dict[str, Any], request_id: Optional[str] = None):
        """Create a success response."""
        return cls(
            status=MessageStatus.SUCCESS.value,
            data=data,
            request_id=request_id
        )
    
    @classmethod
    def error(cls, error_msg: str, request_id: Optional[str] = None):
        """Create an error response."""
        return cls(
            status=MessageStatus.ERROR.value,
            data={},
            error=error_msg,
            request_id=request_id
        )


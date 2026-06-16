"""
Base Service - Common functionality for all model services
Provides subprocess management and error handling.
"""

import asyncio
import subprocess
import tempfile
import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModelService(ABC):
    """Base class for all model services with subprocess management."""
    
    def __init__(self, device_id: int, service_name: str):
        """Initialize base service."""
        self.device_id = device_id
        self.service_name = service_name
        self.is_warmed_up = False
        self.warmup_time = None
        self.inference_count = 0
        
        # Base TT Metal environment setup (common to all services)
        self.base_env = os.environ.copy()
        self.base_env.update({
            'TT_VISIBLE_DEVICES': str(device_id),
            'HF_HOME': '/home/container_app_user/.cache/huggingface',
            'PYTHONPATH': '/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal',
            'TT_METAL_HOME': '/home/container_app_user/tt-metal'
        })
        
        # Each service will add model-specific env vars (HF_MODEL, TT_MESH_GRAPH_DESC_PATH, etc.)
        
        logger.info(f"{self.service_name} service initialized on device {device_id}")
    
    async def run_subprocess(self, cmd: List[str], input_data: Optional[bytes] = None, 
                           timeout: int = 60, extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        """Run a subprocess with proper environment and error handling."""
        try:
            logger.debug(f"{self.service_name}: Running command: {' '.join(cmd)}")
            
            # Combine base environment with service-specific environment
            env = self.base_env.copy()
            if extra_env:
                env.update(extra_env)
            
            # Run subprocess asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd='/home/container_app_user/tt-metal'
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data),
                timeout=timeout
            )
            
            # Create result object similar to subprocess.run
            result = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr
            )
            
            if result.returncode != 0:
                error_msg = f"{self.service_name} subprocess failed: {stderr.decode()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.inference_count += 1
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"{self.service_name} subprocess timed out after {timeout}s")
            raise RuntimeError(f"{self.service_name} inference timed out")
        except Exception as e:
            logger.error(f"{self.service_name} subprocess error: {e}")
            raise
    
    def create_temp_file(self, data: bytes, suffix: str = '.tmp') -> str:
        """Create a temporary file with the given data."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(data)
            return f.name
    
    def cleanup_temp_file(self, filepath: str):
        """Clean up a temporary file."""
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {filepath}: {e}")
    
    @abstractmethod
    async def warmup(self):
        """Warm up the model service."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Shutdown the model service."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "service_name": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "inference_count": self.inference_count
        }
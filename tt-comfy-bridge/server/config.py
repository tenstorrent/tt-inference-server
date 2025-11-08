# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Configuration for TT-Comfy Bridge Server.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BridgeConfig:
    """Configuration for the bridge server."""
    
    # Socket configuration
    socket_path: str = "/tmp/tt-comfy.sock"
    
    # Device configuration
    device_id: int = 0
    
    # Model paths
    tt_metal_home: str = os.getenv("TT_METAL_HOME", "/home/tt-admin/tt-metal")
    tt_media_server_path: str = "/home/tt-admin/tt-inference-server/tt-media-server"
    
    # Performance settings
    max_clients: int = 4
    buffer_size: int = 4096
    
    # Shared memory settings
    shm_size_mb: int = 512
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration."""
        self.socket_path = Path(self.socket_path)
        self.tt_metal_home = Path(self.tt_metal_home)
        self.tt_media_server_path = Path(self.tt_media_server_path)
        
        if not self.tt_metal_home.exists():
            raise ValueError(f"TT_METAL_HOME not found: {self.tt_metal_home}")
        
        if not self.tt_media_server_path.exists():
            raise ValueError(f"tt-media-server path not found: {self.tt_media_server_path}")


# Default configuration
DEFAULT_CONFIG = BridgeConfig()


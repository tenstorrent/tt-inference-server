# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os

from device_specs import DeviceClass, get_deployment


@dataclass
class H3Config:
    """Configuration for the MiniMax-H3 Turbo audio-video standalone server.

    Serves t2va and fl2va (first and/or last keyframe) from one pipeline, with
    the lightx2v 4-step Turbo adapter bound. Unlike LTX, clip shape is chosen
    per request: duration and aspect ratio, or a keyframe's own shape.

    Topology fields are derived from device_specs.DEPLOYMENTS based on `board`.
    """

    board: DeviceClass = None  # set explicitly via CLI; validated in __post_init__

    server_host: str = "127.0.0.1"
    server_port: int = 8000

    num_workers: int = 1
    device_ids: Tuple[int, ...] = ()
    device_mesh_shape: Tuple[int, int] = (0, 0)
    fabric_config_name: str = "FABRIC_1D"

    # tt-media-server's Blackhole default. The audio and video VAEs need a
    # non-zero pool; a bare open fails as "bank size is 0 B".
    l1_small_size: int = 65536

    # Diffusers snapshot of MiniMaxAI/MiniMax-H3. None lets the pipeline resolve
    # it from MINIMAX_H3_MODEL_PATH, then the HF cache.
    weights_dir: Optional[str] = field(default_factory=lambda: os.getenv("MINIMAX_H3_MODEL_PATH"))

    # The fl2va Turbo adapter. None resolves it from the HF cache (downloading on
    # first use), the same file tt-media-server's FL2VA LightX2V runner binds.
    lora_repo: str = "lightx2v/Minimax-h3-Turbo"
    lora_filename: str = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
    lora_path: Optional[str] = field(default_factory=lambda: os.getenv("MINIMAX_H3_LORA_PATH"))

    # The shifts the FL2VA adapter was distilled at, and its step count (4 forwards).
    video_shift: float = 6.0
    audio_shift: float = 3.0
    num_inference_steps: int = 5

    tt_dit_cache_dir: str = field(
        default_factory=lambda: os.getenv("TT_DIT_CACHE_DIR", os.path.expanduser("~/.cache/tt-dit"))
    )

    # Request defaults, matching tt-media-server's H3 policy.
    default_duration_s: int = 5
    default_aspect_ratio: str = "16:9"

    max_queue_size: int = 2
    inference_timeout_seconds: int = 1800

    # A cold start converts ~130 GB of weights into the tilized cache before the
    # warmup generation, which runs past the server's default 20-minute wait.
    warmup_timeout_seconds: int = 3600

    # H3_DEV_MODE=true skips the warmup generation, so the first request pays
    # for kernel compilation instead of startup.
    dev_mode: bool = field(default_factory=lambda: os.getenv("H3_DEV_MODE", "false").lower() == "true")

    def __post_init__(self):
        if self.board is None:
            raise ValueError("H3Config.board is required (pass --board on the CLI)")

        dep = get_deployment("h3", self.board)
        self.num_workers = dep.num_workers
        self.device_mesh_shape = dep.mesh_shape
        self.device_ids = dep.device_ids
        if dep.fabric_config:
            self.fabric_config_name = dep.fabric_config

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os

from device_specs import DeviceClass, get_board_spec, get_deployment


@dataclass
class LTXConfig:
    """Configuration for the LTX-2.3 distilled audio-video standalone server.

    Topology fields (num_workers, device_mesh_shape, device_ids, fabric) are
    derived from device_specs.DEPLOYMENTS based on `board`.
    """

    # Required: which Tenstorrent board this server is running on.
    board: DeviceClass = None  # set explicitly via CLI; validated in __post_init__

    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    # Topology — derived from device_specs.get_deployment("ltx", board) in __post_init__.
    num_workers: int = 1
    device_ids: Tuple[int, ...] = ()
    device_mesh_shape: Tuple[int, int] = (0, 0)
    fabric_config_name: str = "FABRIC_1D"

    # Device parameters. l1_small_size is the audio vocoder's conv-tap scratch pool:
    # it defaults to 0, which OOMs the vocoder during audio decode. No trace_region_size
    # because this pipeline runs untraced (see use_trace).
    l1_small_size: int = 32768

    # Model settings. The spatial upsampler auto-resolves from the Lightricks/LTX-2.3
    # repo; the checkpoint resolves from LTX_CHECKPOINT > ~/.cache/ltx-checkpoints > HF.
    model_name: str = "ltx-2.3-22b-distilled-1.1.safetensors"
    gemma_path: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    hf_home: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Weight cache for tilized tensors. Unset, every weight load is a cache miss and
    # startup goes from ~40s to minutes, so this is exported by the worker.
    tt_dit_cache_dir: str = field(
        default_factory=lambda: os.getenv("TT_DIT_CACHE_DIR", os.path.expanduser("~/.cache/tt-dit"))
    )

    # DiT-linear quantization. all_bf8_lofi is the shipped 1080p tier (its perf/VBench
    # floors are calibrated against it) and roughly halves DiT DRAM.
    quant: str = "all_bf8_lofi"

    # Geometry. Fixed at pipeline creation: the latent upsampler builds its GroupNorm
    # for a specific T*H*W, so a generate() call at a different shape asserts mid-run.
    # Defaults to a 10.04s clip (241 = 8n+1 frames at 24fps) at half 1080p.
    num_frames: int = 241
    height: int = 576
    width: int = 1024
    fps: int = 24

    # Step count is NOT configurable: the distilled schedules are module constants
    # (DISTILLED_SIGMA_VALUES + STAGE_2_DISTILLED_SIGMA_VALUES = 8 + 3 = 11 steps).

    # Untraced, matching the Wan and Flux.2 2x2 precedent: dynamic_load pages weights
    # per stage and a resident trace fights that.
    use_trace: bool = False

    # Queue settings
    max_queue_size: int = 4
    inference_timeout_seconds: int = 1800  # AV video gen is slow

    # Development mode: LTX_DEV_MODE=true → shortest clip that still satisfies 8n+1.
    dev_mode: bool = field(default_factory=lambda: os.getenv("LTX_DEV_MODE", "false").lower() == "true")

    def __post_init__(self):
        if self.board is None:
            raise ValueError("LTXConfig.board is required (pass --board on the CLI)")

        dep = get_deployment("ltx", self.board)
        self.num_workers = dep.num_workers
        self.device_mesh_shape = dep.mesh_shape
        self.device_ids = dep.device_ids
        if dep.fabric_config:
            self.fabric_config_name = dep.fabric_config

        if self.height % 64 or self.width % 64:
            raise ValueError(f"height and width must be divisible by 64 (got {self.height}x{self.width})")
        if (self.num_frames - 1) % 8:
            raise ValueError(f"(num_frames - 1) must be divisible by 8 (got {self.num_frames})")

        if self.dev_mode:
            self.num_frames = 25
            self.height = 576
            self.width = 1024

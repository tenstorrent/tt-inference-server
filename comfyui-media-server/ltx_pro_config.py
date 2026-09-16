# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import ClassVar
import os

from ltx_config import LTXConfig


@dataclass
class LTXProConfig(LTXConfig):
    """Configuration for the LTX-2.3 Pro (one-stage) audio-video server.

    Shares every device and geometry field with LTXConfig -- same mesh, same
    l1_small_size, same shape rules -- and differs in three ways:

      * a different checkpoint. Pro runs the *dev* 22B weights; the distilled
        checkpoint has the few-step schedule baked in and cannot take guidance.
      * guidance is live. The one-stage pipeline runs full MultiModalGuider CFG
        plus spatio-temporal guidance, so step count, the CFG/STG scales and the
        negative prompt all do something here (on the distilled pipeline they do
        not exist or are inert).
      * it is much slower: ~254s for a 241-frame 576x1024 clip against ~38s
        distilled, because it runs 30 guided steps rather than 11 unguided ones.
    """

    # Own deployment row, so topology can diverge from the distilled server later
    # without touching it.
    deployment_key: ClassVar[str] = "ltx_pro"

    # The dev checkpoint, not the distilled one.
    model_name: str = "ltx-2.3-22b-dev.safetensors"

    # Guidance. Defaults match the reference LTX_2_3_PARAMS that
    # pipeline_ltx_one_stage.generate() documents, so an unspecified request
    # reproduces the upstream configuration.
    num_inference_steps: int = 30
    video_cfg_scale: float = 3.0
    audio_cfg_scale: float = 7.0
    video_stg_scale: float = 1.0
    audio_stg_scale: float = 1.0
    video_modality_scale: float = 3.0
    audio_modality_scale: float = 3.0
    rescale_scale: float = 0.7
    stg_block: int = 28

    # Separate dev-mode switch so bringing up one server short does not shrink the
    # other.
    dev_mode: bool = field(default_factory=lambda: os.getenv("LTX_PRO_DEV_MODE", "false").lower() == "true")

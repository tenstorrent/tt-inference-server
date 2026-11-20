# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Placeholders for tt-metal's Wan and MiniMax-H3 policy modules, so the
request modules load where tt-metal isn't installed.

These are not tt-metal's values: they only have to be valid. Every check
accepts everything. No test may assert on them; tests that depend on the
model's values are marked `requires_tt_metal_policy` and skipped here.
"""

from dataclasses import dataclass
from types import ModuleType

WAN_POLICY = "models.tt_dit.pipelines.wan.policy"
H3_POLICY = "models.tt_dit.pipelines.minimax_h3.policy"


def _wan_policy() -> ModuleType:
    module = ModuleType(WAN_POLICY)
    module.WAN_RESOLUTIONS_480P = ((64, 32), (32, 64))
    module.WAN_RESOLUTIONS_720P = ((128, 64), (64, 128))
    module.WAN_DEFAULT_RESOLUTION = (64, 32)
    module.served_resolutions = lambda mesh_shape: (module.WAN_DEFAULT_RESOLUTION,)
    module.validate_request = lambda *, mesh_shape, width, height: None
    return module


class MiniMaxH3AdmissionError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(message for _, message in errors))
        self.errors = errors


@dataclass(frozen=True)
class MiniMaxH3ReferenceInfo:
    name: str
    kind: str
    size: tuple[int, int] | None = None
    fps: float | None = None
    audio_channels: int | None = None


def _h3_policy() -> ModuleType:
    module = ModuleType(H3_POLICY)
    values = {
        "MINIMAX_H3_ASPECT_RATIOS": ((1, 1),),
        "MINIMAX_H3_DEFAULT_ASPECT_RATIO": (1, 1),
        "MINIMAX_H3_DURATIONS_S": (1,),
        "MINIMAX_H3_DEFAULT_DURATION_S": 1,
        "MINIMAX_H3_FPS": 1,
        "MINIMAX_H3_FRAMES_PER_CHUNK": 1,
        "MINIMAX_H3_LATENTS_PER_CHUNK": 1,
        "MINIMAX_H3_SHORT_EDGE": 64,
        "MINIMAX_H3_MAX_PIXELS": 64 * 64,
        "MINIMAX_H3_CANVAS_MULTIPLE": 64,
        "MINIMAX_H3_MIN_ASPECT_RATIO": 1.0,
        "MINIMAX_H3_MAX_ASPECT_RATIO": 1.0,
        "MINIMAX_H3_KEYFRAME_MIN_SIDE": 1,
        "MINIMAX_H3_KEYFRAME_MAX_SIDE": 64,
        "MINIMAX_H3_MAX_REFERENCE_IMAGES": 1,
        "MINIMAX_H3_MAX_REFERENCE_VIDEOS": 1,
        "MINIMAX_H3_MAX_REFERENCE_AUDIOS": 1,
        "MINIMAX_H3_MAX_REFERENCES": 3,
        "MINIMAX_H3_AUDIO_CHANNELS": 2,
    }
    for name, value in values.items():
        setattr(module, name, value)
    module.MiniMaxH3AdmissionError = MiniMaxH3AdmissionError
    module.MiniMaxH3ReferenceInfo = MiniMaxH3ReferenceInfo
    module.get_num_frames = lambda duration_s: 1
    module.resolve_canvas_size = lambda width, height: (64, 64)
    module.admit_request = lambda **kwargs: None
    module.check_reference_kinds = lambda kinds: None
    module.minimax_h3_parse_aspect_ratio = lambda value: (1, 1)
    return module


def placeholder_policies() -> dict[str, ModuleType]:
    return {WAN_POLICY: _wan_policy(), H3_POLICY: _h3_policy()}

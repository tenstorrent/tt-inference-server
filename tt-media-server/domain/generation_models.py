# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib
from collections.abc import Iterator, Mapping

from config.constants import ModelRunners
from domain.generation_request import GenerationModel


class _GenerationModels(Mapping[str, GenerationModel]):
    """Imports a model's module only when its entry is read. The video
    models' modules import tt-metal's serving policies, and every service
    imports this mapping, so a deployment of another service would otherwise
    need them too. Checking a runner with ``in`` imports nothing."""

    def __init__(self, paths: dict[str, tuple[str, str]]):
        self._paths = paths

    def __getitem__(self, runner: str) -> GenerationModel:
        module, name = self._paths[runner]
        return getattr(importlib.import_module(module), name)

    def __contains__(self, runner: object) -> bool:
        return runner in self._paths

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


# What each generation runner serves, keyed by runner name: the module and
# the name of its GenerationModel.
GENERATION_MODELS: Mapping[str, GenerationModel] = _GenerationModels(
    {
        ModelRunners.TT_WAN_2_2_I2V.value: (
            "domain.wan22_i2v_a14b_request",
            "WAN22_I2V_A14B",
        ),
        ModelRunners.TT_QWEN_IMAGE_EDIT_2511.value: (
            "domain.qwen_image_edit_2511_request",
            "QWEN_IMAGE_EDIT_2511",
        ),
        ModelRunners.TT_LTX_2_3_DISTILLED.value: (
            "domain.ltx23_distilled_request",
            "LTX_2_3_22B_DISTILLED",
        ),
        ModelRunners.TT_MINIMAX_H3_FL2VA.value: (
            "domain.minimax_h3_request",
            "MINIMAX_H3_FL2VA",
        ),
        ModelRunners.TT_MINIMAX_H3_FL2VA_LIGHTX2V.value: (
            "domain.minimax_h3_request",
            "MINIMAX_H3_FL2VA_LIGHTX2V",
        ),
        ModelRunners.TT_MINIMAX_H3_REF2VA.value: (
            "domain.minimax_h3_request",
            "MINIMAX_H3_REF2VA",
        ),
        ModelRunners.TT_MINIMAX_H3_REF2VA_LIGHTX2V.value: (
            "domain.minimax_h3_request",
            "MINIMAX_H3_REF2VA_LIGHTX2V",
        ),
    }
)

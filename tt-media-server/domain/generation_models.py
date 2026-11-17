# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib
from collections.abc import Iterator, Mapping

from config.constants import ModelRunners
from domain.generation_request import GenerationModel

# What each generation runner serves, keyed by runner name: (module, attribute).
_MODEL_REFS: dict[str, tuple[str, str]] = {
    ModelRunners.TT_WAN_2_2_I2V.value: (
        "domain.wan22_i2v_a14b_request",
        "WAN22_I2V_A14B",
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
    ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value: (
        "domain.flux3_image_klein_request",
        "FLUX3_IMAGE_KLEIN",
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


class _GenerationModels(Mapping[str, GenerationModel]):
    """Imports a model's request module only when that model is looked up.

    The Wan and MiniMax-H3 request modules import tt-metal's policy modules,
    and with them ttnn. The uvicorn process must not load ttnn: it starts
    Tracy's profiler thread, a device worker forked afterwards inherits
    Tracy's queues without that thread, and nothing ever drains them (about
    10 MB per job on the quad's rank 0). This keeps a FLUX server's uvicorn
    process free of ttnn; a Wan or MiniMax-H3 server still loads it when it
    looks its own model up.
    """

    def __getitem__(self, runner: str) -> GenerationModel:
        module, attribute = _MODEL_REFS[runner]
        return getattr(importlib.import_module(module), attribute)

    def __contains__(self, runner: object) -> bool:
        return runner in _MODEL_REFS

    def __iter__(self) -> Iterator[str]:
        return iter(_MODEL_REFS)

    def __len__(self) -> int:
        return len(_MODEL_REFS)


GENERATION_MODELS: Mapping[str, GenerationModel] = _GenerationModels()

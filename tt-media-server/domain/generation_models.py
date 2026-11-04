# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from config.constants import ModelRunners
from domain.flux2_klein_4b_request import FLUX2_KLEIN_4B
from domain.generation_request import GenerationModel
from domain.ideogram4_request import IDEOGRAM_4
from domain.ltx23_distilled_request import LTX_2_3_22B_DISTILLED
from domain.minimax_h3_request import (
    MINIMAX_H3_FL2VA,
    MINIMAX_H3_FL2VA_LIGHTX2V,
    MINIMAX_H3_REF2VA,
    MINIMAX_H3_REF2VA_LIGHTX2V,
)
from domain.qwen_image_edit_2511_request import QWEN_IMAGE_EDIT_2511
from domain.wan22_i2v_a14b_request import WAN22_I2V_A14B

# What each generation runner serves, keyed by runner name.
GENERATION_MODELS: dict[str, GenerationModel] = {
    ModelRunners.TT_WAN_2_2_I2V.value: WAN22_I2V_A14B,
    ModelRunners.TT_QWEN_IMAGE_EDIT_2511.value: QWEN_IMAGE_EDIT_2511,
    ModelRunners.TT_LTX_2_3_DISTILLED.value: LTX_2_3_22B_DISTILLED,
    ModelRunners.TT_MINIMAX_H3_FL2VA.value: MINIMAX_H3_FL2VA,
    ModelRunners.TT_MINIMAX_H3_FL2VA_LIGHTX2V.value: MINIMAX_H3_FL2VA_LIGHTX2V,
    ModelRunners.TT_FLUX_2_KLEIN_4B.value: FLUX2_KLEIN_4B,
    ModelRunners.TT_MINIMAX_H3_REF2VA.value: MINIMAX_H3_REF2VA,
    ModelRunners.TT_MINIMAX_H3_REF2VA_LIGHTX2V.value: MINIMAX_H3_REF2VA_LIGHTX2V,
    ModelRunners.TT_IDEOGRAM_4.value: IDEOGRAM_4,
}

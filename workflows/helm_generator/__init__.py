# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.base_mapper import HelmValuesMapper
from workflows.helm_generator.forge.mapper import ForgeMapper
from workflows.helm_generator.media.mapper import MediaMapper
from workflows.helm_generator.vllm.mapper import VllmMapper
from workflows.workflow_types import InferenceEngine

MAPPERS = {
    InferenceEngine.VLLM.value: VllmMapper(),
    InferenceEngine.MEDIA.value: MediaMapper(),
    InferenceEngine.FORGE.value: ForgeMapper(),
}

# Precedence used to pick defaultEngine for a model that has multiple engines.
ENGINE_PRECEDENCE = ["vllm", "media", "forge"]

__all__ = [
    "MAPPERS",
    "ENGINE_PRECEDENCE",
    "HelmValuesMapper",
    "VllmMapper",
    "MediaMapper",
    "ForgeMapper",
]

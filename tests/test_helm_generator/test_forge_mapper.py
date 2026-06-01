# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.base_mapper import COMMON_OWNED_PATHS
from workflows.helm_generator.forge.mapper import ForgeMapper


def test_forge_engine_distinct_from_media(forge_spec):
    mapped = ForgeMapper().map(forge_spec)
    assert mapped.engine == "forge"
    assert mapped.model_name == "resnet-50"
    assert mapped.config.probes.liveness.path == "/tt-liveness"


def test_forge_owned_paths_include_liveness_path():
    paths = ForgeMapper().owned_leaf_paths()
    assert ("probes", "liveness", "path") in paths
    assert COMMON_OWNED_PATHS.issubset(paths)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Optional, Set, Tuple

from workflows.helm_generator.base_mapper import COMMON_OWNED_PATHS, HelmValuesMapper


class ForgeMapper(HelmValuesMapper):
    engine = "forge"
    liveness_path: Optional[str] = "/tt-liveness"
    readiness_path: Optional[str] = None

    def owned_leaf_paths(self) -> Set[Tuple[str, ...]]:
        return set(COMMON_OWNED_PATHS) | {("probes", "liveness", "path")}

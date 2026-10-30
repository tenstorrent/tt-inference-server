# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import dataclasses
from typing import ClassVar, Dict, Optional, Set, Tuple

from workflows.helm_generator.base_mapper import COMMON_OWNED_PATHS, HelmValuesMapper
from workflows.helm_generator.schema import HelmImplConfig, HelmProbe
from workflows.model_spec import ModelSpec


class MediaMapper(HelmValuesMapper):
    engine = "media"
    liveness_path: Optional[str] = "/tt-liveness"
    readiness_path: Optional[str] = None
    # MEDIA impls that run their own server instead of tt-media-server, so have no
    # /tt-liveness. tt_lab_glm: `glm-tt serve` answers only /health and /v1/models
    # (and the chart rewrites a non-vllm /v1/models liveness path to /tt-liveness).
    liveness_path_by_impl: ClassVar[Dict[str, str]] = {"tt_lab_glm": "/health"}

    def owned_leaf_paths(self) -> Set[Tuple[str, ...]]:
        return set(COMMON_OWNED_PATHS) | {("probes", "liveness", "path")}

    def _build_impl_config(self, spec: ModelSpec) -> HelmImplConfig:
        config = super()._build_impl_config(spec)
        path = self.liveness_path_by_impl.get(spec.impl.impl_id)
        if path is None:
            return config
        probes = dataclasses.replace(config.probes, liveness=HelmProbe(path=path))
        return dataclasses.replace(config, probes=probes)

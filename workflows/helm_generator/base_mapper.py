# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set, Tuple

from workflows.helm_generator.device import device_key
from workflows.helm_generator.schema import (
    HelmEnvVar,
    HelmImage,
    HelmImplConfig,
    HelmModelSpec,
    HelmProbe,
    HelmProbes,
    HelmResources,
)
from workflows.model_spec import ModelSpec


def model_name_from_spec(spec: ModelSpec) -> str:
    """The model name used as the top-level key in values.yaml. Shared with
    cli.py which groups specs before any mapper runs.
    """
    return Path(spec.hf_model_repo).name


COMMON_OWNED_PATHS: Set[Tuple[str, ...]] = {
    ("image", "repository"),
    ("image", "tag"),
    ("resources", "requests", "memory"),
    ("probes", "liveness", "initialDelaySeconds"),
    ("probes", "readiness", "initialDelaySeconds"),
    ("progressDeadlineSeconds",),
    ("env",),
}


class HelmValuesMapper(ABC):
    """Maps a ModelSpec to a HelmModelSpec. Subclasses declare engine + any
    engine-specific overrides (probe paths, etc.).
    """

    engine: str = ""
    liveness_path: Optional[str] = None
    readiness_path: Optional[str] = None

    @staticmethod
    def _split_image(docker_image: str) -> Tuple[str, str]:
        repo, _, tag = docker_image.rpartition(":")
        if not repo or not tag:
            raise ValueError(f"docker_image '{docker_image}' missing :tag")
        return repo, tag

    @staticmethod
    def _env_list(spec: ModelSpec) -> List[HelmEnvVar]:
        items = sorted(spec.env_vars.items(), key=lambda kv: kv[0])
        return [HelmEnvVar(name=k, value=str(v)) for k, v in items]

    @staticmethod
    def _progress_deadline_seconds(spec: ModelSpec) -> int:
        return int(spec.device_model_spec.tensor_cache_timeout) + 1800

    @staticmethod
    def _initial_delay_seconds(spec: ModelSpec) -> int:
        return int(spec.device_model_spec.tensor_cache_timeout * 2 / 3)

    @staticmethod
    def _requests_memory(spec: ModelSpec) -> Optional[str]:
        if spec.min_ram_gb is None:
            return None
        return f"{int(spec.min_ram_gb)}Gi"

    @abstractmethod
    def owned_leaf_paths(self) -> Set[Tuple[str, ...]]:
        """Leaf paths within an impl block that this mapper controls.

        merge.py overwrites only these paths; anything else in the existing
        impl block is preserved untouched.
        """

    def _build_impl_config(self, spec: ModelSpec) -> HelmImplConfig:
        repo, tag = self._split_image(spec.docker_image)
        initial_delay = self._initial_delay_seconds(spec)
        return HelmImplConfig(
            image=HelmImage(repository=repo, tag=tag),
            progress_deadline_seconds=self._progress_deadline_seconds(spec),
            probes=HelmProbes(
                liveness=HelmProbe(
                    initial_delay_seconds=initial_delay,
                    path=self.liveness_path,
                ),
                readiness=HelmProbe(
                    initial_delay_seconds=initial_delay,
                    path=self.readiness_path,
                ),
            ),
            resources=HelmResources(requests_memory=self._requests_memory(spec)),
            env=self._env_list(spec),
        )

    def map(self, spec: ModelSpec) -> HelmModelSpec:
        return HelmModelSpec(
            model_name=model_name_from_spec(spec),
            engine=self.engine,
            device_name=device_key(spec.device_type),
            impl_id=spec.impl.impl_id,
            is_default=spec.device_model_spec.default_impl,
            config=self._build_impl_config(spec),
        )

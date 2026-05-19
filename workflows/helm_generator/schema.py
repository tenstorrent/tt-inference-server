# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HelmImage:
    repository: str
    tag: str

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {"repository": self.repository, "tag": self.tag}


@dataclass(frozen=True)
class HelmProbe:
    initial_delay_seconds: int
    path: Optional[str] = None

    def to_yaml_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"initialDelaySeconds": self.initial_delay_seconds}
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class HelmProbes:
    liveness: HelmProbe
    readiness: HelmProbe

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {
            "liveness": self.liveness.to_yaml_dict(),
            "readiness": self.readiness.to_yaml_dict(),
        }


@dataclass(frozen=True)
class HelmResources:
    requests_memory: Optional[str] = None

    def to_yaml_dict(self) -> Optional[Dict[str, Any]]:
        if self.requests_memory is None:
            return None
        return {"requests": {"memory": self.requests_memory}}


@dataclass(frozen=True)
class HelmEnvVar:
    name: str
    value: str

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class HelmImplConfig:
    """One implementation of a model on a device. Maps to
    models.<m>.<engine>.<d>.impls.<impl_id> in values.yaml. No serverType
    field — the engine is encoded in the path, not the leaf.
    """

    image: HelmImage
    progress_deadline_seconds: int
    probes: HelmProbes
    env: List[HelmEnvVar] = field(default_factory=list)
    resources: HelmResources = field(default_factory=HelmResources)

    def to_yaml_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "progressDeadlineSeconds": self.progress_deadline_seconds,
            "image": self.image.to_yaml_dict(),
            "probes": self.probes.to_yaml_dict(),
        }
        resources = self.resources.to_yaml_dict()
        if resources is not None:
            out["resources"] = resources
        if self.env:
            out["env"] = [e.to_yaml_dict() for e in self.env]
        return out


@dataclass(frozen=True)
class HelmModelSpec:
    """A single ModelSpec mapped to its destination in values.yaml.

    Carries everything merge.py needs to write the spec into
    models.<model_name>.<engine>.<device_name>.impls.<impl_id>.
    """

    model_name: str
    engine: str
    device_name: str
    impl_id: str
    is_default: bool
    config: HelmImplConfig

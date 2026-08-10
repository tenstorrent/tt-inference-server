# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tenstorrent implementation of the engine's model catalog seam.

Adapts the Tenstorrent model catalog (``workflows.model_spec.MODEL_SPECS`` +
``get_runtime_model_spec``) to :class:`workflow_module.model_catalog.ModelSpecProvider`
so the engine never imports the catalog directly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from workflow_module.model_catalog import ModelSpecProvider
from workflows.model_spec import MODEL_SPECS, ModelSpec, get_runtime_model_spec

logger = logging.getLogger(__name__)

# Exact requirements-document ``deployment.hardware`` -> Tenstorrent device
# taxonomy mappings for SKUs that don't follow a naming pattern. Matched
# case-insensitively; an unmapped value fails loudly rather than guessing.
_HARDWARE_TO_DEVICE = {}

# Pattern-based mappings, tried in order after the exact table. The Super
# Cluster ships in several node counts (SC8, SC12, SC20, ...); every ``SC<N>``
# is the same SUPER_CLUSTER device from the engine's perspective (the node
# count affects capacity, not the device taxonomy the workflow keys on).
_HARDWARE_PATTERN_TO_DEVICE = ((re.compile(r"^SC\d+$"), "SUPER_CLUSTER"),)


def hardware_to_device_name(hardware: str) -> str:
    """Resolve a requirements ``deployment.hardware`` string to a device name.

    Returns the ``DeviceTypes`` member name (e.g. ``"SUPER_CLUSTER"``). Raises
    ``ValueError`` for an unmapped hardware string, listing what is supported.
    """
    from workflows.workflow_types import DeviceTypes

    if not hardware:
        raise ValueError("deployment.hardware is required to resolve a device")
    key = hardware.strip().upper()
    mapped = _HARDWARE_TO_DEVICE.get(key)
    if mapped is not None:
        return mapped
    for pattern, device_name in _HARDWARE_PATTERN_TO_DEVICE:
        if pattern.match(key):
            return device_name
    # Allow a document to name a DeviceTypes member directly (e.g. "galaxy").
    try:
        return DeviceTypes.from_string(key).name
    except (KeyError, ValueError):
        pass
    supported = sorted(
        set(_HARDWARE_TO_DEVICE)
        | {"SC<N> (e.g. SC8, SC12, SC20)"}
        | {d.name for d in DeviceTypes}
    )
    raise ValueError(
        f"Unknown deployment.hardware {hardware!r}; supported values: {supported}"
    )


class TenstorrentModelSpecProvider(ModelSpecProvider):
    """``ModelSpecProvider`` over the Tenstorrent YAML model catalog."""

    def model_names(self) -> List[str]:
        return sorted({spec.model_name for spec in MODEL_SPECS.values()})

    def resolve(self, model: str, device: str) -> ModelSpec:
        model_spec, _, _ = get_runtime_model_spec(model=model, device=device)
        return model_spec

    def resolve_candidates(self, model: str, device: str) -> List[ModelSpec]:
        return [
            config
            for config in MODEL_SPECS.values()
            if config.model_name == model
            and config.device_type.name.lower() == device.lower()
        ]

    def load_runtime_spec(self, path: str) -> Optional[ModelSpec]:
        try:
            return ModelSpec.from_json(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "Could not load model_spec from runtime_model_spec_json=%r (%s); "
                "falling back to catalog resolution by (model, device).",
                path,
                e,
            )
            return None

    def synthesize(
        self,
        *,
        model_name: str,
        hf_model_repo: str,
        device: str,
        max_context: int,
        max_concurrency: int,
    ) -> ModelSpec:
        from workflows.model_spec import DeviceModelSpec, ImplSpec
        from workflows.workflow_types import (
            DeviceTypes,
            InferenceEngine,
            ModelType,
        )

        try:
            device_type = DeviceTypes.from_string(device)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Cannot synthesize spec: unknown device {device!r}"
            ) from e

        # A generic vLLM impl placeholder: off-catalog synthesis targets an
        # already-running OpenAI-compatible server, so no tt-metal code path is
        # exercised — the impl only needs to be structurally valid.
        impl = ImplSpec(
            impl_id="requirements_synthesized",
            impl_name="requirements-synthesized",
            repo_url="",
            code_path="",
        )
        device_model_spec = DeviceModelSpec(
            device=device_type,
            max_concurrency=max_concurrency,
            max_context=max_context,
            default_impl=True,
        )
        model_id = f"{model_name}-{device_type.name.lower()}-requirements"
        logger.warning(
            "Synthesizing off-catalog ModelSpec for model_name=%r hf_repo=%r "
            "device=%s (max_context=%d, max_concurrency=%d). No catalog "
            "validation is applied.",
            model_name,
            hf_model_repo,
            device_type.name,
            max_context,
            max_concurrency,
        )
        return ModelSpec(
            model_id=model_id,
            impl=impl,
            hf_model_repo=hf_model_repo,
            model_name=model_name,
            inference_engine=InferenceEngine.VLLM.value,
            device_type=device_type,
            device_model_spec=device_model_spec,
            model_type=ModelType.LLM,
        )

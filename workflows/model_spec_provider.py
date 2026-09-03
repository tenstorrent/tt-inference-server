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
from typing import List, Optional

from workflow_module.model_catalog import ModelSpecProvider
from workflows.model_spec import MODEL_SPECS, ModelSpec, get_runtime_model_spec

logger = logging.getLogger(__name__)


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

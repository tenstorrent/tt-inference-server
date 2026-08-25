# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tenstorrent implementation of the engine's server lifecycle seam.

Adapts the Tenstorrent docker/local server launchers and the
tt-media-server vs vLLM(tt-metal) bearer-token policy to
:class:`workflow_module.server_lifecycle.ServerLifecycle` so the engine
never imports the launcher stack or the ``InferenceEngine`` taxonomy.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from workflow_module.commands import ServerMode
from workflow_module.server_lifecycle import ServerLifecycle
from workflows.workflow_types import InferenceEngine

logger = logging.getLogger(__name__)

# Engines whose servers validate a literal ``Bearer $API_KEY`` (no JWT decode).
_LITERAL_KEY_ENGINE_VALUES = frozenset(
    {InferenceEngine.FORGE.value, InferenceEngine.MEDIA.value}
)


class TenstorrentServerLifecycle(ServerLifecycle):
    """``ServerLifecycle`` over the Tenstorrent docker/local launchers."""

    def launch(
        self,
        mode: ServerMode,
        model_spec: Any,
        runtime_config: Any,
        setup_config: Any,
        json_fpath: Optional[str],
    ) -> Any:
        if mode is ServerMode.DOCKER:
            from workflows.run_docker_server import run_docker_server

            return run_docker_server(
                model_spec,
                runtime_config,
                setup_config,
                json_fpath,
            )
        if mode is ServerMode.LOCAL:
            from workflows.run_local_server import run_local_server

            return run_local_server(
                model_spec,
                runtime_config,
                json_fpath,
                setup_config,
            )
        raise ValueError(f"unknown server mode: {mode!r}")

    def uses_literal_api_key(self, inference_engine_value: Optional[str]) -> bool:
        return inference_engine_value in _LITERAL_KEY_ENGINE_VALUES

    def normalize_engine_value(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        # Tolerate the enum *name* form ("FORGE") as well as the value form.
        if raw in InferenceEngine.__members__:
            return InferenceEngine[raw].value
        return raw

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned server lifecycle seam.

The engine assumes a running inference server; who brings it up (and how
its auth works) is vendor adapter business. This module defines the seam:

- :class:`ServerLifecycle` — launch an inference server for a run, and
  answer server-facing policy questions the engine needs (bearer-token
  scheme, engine-name normalization).

The Tenstorrent implementation (docker/local launchers, tt-media-server
literal-key vs vLLM JWT auth) lives adapter-side and is injected via
:func:`register_server_lifecycle` at process entry. A lazy Tenstorrent
default is kept for backward compatibility during the extraction; it is
the marked Phase-2 removal point.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from workflow_module.commands import ServerMode

logger = logging.getLogger(__name__)


@runtime_checkable
class ServerLifecycle(Protocol):
    """Brings up the inference server and owns server-facing auth policy."""

    def launch(
        self,
        mode: ServerMode,
        model_spec: Any,
        runtime_config: Any,
        setup_config: Any,
        json_fpath: Optional[str],
    ) -> Any:
        """Bring up the server for ``mode`` (docker/local); returns launch payload.

        Raises on failure; the caller converts to a command result.
        """
        ...

    def uses_literal_api_key(self, inference_engine_value: Optional[str]) -> bool:
        """Whether the server for this engine expects a literal ``Bearer $API_KEY``.

        Tenstorrent: tt-media-server (forge/media) checks a literal key;
        only the vLLM (tt-metal) server validates a JWT — a minted JWT sent
        to a forge/media server 401s every request.
        """
        ...

    def normalize_engine_value(self, raw: Optional[str]) -> Optional[str]:
        """Normalize a serialized engine identifier to the adapter's value form.

        Runtime spec JSON may carry either the enum value (``"forge"``) or
        the enum name (``"FORGE"``); the adapter maps both to its canonical
        value form. Returns ``None`` for missing input.
        """
        ...


_lifecycle: Optional[ServerLifecycle] = None


def register_server_lifecycle(lifecycle: ServerLifecycle) -> None:
    """Install the process-wide server lifecycle (called at entry points)."""
    global _lifecycle
    _lifecycle = lifecycle


def get_server_lifecycle() -> ServerLifecycle:
    """Return the registered lifecycle, lazily defaulting to the TT launchers.

    EXTRACTION SEAM (Phase 2 removal point): the lazy default keeps
    pre-extraction callers working without an explicit registration. Once the
    engine is packaged separately, this fallback disappears and entry points
    must register a lifecycle explicitly.
    """
    global _lifecycle
    if _lifecycle is None:
        from workflows.server_lifecycle_provider import TenstorrentServerLifecycle

        logger.debug("No ServerLifecycle registered; using Tenstorrent launchers.")
        _lifecycle = TenstorrentServerLifecycle()
    return _lifecycle

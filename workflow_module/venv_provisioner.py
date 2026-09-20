# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned tool-environment (venv) provisioning seam.

The engine runs harnesses (lm_eval, lmms-eval, vllm bench, ...) out of
named tool environments keyed by :class:`engine_types.WorkflowVenvType`.
What each environment contains and how it is created is vendor adapter
business. This module defines the seam:

- :class:`VenvProvisioner` — resolve an environment's path/python and
  provision it on demand.

The Tenstorrent implementation (``workflows/workflow_venvs.py``'s
``VENV_CONFIGS``) lives adapter-side and is injected via
:func:`register_venv_provisioner` at process entry. A lazy Tenstorrent
default is kept for backward compatibility during the extraction; it is
the marked Phase-2 removal point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VenvProvisioner(Protocol):
    """Resolves and provisions named tool environments."""

    def has_venv(self, venv_type: Any) -> bool:
        """Whether ``venv_type`` is a known environment."""
        ...

    def venv_path(self, venv_type: Any) -> Path:
        """Root directory of the environment. Raises ``KeyError`` if unknown."""
        ...

    def venv_python(self, venv_type: Any) -> str:
        """Python interpreter of the environment. Raises ``KeyError`` if unknown."""
        ...

    def provision(self, venv_type: Any, model_spec: Any = None) -> bool:
        """Create/update the environment; returns True on success."""
        ...


_provisioner: Optional[VenvProvisioner] = None


def register_venv_provisioner(provisioner: VenvProvisioner) -> None:
    """Install the process-wide venv provisioner (called at entry points)."""
    global _provisioner
    _provisioner = provisioner


def get_venv_provisioner() -> VenvProvisioner:
    """Return the registered provisioner, lazily defaulting to VENV_CONFIGS.

    EXTRACTION SEAM (Phase 2 removal point): the lazy default keeps
    pre-extraction callers working without an explicit registration. Once the
    engine is packaged separately, this fallback disappears and entry points
    must register a provisioner explicitly.
    """
    global _provisioner
    if _provisioner is None:
        from workflows.venv_provisioner_provider import TenstorrentVenvProvisioner

        logger.debug("No VenvProvisioner registered; using Tenstorrent VENV_CONFIGS.")
        _provisioner = TenstorrentVenvProvisioner()
    return _provisioner

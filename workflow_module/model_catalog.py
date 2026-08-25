# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned model catalog seam.

The engine needs to resolve "which model am I validating" without knowing
anything about a specific vendor's catalog format. This module defines the
two protocols that make that possible:

- :class:`ModelSpecLike` — the structural surface the engine reads from a
  resolved model spec (names, repo, modalities, device-scoped limits).
- :class:`ModelSpecProvider` — who can resolve ``(model, device)`` into a
  spec, list the catalog, and load a pre-resolved runtime spec JSON.

The concrete catalog (Tenstorrent's ``workflows.model_spec.MODEL_SPECS``)
lives in the adapter layer and is injected via
:func:`register_model_spec_provider` at process entry (``run_workflows.py``,
``run.py`` dispatch). A lazy Tenstorrent default is kept for backward
compatibility during the extraction; it is the marked Phase-2 removal point.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from workflow_module.engine_types import ModelType

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelSpecLike(Protocol):
    """Structural surface the engine reads from a resolved model spec.

    Vendor adapters satisfy this automatically (structural typing); the
    engine never constructs one itself. Nested vendor-specific details
    (``device_model_spec``) are opaque to the engine core.
    """

    model_name: str
    model_id: str
    hf_model_repo: str
    model_type: Optional[ModelType]
    inference_engine: Any  # vendor engine taxonomy; engine reads ``.value``
    cli_args: Dict[str, Any]
    device_model_spec: Any  # vendor device-scoped limits (max_context, ...)
    supported_modalities: List[str]


@runtime_checkable
class ModelSpecProvider(Protocol):
    """Resolves model specs for the engine without exposing the catalog."""

    def model_names(self) -> List[str]:
        """All selectable model names (for CLI choices / validation)."""
        ...

    def resolve(self, model: str, device: str) -> ModelSpecLike:
        """Resolve the default-impl spec for ``(model, device)``.

        Raises ``ValueError`` when the combination is not in the catalog or
        has no default impl.
        """
        ...

    def resolve_candidates(self, model: str, device: str) -> List[ModelSpecLike]:
        """All specs matching ``(model, device)``, across impls, no policy.

        Pure data access so a caller can apply its own fallback policy when
        :meth:`resolve` raises — e.g. stress-test parameter extraction, which
        historically accepts a non-default impl (with a warning) when no
        default exists. Returns ``[]`` when nothing matches.
        """
        ...

    def load_runtime_spec(self, path: str) -> Optional[ModelSpecLike]:
        """Load a pre-resolved runtime spec JSON written by the launcher.

        Returns ``None`` (caller falls back to :meth:`resolve`) when the
        file is missing or malformed.
        """
        ...


_provider: Optional[ModelSpecProvider] = None


def register_model_spec_provider(provider: ModelSpecProvider) -> None:
    """Install the process-wide model spec provider (called at entry points)."""
    global _provider
    _provider = provider


def get_model_spec_provider() -> ModelSpecProvider:
    """Return the registered provider, lazily defaulting to the TT catalog.

    EXTRACTION SEAM (Phase 2 removal point): the lazy default keeps
    pre-extraction callers working without an explicit registration. Once the
    engine is packaged separately, this fallback disappears and entry points
    must register a provider explicitly.
    """
    global _provider
    if _provider is None:
        from workflows.model_spec_provider import TenstorrentModelSpecProvider

        logger.debug("No ModelSpecProvider registered; using Tenstorrent catalog.")
        _provider = TenstorrentModelSpecProvider()
    return _provider

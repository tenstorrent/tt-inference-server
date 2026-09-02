# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned device catalog seam.

The engine only ever reads a device's ``name`` (for CLI choices, report
metadata, and suite filtering) — vendor hardware taxonomies like
Tenstorrent's ``DeviceTypes`` (N150, T3K, GALAXY, ...) are adapter content.
This module defines the seam:

- :class:`DeviceLike` — an opaque device token exposing ``name``.
- :class:`DeviceCatalog` — who can parse a CLI device string and list the
  valid device names.

The Tenstorrent implementation lives adapter-side and is injected via
:func:`register_device_catalog` at process entry. A lazy Tenstorrent
default is kept for backward compatibility during the extraction; it is
the marked Phase-2 removal point.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DeviceLike(Protocol):
    """Opaque device token the engine carries through to reports."""

    name: str


@runtime_checkable
class DeviceCatalog(Protocol):
    """Resolves CLI device strings without exposing the vendor taxonomy."""

    def device_names(self) -> List[str]:
        """All selectable device names, lowercase (for CLI choices)."""
        ...

    def from_string(self, name: str) -> DeviceLike:
        """Parse a device string into a token. Raises ``ValueError`` if unknown."""
        ...


_catalog: DeviceCatalog | None = None


def register_device_catalog(catalog: DeviceCatalog) -> None:
    """Install the process-wide device catalog (called at entry points)."""
    global _catalog
    _catalog = catalog


def get_device_catalog() -> DeviceCatalog:
    """Return the registered catalog, lazily defaulting to the TT taxonomy.

    EXTRACTION SEAM (Phase 2 removal point): the lazy default keeps
    pre-extraction callers working without an explicit registration. Once the
    engine is packaged separately, this fallback disappears and entry points
    must register a catalog explicitly.
    """
    global _catalog
    if _catalog is None:
        from workflows.device_catalog_provider import TenstorrentDeviceCatalog

        logger.debug("No DeviceCatalog registered; using Tenstorrent taxonomy.")
        _catalog = TenstorrentDeviceCatalog()
    return _catalog

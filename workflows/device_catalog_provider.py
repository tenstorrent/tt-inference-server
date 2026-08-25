# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tenstorrent implementation of the engine's device catalog seam.

Adapts the Tenstorrent hardware taxonomy (``DeviceTypes``) to
:class:`workflow_module.device_catalog.DeviceCatalog` so the engine never
imports vendor device enums directly.
"""

from __future__ import annotations

from typing import List

from workflow_module.device_catalog import DeviceCatalog
from workflows.workflow_types import DeviceTypes


class TenstorrentDeviceCatalog(DeviceCatalog):
    """``DeviceCatalog`` over the Tenstorrent ``DeviceTypes`` enum."""

    def device_names(self) -> List[str]:
        return sorted({d.name.lower() for d in DeviceTypes})

    def from_string(self, name: str) -> DeviceTypes:
        return DeviceTypes.from_string(name)

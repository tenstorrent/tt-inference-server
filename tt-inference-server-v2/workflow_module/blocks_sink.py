# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Stub workflow module — accepts ``Block``s emitted by test_module callers.
Empty for now.
"""

from __future__ import annotations

import logging
from typing import Sequence

from report_module.schema import Block

logger = logging.getLogger(__name__)


def accept_blocks(blocks: Sequence[Block]) -> bool:
    """Accept Blocks from a test caller; stub logs and returns success.

    Returns True so callers can chain on success without special-casing
    the stub. Replace with the real workflow pipeline (render + persist
    + upload) when workflow_module gets fleshed out.
    """
    logger.info("workflow_module: received %d block(s)", len(blocks))
    for b in blocks:
        logger.info(
            "  - kind=%s id=%s targets=%s",
            b.kind,
            b.id,
            dict(b.targets) if b.targets else {},
        )
    return True

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The runner that executes a pre-built list of commands."""

from __future__ import annotations

import logging
from typing import List, Sequence

from .commands import Command, CommandResult

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Thin executor of pre-built commands.

    Iterates the command list, calls ``execute()`` on each, and collects
    results. Has no knowledge of what any command actually does; it stops at the
    first failure and returns that command's return code.
    """

    def __init__(self, commands: Sequence[Command]) -> None:
        self.commands: List[Command] = list(commands)
        self.results: List[CommandResult] = []

    def run(self) -> int:
        for cmd in self.commands:
            logger.info("→ command=%s", cmd.name)
            result = cmd.execute()
            self.results.append(result)
            if not result.succeeded:
                logger.error(
                    "❌ command=%s rc=%d error=%s",
                    cmd.name,
                    result.return_code,
                    result.error,
                )
                return result.return_code
            logger.info("✅ command=%s rc=0", cmd.name)
        return 0


__all__ = ["WorkflowRunner"]

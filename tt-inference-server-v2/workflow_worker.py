#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_V2_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _V2_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from workflow_module import CommandFactory  # noqa: E402
from workflow_runner import _LOG_LEVELS, parse_args  # noqa: E402

logger = logging.getLogger("tt_v2_worker")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=_LOG_LEVELS[args.log_level],
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rc = 0
    for cmd in CommandFactory.build(args):
        result = cmd.execute()
        if not result.succeeded:
            logger.error("❌ %s rc=%d %s", cmd.name, result.return_code, result.error)
            return result.return_code
        logger.info("✅ %s rc=0", cmd.name)
    return rc


if __name__ == "__main__":
    sys.exit(main())

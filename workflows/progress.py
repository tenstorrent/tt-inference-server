# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger("run_log")


def emit_progress(stage, pct, msg, container_info=None):
    """Emit structured DEBUG progress signal (only if enabled)
    
    Args:
        stage: Progress stage name (e.g., "initialization", "setup", "complete")
        pct: Progress percentage (0-100)
        msg: Progress message
        container_info: Optional container information dict
    """
    if os.getenv("TT_PROGRESS_DEBUG") == "1":
        logger.debug("TT_PROGRESS stage=%s pct=%d msg=%s", stage, pct, msg)
    
    # Optional JSON snapshot
    pf = os.getenv("TT_PROGRESS_FILE")
    if pf:
        snap = {
            "stage": stage,
            "progress": pct,
            "message": msg,
            "ts": time.time(),
            "container": container_info or None,
        }
        try:
            Path(pf).parent.mkdir(parents=True, exist_ok=True)
            Path(pf).write_text(json.dumps(snap))
        except Exception:
            pass

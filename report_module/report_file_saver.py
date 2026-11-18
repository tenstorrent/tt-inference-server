# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _write_atomic(content: str, path: Path) -> None:
    """Replace a report only after its new contents have been written and closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # A plain exclusive open, not NamedTemporaryFile: its mode follows the
    # umask like a normal write, instead of a forced 0600.
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "x", encoding="utf-8") as temp:
            temp.write(content)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


class ReportFileSaver:
    @staticmethod
    def write_markdown(content: str, path: Path, strict: bool = False) -> None:
        try:
            _write_atomic(content, path)
            logger.info("Saved markdown to: %s", path)
        except Exception:
            logger.exception("Failed to save markdown to: %s", path)
            if strict:
                raise

    @staticmethod
    def write_json(
        data: Any, path: Path, indent: int = 4, strict: bool = False
    ) -> None:
        try:
            _write_atomic(json.dumps(data, indent=indent, default=str), path)
            logger.info("Saved JSON to: %s", path)
        except Exception:
            logger.exception("Failed to save JSON to: %s", path)
            if strict:
                raise

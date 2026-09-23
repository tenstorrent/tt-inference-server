# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

logger = logging.getLogger(__name__)


def _write_atomic(content: str, path: Path) -> None:
    """Replace a report only after its new contents have been written and closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(content)
        # NamedTemporaryFile creates 0600; keep reports readable as before.
        temp_path.chmod(0o644)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
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

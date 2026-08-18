# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportFileSaver:
    @staticmethod
    def write_markdown(content: str, path: Path, strict: bool = False) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
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
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, default=str)
            logger.info("Saved JSON to: %s", path)
        except Exception:
            logger.exception("Failed to save JSON to: %s", path)
            if strict:
                raise

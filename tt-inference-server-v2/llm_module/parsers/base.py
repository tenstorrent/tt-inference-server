# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Parser interface: raw tool output (dict) -> report ``Block``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

from report_module.schema import Block


class LLMResultParser(ABC):
    """Adapt one LLM perf tool's raw JSON output into a report Block.

    Each concrete parser owns a single ``kind`` and knows the schema of
    that tool's result file. Drivers must not call parsers themselves;
    the runner orchestrates ``driver.run() -> parser.parse()``.
    """

    kind: str

    @abstractmethod
    def parse(self, raw: Mapping[str, Any], *, device: str = "") -> Block:
        """Convert a raw result dict into a single Block for the report."""

    def _wrap_record(self, record: Dict[str, Any]) -> Block:
        """Wrap a flat report record in the canonical Block shape.

        ``data`` carries the report sections only — never a duplicate of
        the envelope fields (``kind``/``model``/``device``/``timestamp``).
        Per-block envelope fields move to ``Block.targets`` so the
        runner can build report-level metadata without hunting them out
        of section data, while the renderer pulls model/device from the
        schema's metadata via its existing fallback in
        :func:`report_module.renderers._resolve_model_device`.
        """
        model = str(record.get("model", ""))
        device = str(record.get("device", ""))
        timestamp = str(record.get("timestamp", ""))
        block_id = _slugify_block_id(model, device)
        section_data = {
            k: v
            for k, v in record.items()
            if k not in ("kind", "model", "device", "timestamp")
        }
        targets: Dict[str, Any] = {}
        if model:
            targets["model"] = model
        if device:
            targets["device"] = device
        if timestamp:
            targets["timestamp"] = timestamp
        return Block(
            kind=self.kind,
            id=block_id or None,
            data=section_data,
            targets=targets,
        )


def _slugify_block_id(model: str, device: str) -> str:
    parts = [p for p in (model, device) if p]
    if not parts:
        return ""
    joined = "_".join(parts)
    return joined.replace("/", "__").replace("\\", "__").replace(" ", "_")

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Accumulate ``Block`` s emitted by ``test_module`` runners and assemble
them into a single :class:`report_module.schema.ReportSchema`.

The dispatch entry points hand each runner's Block to
:func:`accept_blocks` along with a sweep-level envelope (model, device,
generation time). The accumulator stores blocks in insertion order and,
when a sweep finishes, ``build_schema`` produces the unified
``base_schema.json`` whose top-level ``metadata`` carries the envelope —
so the per-block ``Block.targets`` doesn't have to duplicate it.
"""

from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional, Sequence

from report_module.schema import Block, ReportSchema

logger = logging.getLogger(__name__)


class BlockAccumulator:
    """Collect ``Block`` s across a sweep and build a ``ReportSchema``.

    Order of insertion is preserved so the resulting schema's sections
    appear in the order the sweep produced them. The first non-empty
    ``envelope`` passed to :meth:`accept` wins and becomes the schema's
    top-level ``metadata`` — keeping the envelope stable even when later
    accept calls supply different (or no) envelope dicts.
    """

    def __init__(self) -> None:
        self._blocks: List[Block] = []
        self._envelope: dict = {}

    def accept(
        self,
        blocks: Sequence[Block],
        *,
        envelope: Optional[Mapping[str, Any]] = None,
    ) -> None:
        for block in blocks:
            if not isinstance(block, Block):
                raise TypeError(
                    f"BlockAccumulator.accept expected Block, got {type(block).__name__}"
                )
            self._blocks.append(block)
        if envelope:
            for key, value in envelope.items():
                self._envelope.setdefault(key, value)
        logger.info(
            "BlockAccumulator: +%d block(s), total=%d",
            len(blocks),
            len(self._blocks),
        )
        for block in blocks:
            logger.info(
                "  - kind=%s id=%s targets=%s",
                block.kind,
                block.id,
                dict(block.targets) if block.targets else {},
            )

    def clear(self) -> None:
        self._blocks.clear()
        self._envelope.clear()

    @property
    def blocks(self) -> List[Block]:
        return list(self._blocks)

    @property
    def envelope(self) -> dict:
        return dict(self._envelope)

    def build_schema(
        self, metadata: Optional[Mapping[str, Any]] = None
    ) -> ReportSchema:
        """Bundle accumulated Blocks into a :class:`ReportSchema`.

        ``metadata`` overrides the recorded sweep envelope. When absent,
        the envelope passed to :meth:`accept` is used directly. ``report_id``
        is synthesised from ``model_name`` + ``generated_at`` if missing.
        """
        meta: dict = dict(metadata or self._envelope)
        meta.setdefault(
            "report_id", _synthesize_report_id(meta.get("model_name", ""), meta)
        )
        return ReportSchema(metadata=meta, sections=list(self._blocks))


_DEFAULT_ACCUMULATOR = BlockAccumulator()


def accept_blocks(
    blocks: Sequence[Block],
    *,
    envelope: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Append ``blocks`` to the process-global accumulator.

    ``envelope`` records sweep-level metadata (``model_name``, ``device``,
    ``generated_at``) once; subsequent calls with a non-empty envelope
    only fill in keys not yet recorded. Returns ``True`` so callers can
    chain on success.
    """
    _DEFAULT_ACCUMULATOR.accept(blocks, envelope=envelope)
    return True


def get_default_accumulator() -> BlockAccumulator:
    """Return the process-global accumulator used by :func:`accept_blocks`."""
    return _DEFAULT_ACCUMULATOR


def _synthesize_report_id(model_name: str, meta: Mapping[str, Any]) -> str:
    base = (
        model_name.replace("/", "__").replace("\\", "__").replace(" ", "_")
        if model_name
        else "report"
    )
    ts = str(meta.get("generated_at") or "").replace(" ", "_").replace(":", "")
    return f"{base}_{ts}" if ts else base


__all__ = [
    "BlockAccumulator",
    "accept_blocks",
    "get_default_accumulator",
]

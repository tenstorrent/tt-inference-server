# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Re-render the report as a sweep progresses, so a killed run still has one.

A cancelled or timed-out job never reaches ``WorkflowExecution.run``'s report
phase, and the Blocks die in the process-global accumulator with the process
(tt-shield run 35095803588 lost 25 finished benchmark sweep points that way).

:func:`checkpoint_report` writes the same paths as the final report -- the
``report_id`` is synthesised once from the envelope's ``generated_at`` -- so
each render overwrites the last and the end-of-workflow ``generate_report``
replaces it with the graded version. Checkpoints carry
``metadata.report_partial`` (cleared by ``inject_metadata`` when the workflow
finishes) and ``report_blocks``.

``WorkflowExecution.run`` calls this on every ``accept`` (the accumulator's
``on_accept`` hook) with ``prepare=self.inject_metadata``, so a checkpoint has
the same ``workflow`` / ``run_command`` / provenance fields as the final
report -- the exabox merged report and ``exabox_unreported_tests`` key on them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .blocks_sink import BlockAccumulator, get_default_accumulator

if TYPE_CHECKING:
    from report_module.schema import ReportSchema

logger = logging.getLogger(__name__)


def checkpoint_report(
    report_dir: Path,
    *,
    accumulator: Optional[BlockAccumulator] = None,
    prepare: Optional[Callable[["ReportSchema"], None]] = None,
) -> bool:
    """Write a partial report for the Blocks accumulated so far.

    ``report_dir`` must be the directory ``WorkflowExecution.generate_report``
    uses (``Path(ctx.output_path).parent``), so the checkpoint and the final
    report land on the same paths. ``prepare`` edits the schema before it is
    marked partial (``WorkflowExecution.inject_metadata``).

    Returns ``True`` when a report was written, ``False`` when there was nothing
    to write or the attempt failed.
    """
    acc = accumulator or get_default_accumulator()
    blocks = acc.blocks
    if not blocks:
        return False
    # Local: report_module imports workflow_module.engine_types, so a
    # module-scope import closes a cycle through workflow_module/__init__.
    from report_module import ReportGenerator

    try:
        schema = acc.build_schema()
        if prepare is not None:
            prepare(schema)
        # After prepare: inject_metadata writes report_partial=False.
        schema.metadata["report_partial"] = True
        schema.metadata["report_blocks"] = len(blocks)
        result = ReportGenerator().generate(schema, report_dir)
        logger.info(
            "Checkpointed partial report (%d block(s)): %s",
            len(blocks),
            result.json_path,
        )
        return True
    except Exception:
        logger.exception("Partial report checkpoint failed (continuing sweep)")
        return False


__all__ = ["checkpoint_report"]

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Re-render the report as a sweep progresses, so a killed run still has one.

A cancelled or timed-out CI job never reaches ``WorkflowExecution.run``'s report
phase: the Blocks live only in the process-global accumulator and die with the
process. tt-shield run 35095803588 lost 25 completed benchmark sweep points that
way -- the per-point ``vllm bench serve`` JSONs were on the runner's disk, but no
``report_data_*.json`` existed, so the aggregate job found nothing to merge.

:func:`checkpoint_report` writes the report after every sweep point into the SAME
paths the final report uses: ``report_id`` is synthesised once from the envelope's
``generated_at`` (see ``blocks_sink._synthesize_report_id``), so each call
overwrites rather than piling up files, and the end-of-workflow
``generate_report`` replaces the last checkpoint with the graded,
acceptance-annotated version.

A checkpoint is strictly best-effort: a sweep that is producing data must never
die because a partial render failed, so nothing here raises -- callers get a bool.

The two markers a checkpoint leaves in ``metadata`` are what tells a consumer it
is looking at an interrupted run:

``report_partial``
    ``True`` in every checkpoint; set ``False`` by ``WorkflowExecution``'s
    ``inject_metadata`` on the final report. A report on disk still carrying
    ``True`` means the workflow never finished.
``report_blocks``
    how many Blocks the checkpoint holds, so a reader can see how far the sweep
    got without counting sections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .blocks_sink import BlockAccumulator, get_default_accumulator

logger = logging.getLogger(__name__)


def checkpoint_report(
    report_dir: Path, *, accumulator: Optional[BlockAccumulator] = None
) -> bool:
    """Write a partial report for the Blocks accumulated so far.

    ``report_dir`` must be the directory ``WorkflowExecution.generate_report``
    uses (``Path(ctx.output_path).parent``), so the checkpoint and the final
    report land on the same paths.

    Returns ``True`` when a report was written, ``False`` when there was nothing
    to write or the attempt failed.
    """
    acc = accumulator or get_default_accumulator()
    blocks = acc.blocks
    if not blocks:
        return False
    # Imported here, not at module scope: report_module imports
    # workflow_module.engine_types, so a top-level import would close a cycle
    # through workflow_module/__init__.
    from report_module import ReportGenerator

    try:
        schema = acc.build_schema()
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
        # Never let a partial render take down a sweep that is still producing
        # data -- the next sweep point will try again anyway.
        logger.exception("Partial report checkpoint failed (continuing sweep)")
        return False


__all__ = ["checkpoint_report"]

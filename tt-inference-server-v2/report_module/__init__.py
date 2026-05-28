# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from report_module.acceptance_criteria import (
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)
from report_module.generator import (
    GenerateResult,
    ReportGenerator,
    generate_report,
)
from report_module.report_file_saver import ReportFileSaver
from report_module.schema import Block, ReportSchema, SchemaLike
from report_module.spec_decode_pairing import (
    SPEC_DECODE_BLOCK_KIND,
    SPEC_DECODE_PAIR_BLOCK_KIND,
    compute_speedup,
    pair_baseline_spec,
)

__all__ = [
    "Block",
    "GenerateResult",
    "ReportFileSaver",
    "ReportGenerator",
    "ReportSchema",
    "SchemaLike",
    "SPEC_DECODE_BLOCK_KIND",
    "SPEC_DECODE_PAIR_BLOCK_KIND",
    "acceptance_criteria_check",
    "compute_speedup",
    "format_acceptance_summary_markdown",
    "generate_report",
    "pair_baseline_spec",
]

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from .classification import (
    classify_exception,
    fault_report_from_exception,
    format_fault_log_line,
    log_fault_report,
)
from .fault_types import (
    BoundaryError,
    ClassificationResult,
    FaultOrigin,
    FaultReport,
)
from .wrap import external_call_boundary, wrap_external_call

__all__ = [
    "BoundaryError",
    "ClassificationResult",
    "FaultOrigin",
    "FaultReport",
    "classify_exception",
    "external_call_boundary",
    "fault_report_from_exception",
    "format_fault_log_line",
    "log_fault_report",
    "wrap_external_call",
]

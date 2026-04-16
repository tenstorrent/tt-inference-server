# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FaultOrigin(str, Enum):
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"
    BOUNDARY = "BOUNDARY"
    UNKNOWN = "UNKNOWN"


COMPONENT_INFERENCE = "inference"
COMPONENT_TTNN = "ttnn"
COMPONENT_TORCH = "torch"


@dataclass(frozen=True)
class ClassificationResult:
    origin: FaultOrigin
    component: str
    reason: Optional[str] = None


@dataclass
class FaultReport:
    origin: FaultOrigin
    component: str
    operation: str
    exc_type: str
    message: str
    request_id: Optional[str] = None
    tt_metal_commit: Optional[str] = None
    image_tag: Optional[str] = None
    classification_reason: Optional[str] = None


class BoundaryError(Exception):
    """Raised when a classified fault crosses the inference↔metal boundary."""

    def __init__(self, fault_report: FaultReport):
        self.fault_report = fault_report
        super().__init__(
            f"{fault_report.origin.value} {fault_report.component} "
            f"{fault_report.operation}: {fault_report.exc_type}: {fault_report.message}"
        )

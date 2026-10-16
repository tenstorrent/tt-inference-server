# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from enum import Enum


class WorkerReplacementOutcome(str, Enum):
    REPLACED = "replaced"
    ASSIGNMENT_RELEASED = "assignment_released"
    WORKER_MISMATCH = "worker_mismatch"
    RETRY_REQUIRED = "retry_required"

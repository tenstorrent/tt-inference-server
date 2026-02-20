# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from enum import IntEnum


class PerformanceResult(IntEnum):
    """Accuracy/performance check result for benchmark and eval reports.

    Values match existing report schema (0/2/3) used by workflow parsers.
    """

    UNDEFINED = 0
    PASS = 2
    FAIL = 3

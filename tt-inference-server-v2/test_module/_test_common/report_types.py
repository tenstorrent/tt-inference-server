# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from enum import IntEnum, auto


class ReportCheckTypes(IntEnum):
    NA = auto()
    PASS = auto()
    FAIL = auto()

    @classmethod
    def from_result(cls, result: bool | None) -> "ReportCheckTypes":
        res_map = {
            None: ReportCheckTypes.NA,
            True: ReportCheckTypes.PASS,
            False: ReportCheckTypes.FAIL,
        }
        return res_map[result]

    @classmethod
    def to_display_string(cls, check_type: "ReportCheckTypes") -> str:
        disp_map = {
            ReportCheckTypes.NA: "N/A",
            ReportCheckTypes.PASS: "PASS ✅",
            ReportCheckTypes.FAIL: "FAIL ⛔",
        }
        return disp_map[check_type]


__all__ = ["ReportCheckTypes"]

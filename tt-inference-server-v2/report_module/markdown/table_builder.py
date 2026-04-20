# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Unified markdown table builder extracted from benchmarking.summary_report and
stress_tests.stress_tests_summary_report.

Provides unicode-aware column alignment, decimal alignment for numeric columns,
and header-description footnotes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List

logger = logging.getLogger(__name__)

EXPLANATION_MAP: Dict[str, str] = {
    "ISL": "Input Sequence Length (tokens)",
    "OSL": "Output Sequence Length (tokens)",
    "Concurrency": "number of concurrent requests (batch size)",
    "N Req": "total number of requests (sample size, N)",
    "TTFT": "Time To First Token (ms)",
    "TPOT": "Time Per Output Token (ms)",
    "Tput User": "Throughput per user (TPS)",
    "Tput Decode": "Throughput for decode tokens, across all users (TPS)",
    "Tput Prefill": "Throughput for prefill tokens (TPS)",
    "E2EL": "End-to-End Latency (ms)",
    "Req Tput": "Request Throughput (RPS)",
}


def sanitize_cell(text: str) -> str:
    text = str(text).replace("|", "\\|").replace("\n", " ")
    return text.strip()


def _cell_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def wcswidth(text: str) -> int:
    """Return the number of monospace columns *text* will occupy."""
    return sum(_cell_width(ch) for ch in text)


def pad_right(text: str, width: int) -> str:
    return text + " " * max(width - wcswidth(text), 0)


def pad_left(text: str, width: int) -> str:
    return " " * max(width - wcswidth(text), 0) + text


def pad_center(text: str, width: int) -> str:
    total = width - wcswidth(text)
    left = total // 2
    return " " * max(left, 0) + text + " " * max(total - left, 0)


def get_markdown_table(
    display_dicts: List[Dict[str, str]],
    include_notes: bool = True,
) -> str:
    """Build a GitHub-flavoured markdown table from a list of display dicts.

    Each dict maps column header -> cell value (all strings).  Numeric columns
    are decimal-aligned; text columns are left-aligned.

    When *include_notes* is False the benchmark-specific footer (metric means
    disclaimer and header explanations) is omitted — useful for non-benchmark
    tables such as test reports.
    """
    if not display_dicts:
        return ""

    headers = list(display_dicts[0].keys())

    numeric_cols = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(d.get(header, "")).strip())
            for d in display_dicts
        )
        for header in headers
    }

    max_left: Dict[str, int] = {}
    max_right: Dict[str, int] = {}
    for header in headers:
        max_left[header] = max_right[header] = 0
        if numeric_cols[header]:
            for d in display_dicts:
                val = str(d.get(header, "")).strip()
                left, _, right = val.partition(".")
                max_left[header] = max(max_left[header], len(left))
                max_right[header] = max(max_right[header], len(right))

    def _format_numeric(val: str, header: str) -> str:
        left, _, right = val.partition(".")
        left = left.rjust(max_left[header])
        if max_right[header] > 0:
            right = right.ljust(max_right[header])
            return f"{left}.{right}"
        return left

    col_widths: Dict[str, int] = {}
    for header in headers:
        if numeric_cols[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            col_widths[header] = max(wcswidth(header), numeric_width)
        else:
            max_content = max(
                wcswidth(sanitize_cell(str(d.get(header, "")))) for d in display_dicts
            )
            col_widths[header] = max(wcswidth(header), max_content)

    header_row = (
        "| "
        + " | ".join(
            pad_center(sanitize_cell(header), col_widths[header]) for header in headers
        )
        + " |"
    )

    separator_row = (
        "|" + "|".join("-" * (col_widths[header] + 2) for header in headers) + "|"
    )

    value_rows = []
    for d in display_dicts:
        cells = []
        for header in headers:
            raw = sanitize_cell(str(d.get(header, "")).strip())
            if numeric_cols[header]:
                num = _format_numeric(raw, header)
                cell = pad_left(num, col_widths[header])
            else:
                cell = pad_right(raw, col_widths[header])
            cells.append(cell)
        value_rows.append("| " + " | ".join(cells) + " |")

    table_str = "\n".join([header_row, separator_row] + value_rows)

    if not include_notes:
        return table_str

    end_notes = "\n\nNote: all metrics are means across benchmark run unless otherwise stated.\n"

    def _clean_header(h: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", h).strip()

    key_list = [_clean_header(k) for k in headers]
    explain_str = "\n".join(
        f"> {key}: {EXPLANATION_MAP[key]}" for key in key_list if key in EXPLANATION_MAP
    )

    return table_str + end_notes + explain_str

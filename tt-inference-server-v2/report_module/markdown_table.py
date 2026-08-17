# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

_NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def sanitize_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _cell_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def wcswidth(text: str) -> int:
    return sum(_cell_width(ch) for ch in text)


def _pad_right(text: str, width: int) -> str:
    return text + " " * max(width - wcswidth(text), 0)


def _pad_left(text: str, width: int) -> str:
    return " " * max(width - wcswidth(text), 0) + text


def _pad_center(text: str, width: int) -> str:
    total = width - wcswidth(text)
    left = total // 2
    return " " * max(left, 0) + text + " " * max(total - left, 0)


def build_markdown_table(display_dicts: List[Dict[str, str]]) -> str:
    """Emit a markdown table from a list of header→cell dicts."""
    if not display_dicts:
        return ""

    headers = list(display_dicts[0].keys())

    numeric_cols = {
        header: all(
            _NUMERIC_PATTERN.match(str(d.get(header, "")).strip())
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
        left, has_dot, right = val.partition(".")
        left = left.rjust(max_left[header])
        if max_right[header] == 0:
            return left
        sep = "." if has_dot else " "
        return f"{left}{sep}{right.ljust(max_right[header])}"

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
            _pad_center(sanitize_cell(header), col_widths[header]) for header in headers
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
                cell = _pad_left(num, col_widths[header])
            else:
                cell = _pad_right(raw, col_widths[header])
            cells.append(cell)
        value_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, separator_row] + value_rows)

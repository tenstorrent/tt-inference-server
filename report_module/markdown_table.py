# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import unicodedata
from typing import Dict, List


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


def build_markdown_table(display_dicts: List[Dict[str, str]]) -> str:
    """Emit a markdown table with every column left-aligned."""
    if not display_dicts:
        return ""

    headers = list(display_dicts[0].keys())

    cells_by_row = [
        {header: sanitize_cell(str(d.get(header, ""))) for header in headers}
        for d in display_dicts
    ]

    col_widths: Dict[str, int] = {}
    for header in headers:
        max_content = max(wcswidth(row[header]) for row in cells_by_row)
        col_widths[header] = max(wcswidth(sanitize_cell(header)), max_content)

    header_row = (
        "| "
        + " | ".join(
            _pad_right(sanitize_cell(header), col_widths[header]) for header in headers
        )
        + " |"
    )

    # ``:---`` per column requests left alignment from the renderer.
    separator_row = (
        "|" + "|".join(":" + "-" * (col_widths[header] + 1) for header in headers) + "|"
    )

    value_rows = [
        "| "
        + " | ".join(_pad_right(row[header], col_widths[header]) for header in headers)
        + " |"
        for row in cells_by_row
    ]

    return "\n".join([header_row, separator_row] + value_rows)

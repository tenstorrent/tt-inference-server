# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.markdown_table`` rendering + alignment."""

from __future__ import annotations

from report_module.markdown_table import (
    build_markdown_table,
    sanitize_cell,
    wcswidth,
)


def test_sanitize_escapes_pipes_and_flattens_newlines():
    assert sanitize_cell("a|b\nc ") == "a\\|b c"


def test_wcswidth_counts_wide_and_combining_chars():
    # ASCII is width 1, CJK is width 2, combining accent is width 0.
    assert wcswidth("ab") == 2
    assert wcswidth("世界") == 4
    assert wcswidth("é") == 1  # 'e' + combining acute accent


def test_empty_input_returns_empty_string():
    assert build_markdown_table([]) == ""


def test_basic_table_has_header_separator_and_rows():
    table = build_markdown_table([{"Name": "a", "Val": "1"}, {"Name": "bb", "Val": "2"}])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert lines[0].startswith("|") and lines[0].endswith("|")
    assert set(lines[1]) <= {"|", "-"}


def test_numeric_columns_align_on_decimal_point():
    table = build_markdown_table(
        [{"Metric": "10.5"}, {"Metric": "1.25"}, {"Metric": "100"}]
    )
    # Single-column rows: the cell is everything between the bordering pipes.
    cells = [r.split("|")[1] for r in table.splitlines()[2:]]
    # The integer part is right-justified, so the decimal points line up.
    assert cells[0].index(".") == cells[1].index(".")
    # All cells share one width (the column is rectangular).
    assert len({len(c) for c in cells}) == 1


def test_non_numeric_column_left_aligned():
    table = build_markdown_table([{"Name": "ab"}, {"Name": "abcd"}])
    rows = [r for r in table.splitlines()[2:]]
    # Left-aligned: value starts immediately after "| ".
    assert rows[0].startswith("| ab")
    assert rows[1].startswith("| abcd")


def test_header_width_governs_when_wider_than_cells():
    table = build_markdown_table([{"LongHeader": "x"}])
    header = table.splitlines()[0]
    # Column is at least as wide as the header text.
    assert "LongHeader" in header


def test_pipe_in_cell_is_escaped_in_output():
    table = build_markdown_table([{"Col": "a|b"}])
    assert "a\\|b" in table

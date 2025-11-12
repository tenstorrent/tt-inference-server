#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# CLI to parse a release report JSON and print a filtered markdown table
# using only Python stdlib.

import argparse
import json
import sys
from datetime import datetime, date
from typing import Any, Dict, Iterable, List, Optional, Tuple


COLUMNS = [
    "max_con",
    "num_prompts",
    "mean_ttft_ms",
    "std_ttft_ms",
    "mean_tps",
    "std_tps",
    "tps_decode_throughput",
    "tps_prefill_throughput",
    "mean_e2el_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render markdown table from release report JSON benchmarks."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to report_data_*.json (from workflows/run_reports.py)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=128,
        help="Filter: input_sequence_length (default: 128)",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=128,
        help="Filter: output_sequence_length (default: 128)",
    )
    parser.add_argument(
        "--after",
        type=str,
        default="2025-11-10",
        help="Filter: include rows with timestamp strictly after this YYYY-MM-DD date (default: 2025-11-10)",
    )
    parser.add_argument(
        "--max-con",
        type=int,
        default=None,
        help="Optional filter: max_con (concurrency) must equal this value if provided",
    )
    return parser.parse_args()


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() == "n/a":
        return None
    # Remove common formatting artifacts
    s = s.replace(",", "")
    try:
        if "." in s:
            return float(s)
        return float(int(s))
    except ValueError:
        return None


def _parse_timestamp(ts: str) -> Optional[datetime]:
    # Expected primary format from benchmarking/summary_report: YYYY-MM-DD_HH-MM-SS
    # Fallbacks: ISO-like "YYYY-MM-DD HH:MM:SS", possibly with 'Z'
    if not ts:
        return None
    ts = ts.strip().replace("Z", "")
    fmts = ["%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    # Determine column widths (consider headers and rows)
    widths: List[int] = []
    for col_idx, header in enumerate(headers):
        max_len = len(header)
        for r in rows:
            if col_idx < len(r):
                max_len = max(max_len, len(r[col_idx]))
        widths.append(max_len)

    # Determine which columns are numeric (all rows numeric or 'n/a')
    numeric_cols: List[bool] = []
    for col_idx in range(len(headers)):
        all_numeric = True
        for r in rows:
            cell = r[col_idx].strip()
            if cell.lower() == "n/a":
                continue
            if not _is_numeric_str(cell):
                all_numeric = False
                break
        numeric_cols.append(all_numeric)

    # Build header row (left-aligned)
    header_cells = []
    for i, h in enumerate(headers):
        header_cells.append(h.ljust(widths[i]))
    header_row = "| " + " | ".join(header_cells) + " |"

    # Separator row with exact widths (min 3 dashes) and matching spacing
    sep_cells = []
    for w in widths:
        sep_cells.append("-" * max(3, w))
    sep_row = "| " + " | ".join(sep_cells) + " |"

    # Build body rows with alignment
    body = []
    for r in rows:
        cells = []
        for i, cell in enumerate(r):
            if numeric_cols[i]:
                cells.append(cell.rjust(widths[i]))
            else:
                cells.append(cell.ljust(widths[i]))
        body.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, sep_row] + body)


def _format_cell(value: Optional[float], column_name: str) -> str:
    if value is None:
        return "n/a"
    # Integer display for count columns
    if column_name in ("max_con", "num_prompts"):
        try:
            return str(int(round(float(value))))
        except Exception:
            return "n/a"
    # Two decimal places for metrics
    return f"{float(value):.2f}"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _extract_filtered_rows(
    report: Dict[str, Any],
    isl: int,
    osl: int,
    after_date: date,
    max_con: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = report.get("benchmarks")
    if not isinstance(rows, list):
        raise ValueError('Expected "benchmarks" to be a list in the JSON file.')

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        # sequence length filters
        row_isl = _coerce_number(row.get("input_sequence_length"))
        row_osl = _coerce_number(row.get("output_sequence_length"))
        if row_isl is None or row_osl is None:
            continue
        if int(row_isl) != isl or int(row_osl) != osl:
            continue

        # optional max_con filter
        if max_con is not None:
            row_max_con = _coerce_number(row.get("max_con"))
            if row_max_con is None or int(row_max_con) != int(max_con):
                continue

        # timestamp filter (strictly after)
        ts = str(row.get("timestamp", "")).strip()
        dt = _parse_timestamp(ts)
        if dt is None:
            continue
        if not (dt.date() > after_date):
            continue

        filtered.append(row)

    return filtered


def _project_row(row: Dict[str, Any]) -> List[Optional[float]]:
    # Return columns in COLUMNS order, coercing to numbers where applicable
    out: List[Optional[float]] = []
    for key in COLUMNS:
        out.append(_coerce_number(row.get(key)))
    return out


def _stringify_row(row: List[Optional[float]], columns: List[str]) -> List[str]:
    return [_format_cell(row[i], columns[i]) for i in range(len(columns))]


def _compute_means(rows: List[List[Optional[float]]]) -> List[Optional[float]]:
    # Column-wise means
    transposed: List[List[Optional[float]]] = [[] for _ in range(len(COLUMNS))]
    for r in rows:
        for i, val in enumerate(r):
            transposed[i].append(val)
    return [_mean(col) for col in transposed]


def generate_markdown_from_benchmarks(
    json_path: str, isl: int, osl: int, after_str: str, max_con: Optional[int] = None
) -> Tuple[str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    try:
        after_date = datetime.strptime(after_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("--after must be in YYYY-MM-DD format")

    filtered_rows = _extract_filtered_rows(
        report, isl=isl, osl=osl, after_date=after_date, max_con=max_con
    )
    if not filtered_rows:
        raise ValueError("No rows match the provided filters.")

    numeric_rows = [_project_row(r) for r in filtered_rows]
    display_rows = [_stringify_row(r, COLUMNS) for r in numeric_rows]

    # Add ISL/OSL as the first two columns with values in each row
    headers_with_ctx = ["ISL", "OSL"] + COLUMNS
    display_rows_with_ctx = [[str(isl), str(osl)] + r for r in display_rows]

    table = _to_markdown_table(headers_with_ctx, display_rows_with_ctx)

    means = _compute_means(numeric_rows)
    # Round mean for integer columns
    means_for_display: List[Optional[float]] = []
    for i, v in enumerate(means):
        if COLUMNS[i] in ("max_con", "num_prompts"):
            means_for_display.append(float(int(round(v))) if v is not None else None)
        else:
            means_for_display.append(v)
    mean_row = _stringify_row(means_for_display, COLUMNS)
    mean_table = _to_markdown_table(headers_with_ctx, [[str(isl), str(osl)] + mean_row])

    return table, mean_table


def main() -> int:
    args = parse_args()
    try:
        table, mean_table = generate_markdown_from_benchmarks(
            json_path=args.json,
            isl=args.isl,
            osl=args.osl,
            after_str=args.after,
            max_con=args.max_con,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(table)
    print()
    print("Averages")
    print(mean_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())




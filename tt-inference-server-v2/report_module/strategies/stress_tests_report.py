# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Stress test report strategy.

Consumes ``stress_test_<model_id>_*.json`` files produced by the
``stress_tests`` workflow (see ``stress_tests/run_stress_tests.py``) and
emits the ``stress_tests`` section of the release bundle.

The strategy mirrors the four steps performed by
``workflows/run_reports.py::stress_test_generate_report``:

1. Discover ``stress_test_<model_id>_*.json`` under
   ``<workflow_log_dir>/stress_tests_output/``.
2. Parse each JSON via
   :func:`report_module.parsing.stress_test_parser.process_stress_test_files`
   into a list of normalised metric dicts (``release_raw``).
3. Render a markdown table — simple (means only) or detailed
   (per-metric percentiles) based on ``context.percentile_report``.
4. Return a :class:`ReportResult` so the aggregator can place the raw
   list under the ``stress_tests`` key of the release JSON and the file
   saver can persist the per-strategy markdown.
"""

from __future__ import annotations

import logging
from glob import glob
from typing import Any, Dict, List, Tuple

from report_module.base_strategy import ReportStrategy
from report_module.markdown.table_builder import get_markdown_table
from report_module.parsing.stress_test_parser import process_stress_test_files
from report_module.types import NOT_MEASURED_STR, ReportContext, ReportResult

logger = logging.getLogger(__name__)

_STRESS_TESTS_DIRNAME = "stress_tests_output"

ColumnSpec = Tuple[str, str, int]

_CONFIG_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("input_sequence_length", "ISL", 0),
    ("output_sequence_length", "OSL", 0),
    ("max_con", "Concurrency", 0),
    ("num_prompts", "Num Prompts", 0),
)

_SIMPLE_METRIC_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("mean_ttft_ms", "TTFT (ms)", 1),
    ("mean_tpot_ms", "TPOT (ms)", 1),
    ("mean_itl_ms", "ITL (ms)", 1),
    ("mean_e2el_ms", "E2EL (ms)", 1),
)

_PERCENTILE_METRIC_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("ttft", "TTFT"),
    ("tpot", "TPOT"),
    ("itl", "ITL"),
    ("e2el", "E2EL"),
)

_PERCENTILE_SUFFIXES: Tuple[str, ...] = ("mean", "p5", "p25", "p50", "p95", "p99")

_THROUGHPUT_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("mean_tps", "Tput User (TPS)", 2),
    ("tps_decode_throughput", "Tput Decode (TPS)", 1),
)


def _build_detailed_metric_columns() -> Tuple[ColumnSpec, ...]:
    """Expand the percentile metric grid into per-column specs.

    Produces, in order, mean / p5 / p25 / p50 / p95 / p99 for each of
    TTFT, TPOT, ITL, E2EL — matching ``generate_stress_tests_markdown_table_detailed``
    in ``workflows/run_reports.py``.
    """
    columns: List[ColumnSpec] = []
    for metric_key, metric_label in _PERCENTILE_METRIC_GROUPS:
        for suffix in _PERCENTILE_SUFFIXES:
            if suffix == "mean":
                data_key = f"mean_{metric_key}_ms"
                header = f"{metric_label} (ms)"
            else:
                data_key = f"{suffix}_{metric_key}_ms"
                header = f"{suffix.upper()} {metric_label} (ms)"
            columns.append((data_key, header, 1))
    return tuple(columns)


_DETAILED_METRIC_COLUMNS: Tuple[ColumnSpec, ...] = _build_detailed_metric_columns()


def _format_cell(value: Any, decimals: int) -> str:
    if value is None or value == "" or value == NOT_MEASURED_STR:
        return NOT_MEASURED_STR
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:  # NaN guard
        return NOT_MEASURED_STR
    if decimals == 0:
        return str(int(numeric))
    return f"{numeric:.{decimals}f}"


def _render_table(
    release_raw: List[Dict[str, Any]],
    columns: Tuple[ColumnSpec, ...],
) -> str:
    display_dicts: List[Dict[str, str]] = []
    for row in release_raw:
        display_dicts.append(
            {header: _format_cell(row.get(data_key), decimals)
             for data_key, header, decimals in columns}
        )
    return get_markdown_table(display_dicts)


class StressTestsStrategy(ReportStrategy):
    """Release-JSON strategy for the ``stress_tests`` section."""

    name = "stress_tests"

    def is_applicable(self, context: ReportContext) -> bool:
        return bool(self._discover_files(context))

    def generate(self, context: ReportContext) -> Dict[str, ReportResult]:
        files = self._discover_files(context)
        if not files:
            return {self.name: ReportResult.empty(self.name)}

        logger.info(f"Stress tests: processing {len(files)} file(s)")

        try:
            release_raw = process_stress_test_files(files)
        except ValueError:
            logger.exception("Stress tests: no files could be parsed")
            return {self.name: ReportResult.empty(self.name)}

        columns = (
            _CONFIG_COLUMNS + _DETAILED_METRIC_COLUMNS + _THROUGHPUT_COLUMNS
            if context.percentile_report
            else _CONFIG_COLUMNS + _SIMPLE_METRIC_COLUMNS + _THROUGHPUT_COLUMNS
        )
        logger.info(
            "Stress tests: generating "
            f"{'detailed percentile' if context.percentile_report else 'simple'} "
            "report"
        )

        table_md = _render_table(release_raw, columns)
        release_markdown = (
            f"### Stress Test Results for {context.model_name} "
            f"on {context.device_str}\n\n{table_md}"
        )

        return {
            self.name: ReportResult(
                name=self.name,
                markdown=release_markdown,
                data=release_raw,
                md_filename=f"stress_test_summary_{context.report_id}.md",
            )
        }

    @staticmethod
    def _discover_files(context: ReportContext) -> List[str]:
        pattern = (
            f"{context.workflow_log_dir}/{_STRESS_TESTS_DIRNAME}/"
            f"stress_test_{context.model_spec.model_id}_*.json"
        )
        return sorted(glob(pattern))

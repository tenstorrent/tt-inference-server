# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from report_module.report_file_saver import ReportFileSaver
from report_module.report_generator import ReportGenerator
from report_module.strategies import (
    AiPerfStrategy,
    GenAiPerfStrategy,
    ParameterSupportTestsStrategy,
    StandardReportStrategy,
    StressTestsStrategy,
    TestReportStrategy,
)
from report_module.types import ReportRequest, ReportResult, build_context

logger = logging.getLogger(__name__)

ALL_STRATEGIES = [
    StandardReportStrategy,
    AiPerfStrategy,
    GenAiPerfStrategy,
    TestReportStrategy,
    ParameterSupportTestsStrategy,
    StressTestsStrategy,
]

STRATEGY_NAME_MAP = {cls.name: cls for cls in ALL_STRATEGIES}

SUB_SECTION_TO_STRATEGY = {
    section: StandardReportStrategy
    for section in StandardReportStrategy.SUB_SECTIONS
}

ALL_VALID_NAMES = set(STRATEGY_NAME_MAP.keys()) | set(SUB_SECTION_TO_STRATEGY.keys())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model release reports")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        required=True,
        help="Path to the runtime model specification JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Root directory for report output",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="",
        help="Device override (uses value from model spec JSON if omitted)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="",
        help="Model name override (uses value from model spec JSON if omitted)",
    )
    parser.add_argument(
        "--reports",
        type=str,
        nargs="*",
        default=[],
        help=f"Run only the listed report strategies or sub-sections. Available: {sorted(ALL_VALID_NAMES)}",
    )
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> ReportRequest:
    return ReportRequest(
        output_path=args.output_path,
        runtime_model_spec_json=args.runtime_model_spec_json,
        device=args.device,
        model=args.model,
        selected_reports=args.reports or [],
    )


def select_strategies(selected: List[str]) -> List:
    if not selected:
        return [cls() for cls in ALL_STRATEGIES]

    unknown = [name for name in selected if name not in ALL_VALID_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown report names: {unknown}. "
            f"Available: {sorted(ALL_VALID_NAMES)}"
        )

    strategy_classes = []
    seen = set()
    for name in selected:
        if name in STRATEGY_NAME_MAP:
            cls = STRATEGY_NAME_MAP[name]
        else:
            cls = SUB_SECTION_TO_STRATEGY[name]

        if cls not in seen:
            seen.add(cls)
            strategy_classes.append(cls)

    return [cls() for cls in strategy_classes]


def run(request: ReportRequest) -> Dict[str, ReportResult]:
    """Programmatic entry point — build context, run strategies, save output."""
    context = build_context(request)
    strategies = select_strategies(request.selected_reports)
    generator = ReportGenerator(strategies=strategies, file_saver=ReportFileSaver())

    logger.info(
        f"Running {len(strategies)} strategies for "
        f"{context.model_name} on {context.device_str}"
    )

    results = generator.generate_and_save_release(context)

    succeeded = sum(1 for r in results.values() if r.markdown)
    logger.info(f"Report complete: {succeeded}/{len(strategies)} strategies produced output")
    return results


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args = _parse_args()
    request = build_request(args)

    try:
        run(request)
    except Exception:
        logger.exception("Report generation failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

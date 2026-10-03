# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec-test wrapper for the tool-call JSON-schema conformance suite.

Runs ``llm_module/test_tool_call_json_schema.py`` in a child pytest process
(same mechanics as :class:`VLLMParamConformanceTest`) and grades it on a pass
rate rather than all-or-nothing: the suite sends 408 tool-call requests whose
arguments must validate against walle JSON Schemas, and a model that gets
>= ``pass_rate_threshold`` (default 99.5%) of them right passes.

Status: PASS when ``pass_rate >= threshold``, FAIL below it, ERROR when the
child pytest produced no report, ran no cases, or never reached the server
(every case a connection error).
"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .vllm_param_conformance_test import (
    FAILED_STATUS,
    PASSED_STATUS,
    VLLMParamConformanceTest,
)

logger = logging.getLogger(__name__)

DEFAULT_PASS_RATE_THRESHOLD = 0.995
THRESHOLD_KEY = "pass_rate_threshold"
# Must match test_tool_call_json_schema.REPORT_RESULTS_KEY (not imported: that
# would pull llm_module into the orchestrator's import chain).
REPORT_RESULTS_KEY = "test_tool_call_json_schema"
CONNECTION_ERROR_CAUSE = "connection_error"
UNKNOWN_CAUSE = "unknown"

# test_config key -> (pytest option, kind). Unset keys keep the suite defaults
# in llm_module/tool_call_schema.py.
_PYTEST_OPTIONS = (
    ("tool_choice", "--schema-tool-choice", "value"),
    ("think_mode", "--schema-think-mode", "value"),
    ("thinking", "--schema-thinking", "flag"),
    ("case_selection", "--schema-selection", "value"),
    ("max_cases", "--schema-max-cases", "value"),
    ("max_tokens", "--schema-max-tokens", "value"),
    ("case_retries", "--schema-retries", "value"),
    ("workers", "--schema-workers", "value"),
    ("request_timeout", "--schema-request-timeout", "value"),
)


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


def grade_tool_call_schema_results(
    cases: Sequence[Mapping[str, Any]], threshold: float
) -> Dict[str, Any]:
    """Pass-rate summary of per-case records from the suite's report.

    ``success`` is ``pass_rate >= threshold``. Failures are counted by cause
    and listed with their full messages.
    """
    total = len(cases)
    passed = sum(1 for c in cases if c.get("status") == PASSED_STATUS)
    failed_cases = [c for c in cases if c.get("status") != PASSED_STATUS]
    pass_rate = passed / total if total else 0.0

    by_mode: Dict[str, Counter] = {}
    for case in cases:
        counts = by_mode.setdefault(str(case.get("mode", "")), Counter())
        counts["total"] += 1
        counts["passed"] += case.get("status") == PASSED_STATUS

    causes = Counter(str(c.get("cause") or UNKNOWN_CAUSE) for c in failed_cases)

    return {
        "total": total,
        "passed": passed,
        "failed": len(failed_cases),
        "pass_rate": pass_rate,
        "pass_rate_percent": _percent(pass_rate),
        "threshold": threshold,
        "threshold_percent": _percent(threshold),
        "passed_after_retry": sum(
            1
            for c in cases
            if c.get("status") == PASSED_STATUS and (c.get("attempts") or 1) > 1
        ),
        "results_by_mode": [
            {
                "mode": mode,
                "total": counts["total"],
                "passed": counts["passed"],
                "failed": counts["total"] - counts["passed"],
                "pass_rate_percent": _percent(counts["passed"] / counts["total"]),
            }
            for mode, counts in sorted(by_mode.items())
        ],
        "failures_by_cause": [
            {"cause": cause, "count": count} for cause, count in causes.most_common()
        ],
        "failed_cases": [
            {
                "case": c.get("case_id"),
                "mode": c.get("mode"),
                "selection_reason": c.get("selection_reason"),
                "cause": c.get("cause") or UNKNOWN_CAUSE,
                "attempts": c.get("attempts"),
                "http_status": c.get("http_status"),
                "message": c.get("message", ""),
            }
            for c in sorted(
                failed_cases,
                key=lambda c: (str(c.get("case_id")), str(c.get("mode"))),
            )
        ],
        "success": total > 0 and pass_rate >= threshold,
    }


class ToolCallSchemaConformanceTest(VLLMParamConformanceTest):
    """Grade the tool-call JSON-schema suite on a pass-rate threshold."""

    KIND = "tool_call_json_schema"
    # "functional" so acceptance_criteria counts it (see VLLMParamConformanceTest).
    TASK_TYPE = "functional"

    PYTEST_FILENAME = "test_tool_call_json_schema.py"
    ENDPOINT_PATH = "/v1/chat/completions"
    REPORT_TASK_NAME = "tool_call_json_schema"

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        threshold = self._resolve_threshold()
        endpoint_url = f"{self.base_url}{self.ENDPOINT_PATH}"
        model_name = self._resolve_model_name()

        with tempfile.TemporaryDirectory(prefix="tool_call_schema_") as tmp_dir:
            report = await self._run_pytest_suite(tmp_dir, endpoint_url, model_name)

        cases = (report.get("results") or {}).get(REPORT_RESULTS_KEY) or []
        self._raise_if_not_graded(cases)

        graded = grade_tool_call_schema_results(cases, threshold)
        logger.info(
            "Tool-call schema pass rate %s (%d/%d) vs threshold %s -> %s",
            graded["pass_rate_percent"],
            graded["passed"],
            graded["total"],
            graded["threshold_percent"],
            "PASS" if graded["success"] else "FAIL",
        )
        # Surface the gate next to the measurement on the Block's targets
        # (copied: the dict passed in belongs to the suite definition).
        self.targets = {
            **self.targets,
            THRESHOLD_KEY: threshold,
            "pass_rate": graded["pass_rate"],
        }
        return {
            "endpoint_url": report.get("endpoint_url", endpoint_url),
            "model_name": report.get("model_name", model_name),
            "task_name": report.get("task_name", self.REPORT_TASK_NAME),
            "settings": report.get("settings", {}),
            **graded,
        }

    def _resolve_threshold(self) -> float:
        """targets.pass_rate_threshold (per-model override) > test_config > 0.995."""
        raw = self.targets.get(THRESHOLD_KEY)
        if raw is None:
            raw = self.config.get(THRESHOLD_KEY, DEFAULT_PASS_RATE_THRESHOLD)
        try:
            threshold = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{THRESHOLD_KEY} must be a number, got {raw!r}") from e
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{THRESHOLD_KEY} must be in [0, 1], got {threshold}")
        return threshold

    @staticmethod
    def _raise_if_not_graded(cases: Sequence[Mapping[str, Any]]) -> None:
        """No verdict (-> ERROR) when nothing ran or the server was never reached."""
        if not cases:
            raise RuntimeError("tool-call schema suite ran no cases")
        if all(
            c.get("status") == FAILED_STATUS
            and c.get("cause") == CONNECTION_ERROR_CAUSE
            for c in cases
        ):
            raise RuntimeError(
                "server unreachable: every tool-call schema case failed with a "
                f"connection error, e.g. {cases[0].get('message', '')}"
            )

    def _extra_pytest_args(self) -> List[str]:
        args: List[str] = []
        for key, option, kind in _PYTEST_OPTIONS:
            value: Optional[Any] = self.config.get(key)
            if value is None:
                continue
            if kind == "flag":
                if value:
                    args.append(option)
            else:
                args.extend([option, str(value)])
        # One line per failing case instead of a traceback each.
        args.append("--tb=line")
        return args

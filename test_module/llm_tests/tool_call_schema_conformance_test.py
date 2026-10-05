# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec-test wrapper for the tool-call JSON-schema conformance suite.

Runs ``llm_module/test_tool_call_json_schema.py`` in a child pytest process
(same mechanics as :class:`VLLMParamConformanceTest`) and grades it on a pass
rate: the suite sends 408 tool-call requests whose arguments must validate
against walle JSON Schemas, and a model passes when it gets
>= ``pass_rate_threshold`` of them right. The default is 100% (every case);
a model can be given a lower threshold through its targets.

``exclude_ref_schemas`` keeps the cases whose schema uses ``$ref`` out of the
run, so they are neither sent nor graded; the result lists them under
``excluded_cases``. MiniMax-M3 sets it because the server's MiniMax tool-call
parser loses ``$ref`` types (see ``SuiteSettings.exclude_ref_schemas``).

Status: PASS when ``pass_rate >= threshold``, FAIL below it, ERROR when the
child pytest produced no report, ran no cases, or never reached the server
(every case a connection error).

Settings come from ``test_config``; an environment variable
``TOOL_CALL_SCHEMA_<KEY>`` overrides the key of the same name (e.g.
``TOOL_CALL_SCHEMA_CASE_RETRIES=3``), so a CI dispatch can change a run without
editing the suite files.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from report_module.schema import Block

from .vllm_param_conformance_test import (
    FAILED_STATUS,
    PASSED_STATUS,
    VLLMParamConformanceTest,
)

logger = logging.getLogger(__name__)

DEFAULT_PASS_RATE_THRESHOLD = 1.0
THRESHOLD_KEY = "pass_rate_threshold"
# Must match test_tool_call_json_schema.REPORT_RESULTS_KEY (not imported: that
# would pull llm_module into the orchestrator's import chain).
REPORT_RESULTS_KEY = "test_tool_call_json_schema"
# Must match test_tool_call_json_schema.PROGRESS_PREFIX (same reason).
PROGRESS_PREFIX = "[tool-call-schema progress]"
# Must match tool_call_schema.DEFAULT_TOOL_CHOICE (same reason).
DEFAULT_TOOL_CHOICE = "auto"
CONNECTION_ERROR_CAUSE = "connection_error"
UNKNOWN_CAUSE = "unknown"

ENV_PREFIX = "TOOL_CALL_SCHEMA_"
_FALSE_STRINGS = ("", "0", "false", "no", "off")

# test_config key -> (pytest option, kind). Unset keys keep the suite defaults
# in llm_module/tool_call_schema.py. kind: "value" is passed as a string,
# "flag" as a bare option when true, "json" as a JSON object.
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
    ("default_sampling", "--schema-default-sampling", "flag"),
    ("sampling_params", "--schema-sampling-params", "json"),
    ("cache_bypass", "--schema-cache-bypass", "flag"),
    ("exclude_ref_schemas", "--schema-exclude-ref", "flag"),
)


def _setting(config: Mapping[str, Any], key: str, kind: str) -> Optional[Any]:
    """``TOOL_CALL_SCHEMA_<KEY>`` from the environment, else ``config[key]``."""
    raw = os.environ.get(ENV_PREFIX + key.upper())
    if raw is None:
        return config.get(key)
    logger.info(
        "Tool-call schema setting %s overridden by %s%s", key, ENV_PREFIX, key.upper()
    )
    if kind == "flag":
        return raw.strip().lower() not in _FALSE_STRINGS
    return raw


def _option_args(key: str, option: str, kind: str, value: Any) -> List[str]:
    if kind == "flag":
        return [option] if value else []
    if kind == "json":
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else {}
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a JSON object, got {value!r}")
        return [option, json.dumps(value)]
    return [option, str(value)]


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
            "excluded_cases": report.get("excluded_cases") or [],
        }

    def _resolve_threshold(self) -> float:
        """targets.pass_rate_threshold (per-model override) > test_config > 1.0."""
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

    def _block(self, data: Dict[str, Any]) -> Block:
        """Name the tool choice in the title: a suite runs this test once per
        ``tool_choice`` (auto, then required), and the blocks must be told apart."""
        block = super()._block(data)
        tool_choice = _setting(self.config, "tool_choice", "value")
        return replace(
            block,
            title=f"{block.title} (tool_choice={tool_choice or DEFAULT_TOOL_CHOICE})",
        )

    def _on_pytest_output_line(self, line: str) -> None:
        if line.startswith(PROGRESS_PREFIX):
            logger.info("%s", line)

    def _extra_pytest_args(self) -> List[str]:
        args: List[str] = []
        for key, option, kind in _PYTEST_OPTIONS:
            value = _setting(self.config, key, kind)
            if value is not None:
                args.extend(_option_args(key, option, kind, value))
        # One line per failing case instead of a traceback each.
        args.append("--tb=line")
        return args

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Grading for MiniMax-Provider-Verifier runs.

The verifier's own tests run unchanged from the pinned checkout (``verify.py``
and the ``m3_format_check`` pytest suites); this module turns their raw output
into a verdict, so thresholds and pass/fail rules live with the other spec
tests rather than in the verifier:

* ``verify``: :func:`verify_metrics` recomputes the README metrics from
  ``verify.py``'s ``results.jsonl`` (the formulas of the verifier's
  ``scripts/calculate_batch_metrics.py``), and :func:`grade_verify` checks each
  against its threshold. ToolCalls-Trigger-Similarity is the F1 of "made a tool
  call" against MiniMax's official deployment on the same cases
  (``scripts/calculate_toolcall_similarity.py``).
* pytest suites: :func:`parse_junit` reads the JUnit XML and
  :func:`grade_pytest` grades the pass rate, excluding waived failures.

Pure functions over files and dicts: nothing here imports the verifier.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PASS = "PASS"
FAIL = "FAIL"
NA = "NA"
SUCCESS_STATUS = "success"
TOOL_CALLS = "tool_calls"
STOP = "stop"
_EPSILON = 1e-9

# metric key -> (label, comparison, default threshold). The defaults are the
# verifier README's "Reference Thresholds"; ToolCalls-Match-Rate is documented
# as "about 98% +/- 1%", so it is graded >= 97%. A test case overrides any of
# them through its targets (same key).
VERIFY_METRICS: Dict[str, Tuple[str, str, float]] = {
    "query_success_rate": ("Query-Success-Rate", ">=", 1.0),
    "tool_calls_match_rate": ("ToolCalls-Match-Rate", ">=", 0.97),
    "tool_calls_trigger_similarity": ("ToolCalls-Trigger-Similarity", ">=", 0.98),
    "tool_calls_schema_accuracy": ("ToolCalls-Schema-Accuracy", ">=", 0.98),
    "error_only_reasoning_rate": ("Error-Only-Reasoning-Rate", "<=", 0.0),
    "language_following_success_rate": (
        "Language-Following-Success-Rate",
        ">=",
        0.40,
    ),
    "scenario_check_pass_rate": ("Scenario-Check-Pass-Rate", ">=", 1.0),
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _message(row: Mapping[str, Any]) -> Dict[str, Any]:
    choices = (row.get("response") or {}).get("choices") or [{}]
    return choices[0].get("message") or {}


def finish_reason(row: Mapping[str, Any]) -> Optional[str]:
    choices = (row.get("response") or {}).get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def is_reasoning_only(response: Optional[Mapping[str, Any]]) -> bool:
    """Reasoning with neither content nor a tool call (verify.py's
    ``_is_error_only_reasoning_response``). vLLM returns ``reasoning_content``;
    other servers ``reasoning``.
    """
    choices = (response or {}).get("choices") or []
    if not choices:
        return False
    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return (
        bool(reasoning) and not message.get("content") and not message.get("tool_calls")
    )


def baseline_tool_calls(path: Path) -> Dict[Any, bool]:
    """data_index -> whether MiniMax's official deployment made a tool call."""
    return {
        row.get("data_index"): finish_reason(row) == TOOL_CALLS
        for row in load_jsonl(path)
    }


def verify_metrics(
    rows: Sequence[Mapping[str, Any]],
    summary: Optional[Mapping[str, Any]],
    baseline: Optional[Mapping[Any, bool]],
) -> Dict[str, Dict[str, Any]]:
    """The README metrics of one ``verify.py`` run: key -> {value, detail}.

    ``value`` is None when a metric has no cases (e.g. no baseline).
    """
    attempts = (summary or {}).get("all_count") or len(rows)
    succeeded = sum(r.get("status") == SUCCESS_STATUS for r in rows)
    labelled = [r for r in rows if r.get("expected_tool_call") in (True, False)]
    # verify.py's own finish reason per case; cases routed to another check
    # (check_type) have none, stay in the denominator and never match.
    tp = sum(
        r["expected_tool_call"] is True
        and r.get("tool_calls_finish_reason") == TOOL_CALLS
        for r in labelled
    )
    tn = sum(
        r["expected_tool_call"] is False and r.get("tool_calls_finish_reason") == STOP
        for r in labelled
    )
    schema_ok = sum(
        r["expected_tool_call"] is True
        and r.get("tool_calls_finish_reason") == TOOL_CALLS
        and bool(r.get("tool_calls_valid"))
        for r in labelled
    )
    reasoning_only = sum(is_reasoning_only(r.get("response")) for r in rows)
    language = [r for r in rows if r.get("language_following_checked")]
    language_ok = sum(bool(r.get("language_following_valid")) for r in language)
    scenario = [r for r in rows if r.get("scenario_check_checked")]
    scenario_ok = sum(bool(r.get("scenario_check_valid")) for r in scenario)

    metrics = {
        "query_success_rate": {
            "value": _ratio(succeeded, attempts),
            "detail": f"{succeeded}/{attempts} requests succeeded (incl. retries)",
        },
        "tool_calls_match_rate": {
            "value": _ratio(tp + tn, len(labelled)),
            "detail": f"{tp + tn}/{len(labelled)} labelled cases (TP={tp}, TN={tn})",
        },
        "tool_calls_schema_accuracy": {
            "value": _ratio(schema_ok, tp),
            "detail": f"{schema_ok}/{tp} expected tool calls pass schema validation",
        },
        "error_only_reasoning_rate": {
            "value": _ratio(reasoning_only, len(rows)),
            "detail": f"{reasoning_only}/{len(rows)} responses contain reasoning only",
        },
        "language_following_success_rate": {
            "value": _ratio(language_ok, len(language)),
            "detail": f"{language_ok}/{len(language)} checked cases",
        },
        "scenario_check_pass_rate": {
            "value": _ratio(scenario_ok, len(scenario)),
            "detail": f"{scenario_ok}/{len(scenario)} checked cases",
        },
        "tool_calls_trigger_similarity": _trigger_similarity(rows, baseline),
    }
    return {key: metrics[key] for key in VERIFY_METRICS}


def _trigger_similarity(
    rows: Sequence[Mapping[str, Any]], baseline: Optional[Mapping[Any, bool]]
) -> Dict[str, Any]:
    """F1 of "made a tool call" against the official deployment, on the cases
    both ran (calculate_toolcall_similarity.calculate_tool_call_f1)."""
    if not baseline:
        return {"value": None, "detail": "no official baseline for this model"}
    counts: Counter = Counter()
    for row in rows:
        official = baseline.get(row.get("data_index"))
        if official is None:
            continue
        made = finish_reason(row) == TOOL_CALLS
        counts[
            {(True, True): "tp", (False, True): "fp", (True, False): "fn"}.get(
                (official, made), "tn"
            )
        ] += 1
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    if not tp + fp + fn + tn:
        return {"value": None, "detail": "no case in common with the baseline"}
    precision = _ratio(tp, tp + fp) or 0.0
    recall = _ratio(tp, tp + fn) or 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "value": f1,
        "detail": f"F1 on {tp + fp + fn + tn} cases (TP={tp}, FP={fp}, FN={fn}, TN={tn})",
    }


def mean_metrics(
    per_run: Sequence[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Each metric's mean over runs (the README thresholds are pass@N means)."""
    if len(per_run) == 1:
        return {key: dict(m) for key, m in per_run[0].items()}
    out = {}
    for key in VERIFY_METRICS:
        values = [run[key]["value"] for run in per_run if run[key]["value"] is not None]
        runs = ", ".join(_percent(run[key]["value"]) for run in per_run)
        out[key] = {
            "value": sum(values) / len(values) if values else None,
            "detail": f"mean of {len(values)}/{len(per_run)} runs ({runs})",
        }
    return out


def _metric_status(op: str, value: Optional[float], threshold: float) -> str:
    if value is None:
        return NA
    if op == ">=":
        return PASS if value >= threshold - _EPSILON else FAIL
    return PASS if value <= threshold + _EPSILON else FAIL


def grade_verify(
    metrics: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, float]
) -> Dict[str, Any]:
    """Per-metric verdicts; ``success`` when no metric FAILs (NA is not a FAIL)."""
    rows = []
    for key, (label, op, default) in VERIFY_METRICS.items():
        threshold = float(thresholds.get(key, default))
        value = metrics[key]["value"]
        rows.append(
            {
                "metric": label,
                "key": key,
                "value": value,
                "value_percent": _percent(value),
                "threshold": f"{op} {_percent(threshold)}",
                "status": _metric_status(op, value, threshold),
                "detail": metrics[key]["detail"],
            }
        )
    failed = [r["metric"] for r in rows if r["status"] == FAIL]
    return {"metrics": rows, "failed_metrics": failed, "success": not failed}


def verify_case_failures(
    rows: Sequence[Mapping[str, Any]], run: Optional[int] = None
) -> List[Dict[str, Any]]:
    """One entry per (case, cause) that cost a metric, for the report."""
    out: List[Dict[str, Any]] = []

    def add(row, cause, detail=""):
        entry = {"case": row.get("data_index"), "cause": cause, "detail": detail}
        if run is not None:
            entry["run"] = run
        out.append(entry)

    for row in rows:
        reason, expected = finish_reason(row), row.get("expected_tool_call")
        if row.get("status") != SUCCESS_STATUS:
            error = (row.get("response") or {}).get("error", "")
            add(row, "request failed after all retries", str(error)[:200])
            continue
        if expected is True and reason != TOOL_CALLS:
            add(row, f"tool call expected, finish_reason={reason}")
        elif expected is False and reason != STOP:
            add(row, f"plain answer expected, finish_reason={reason}")
        if (
            expected is True
            and reason == TOOL_CALLS
            and not row.get("tool_calls_valid")
        ):
            add(row, "tool call failed the schema check", schema_failure(row))
        if is_reasoning_only(row.get("response")):
            add(row, "reasoning-only response (no content, no tool call)")
        if row.get("language_following_checked") and not row.get(
            "language_following_valid"
        ):
            add(row, "language not followed (Russian characters in the answer)")
        if row.get("scenario_check_checked") and not row.get("scenario_check_valid"):
            add(row, "tool parameter order not preserved")
    return out


def schema_failure(row: Mapping[str, Any]) -> str:
    """Why a tool call failed verify.py's schema check, as far as JSON Schema
    explains it (verify.py also rejects array ``command`` values merged into
    one string, reported here as the fallback)."""
    from jsonschema import Draft202012Validator

    tools = {
        (t.get("function") or {}).get("name"): (t.get("function") or {}).get(
            "parameters"
        )
        for t in (row.get("request") or {}).get("tools") or []
    }
    calls = _message(row).get("tool_calls") or []
    if not calls:
        return "finish_reason=tool_calls but no tool_calls returned"
    for call in calls:
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments")
        schema = tools.get(name)
        if not schema:
            return f"call to undefined tool {name!r}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as e:
            return f"{name}: arguments are not valid JSON ({e})"
        error = next(iter(Draft202012Validator(schema).iter_errors(args)), None)
        if error is not None:
            path = "/".join(str(p) for p in error.absolute_path) or "<root>"
            return f"{name} at {path}: {error.message[:160]}"
    return "verifier-specific check (e.g. array command merged into one string)"


def parse_junit(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """(test results, collection error) from a pytest JUnit XML file.

    Outcomes: passed, failed, error, skipped, xfailed.
    """
    results: List[Dict[str, Any]] = []
    collection_error = None
    for case in ET.parse(path).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is None:
                continue
            outcome = {"failure": "failed"}.get(tag, tag)
            if tag == "skipped" and node.get("type") == "pytest.xfail":
                outcome = "xfailed"
            message = node.get("message") or ""
            if tag in ("failure", "error"):
                message = _assertion_lines(node.text or "") or message
            break
        # A collection error is an <error> testcase without a classname.
        if outcome == "error" and not case.get("classname"):
            lines = message.strip().splitlines()
            collection_error = lines[-1] if lines else "collection error"
            continue
        results.append(
            {
                "test": f"{case.get('classname', '')}::{case.get('name', '?')}",
                "outcome": outcome,
                "message": message,
            }
        )
    return results, collection_error


def _assertion_lines(traceback: str) -> str:
    """The pytest ``E`` lines of a traceback, without the source lines."""
    lines = [
        line[1:].strip() for line in traceback.splitlines() if line.startswith("E ")
    ]
    return "\n".join(line for line in lines if line)


def grade_pytest(
    results: Sequence[Mapping[str, Any]],
    threshold: float,
    waivers: Sequence[Tuple[str, str]] = (),
) -> Dict[str, Any]:
    """Pass rate over executed tests (skipped and xfailed excluded), with
    failures matching a waiver ``(test-id regex, reason)`` taken out of it.

    ``success`` is ``pass_rate >= threshold``.
    """
    counts = Counter(r["outcome"] for r in results)
    failures, waived = [], []
    for r in results:
        if r["outcome"] not in ("failed", "error"):
            continue
        reason = next(
            (why for pattern, why in waivers if re.search(pattern, r["test"])), None
        )
        entry = {"test": r["test"], "outcome": r["outcome"], "message": r["message"]}
        if reason:
            waived.append({**entry, "waived": reason})
        else:
            failures.append(entry)
    executed = counts["passed"] + counts["failed"] + counts["error"]
    pass_rate = _ratio(counts["passed"], executed - len(waived))
    return {
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
        "xfailed": counts["xfailed"],
        "waived": len(waived),
        "pass_rate": pass_rate,
        "pass_rate_percent": _percent(pass_rate),
        "threshold": threshold,
        "threshold_percent": _percent(threshold),
        "failures": failures,
        "waived_failures": waived,
        # Every executed test failed and was waived: nothing left to grade.
        "success": (
            not failures if pass_rate is None else pass_rate >= threshold - _EPSILON
        ),
    }

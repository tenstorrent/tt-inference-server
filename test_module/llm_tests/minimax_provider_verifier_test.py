# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec tests that run MiniMax's provider verifier against a served MiniMax-M3.

MiniMax publishes a vendor-neutral verifier for third-party deployments of
its models (https://github.com/MiniMax-AI/MiniMax-Provider-Verifier). It has
two independent parts, and each gets its own spec test here:

* :class:`MiniMaxProviderVerifierTest` runs ``scripts/batch_verify.py`` over
  ``sample.jsonl`` (102 requests: tool-call trigger labels, minor-language
  following, a tool-schema key-order scenario) ``loops`` times -- pass@10 is
  the upstream method -- and grades the mean of each metric against the
  thresholds in the verifier's README. Every metric is one row of
  ``parameter_conformance_summary``, so a model spec ``known_issues`` entry
  (workflow_type SPEC_TESTS, task_name = the metric key) waives exactly that
  metric.
* :class:`MiniMaxM3FormatCheckTest` runs the ``m3_format_check`` pytest suite
  (the M3 ``/v1/chat/completions`` contract: SSE framing, usage, finish
  reasons, tool calls, error codes, ...) and grades it like the vLLM
  parameter suites: one summary row per test class, one detail row per case,
  any failure fails the block, and ``known_issues`` can waive a class
  (``m3_text_tests.TestUsageField``) or one case by its pytest name.

The verifier is checked out at ``MINIMAX_PROVIDER_VERIFIER_COMMIT`` into the
MINIMAX_PROVIDER_VERIFIER venv and its scripts run unmodified, so the numbers
stay comparable with the ones MiniMax publishes. Only the aggregation is done
here: the per-loop metric formulas are those of
``scripts/calculate_batch_metrics.py``, and ToolCalls-Trigger-Similarity is
computed here because the upstream lookup of the M3 baseline
(``<repo>/MiniMax-M3/**/minimax_results.jsonl``) matches no file in the
repository, which leaves that metric out of the upstream M3 report.

Settings come from ``test_config``; an environment variable
``MINIMAX_VERIFIER_<KEY>`` overrides the key of the same name (e.g.
``MINIMAX_VERIFIER_LOOPS=1``), so a CI dispatch can shorten a run without
editing the suite files. ``MINIMAX_PROVIDER_VERIFIER_DIR`` and
``MINIMAX_PROVIDER_VERIFIER_PYTHON`` point both tests at an existing checkout
and interpreter instead of the provisioned venv (local development).

``python -m test_module.llm_tests.minimax_provider_verifier_test --server-url
<endpoint>`` runs the two tests alone against an endpoint (``spec_tests``
runs every spec test of the model), with their suite-template settings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .vllm_param_conformance_test import (
    FAILED_STATUS,
    PASSED_STATUS,
    SKIPPED_STATUS,
    VLLMParamConformanceTest,
    _truncate_message,
)

logger = logging.getLogger(__name__)

ENV_PREFIX = "MINIMAX_VERIFIER_"
VERIFIER_DIR_ENV = "MINIMAX_PROVIDER_VERIFIER_DIR"
VERIFIER_PYTHON_ENV = "MINIMAX_PROVIDER_VERIFIER_PYTHON"
_FALSE_STRINGS = ("", "0", "false", "no", "off")

# Provider name batch_verify.py writes its files under: <name>_results.jsonl,
# <name>_summary.json.
PROVIDER_NAME = "tt-inference-server"
SAMPLE_FILE = "sample.jsonl"
# The official M3 deployment's first pass@10 loop, the reference
# ToolCalls-Trigger-Similarity is measured against (upstream compares against
# the first reference file it finds).
DEFAULT_BASELINE_RESULTS = (
    "output-dir/MiniMax-M3/loop_01/minimax-m3-official/"
    "minimax-m3-official_results.jsonl"
)
FORMAT_CHECK_DIR = "m3_format_check"
# Image and video are left out: the served M3 is text-only. The m3_a_* files
# test the separate M3-a variant.
DEFAULT_FORMAT_SUITES = ("m3_text_tests.py", "m3_stream_tests.py")
# Lines of each child's output kept in the block's logs; the full output is
# written next to the run's artifacts.
LOG_TAIL_LINES = 200
# Rows of the per-case failure table kept in the block (most-failing first);
# the full per-loop results stay in the artifacts directory.
MAX_FAILING_CASE_ROWS = 50

GE = ">="
LE = "<="

# (key, upstream name, direction, default threshold). Defaults are the
# "Reference Thresholds" of the verifier's README: Query-Success 100%,
# ToolCalls-Match "≈98%, ±1%" (so 97% is the floor), Trigger-Similarity and
# Schema-Accuracy >= 98%, Error-Only-Reasoning 0%, Language-Following >= 40%,
# Scenario-Check 100%. A targets entry with the same key overrides one.
METRICS: Tuple[Tuple[str, str, str, float], ...] = (
    ("query_success_rate", "Query-Success-Rate", GE, 1.0),
    ("tool_calls_match_rate", "ToolCalls-Match-Rate", GE, 0.97),
    ("tool_calls_trigger_similarity", "ToolCalls-Trigger-Similarity", GE, 0.98),
    ("tool_calls_schema_accuracy", "ToolCalls-Schema-Accuracy", GE, 0.98),
    ("error_only_reasoning_rate", "Error-Only-Reasoning-Rate", LE, 0.0),
    ("language_following_success_rate", "Language-Following-Success-Rate", GE, 0.40),
    ("scenario_check_pass_rate", "Scenario-Check-Pass-Rate", GE, 1.0),
)


def _setting(config: Mapping[str, Any], key: str, kind: str = "value") -> Any:
    """``MINIMAX_VERIFIER_<KEY>`` from the environment, else ``config[key]``."""
    raw = os.environ.get(ENV_PREFIX + key.upper())
    if raw is None:
        return config.get(key)
    logger.info(
        "MiniMax verifier setting %s overridden by %s%s", key, ENV_PREFIX, key.upper()
    )
    if kind == "flag":
        return raw.strip().lower() not in _FALSE_STRINGS
    if kind == "json":
        return json.loads(raw) if raw.strip() else None
    if kind == "list":
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _int_setting(config: Mapping[str, Any], key: str, default: int) -> int:
    raw = _setting(config, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from e
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    return value


def _percent(fraction: Optional[float]) -> str:
    return "n/a" if fraction is None else f"{fraction * 100:.2f}%"


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator > 0 else None


# --------------------------------------------------------------------------
# MiniMaxProviderVerifierTest grading
# --------------------------------------------------------------------------


def expected_tool_call_total(sample_rows: Sequence[Mapping[str, Any]]) -> int:
    """Rows labelled with a boolean ``expected_tool_call`` (the Match-Rate
    denominator, as in upstream ``get_expected_tool_call_count``)."""
    return sum(1 for r in sample_rows if r.get("expected_tool_call") in (True, False))


def finish_reason(result: Mapping[str, Any]) -> str:
    """Upstream ``calculate_toolcall_similarity.get_finish_reason``."""
    response = result.get("response") if result else None
    if not isinstance(response, Mapping):
        return "error_no_response"
    choices = response.get("choices")
    if not choices:
        return "error_no_choices"
    reason = (
        choices[0].get("finish_reason") if isinstance(choices[0], Mapping) else None
    )
    return "error_no_finish_reason" if reason is None else str(reason)


def trigger_similarity(
    baseline: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]
) -> Optional[float]:
    """F1 of "finished with tool_calls" against the official deployment.

    Upstream ``calculate_tool_call_f1`` pairs the two runs by position; they
    are paired by ``data_index`` here, so a missing row cannot shift every
    later comparison. ``None`` when no case is shared.
    """
    by_index = {r.get("data_index"): r for r in baseline}
    tp = fp = fn = 0
    shared = 0
    for result in results:
        reference = by_index.get(result.get("data_index"))
        if reference is None:
            continue
        shared += 1
        official = finish_reason(reference) == "tool_calls"
        actual = finish_reason(result) == "tool_calls"
        tp += official and actual
        fp += actual and not official
        fn += official and not actual
    if not shared:
        return None
    precision = _ratio(tp, tp + fp) or 0.0
    recall = _ratio(tp, tp + fn) or 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def loop_metrics(
    summary: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    expected_total: int,
) -> Dict[str, Optional[float]]:
    """One loop's metrics from its summary, with the formulas of upstream
    ``calculate_batch_metrics.calculate_metrics``. ``None`` = not measured."""
    tp = summary.get("tool_calls_finish_tool_calls", 0)
    match_denominator = (
        expected_total
        or summary.get("expected_tool_call_total_count", 0)
        or summary.get("success_count", 0)
    )
    return {
        "query_success_rate": _ratio(
            summary.get("success_count", 0), summary.get("all_count", 0)
        ),
        "tool_calls_match_rate": _ratio(
            tp + summary.get("stop_finish_stop", 0), match_denominator
        ),
        "tool_calls_trigger_similarity": trigger_similarity(baseline, results),
        "tool_calls_schema_accuracy": _ratio(
            summary.get("tool_calls_successful_count", 0), tp
        ),
        "error_only_reasoning_rate": _ratio(
            summary.get("error_only_reasoning_count", 0),
            summary.get("error_only_reasoning_checked_count", 0),
        ),
        "language_following_success_rate": _ratio(
            summary.get("language_following_valid_count", 0),
            summary.get("language_following_checked_count", 0),
        ),
        "scenario_check_pass_rate": _ratio(
            summary.get("scenario_check_valid_count", 0),
            summary.get("scenario_check_checked_count", 0),
        ),
    }


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    measured = [v for v in values if v is not None]
    return sum(measured) / len(measured) if measured else None


def _case_failures(result: Mapping[str, Any]) -> List[str]:
    """Which checks one result row failed (empty when it passed them all)."""
    failures = []
    if result.get("status") != "success":
        failures.append("request_failed")
        return failures
    expected = result.get("expected_tool_call")
    called = finish_reason(result) == "tool_calls"
    if expected is True and not called:
        failures.append("missed_tool_call")
    elif expected is False and called:
        failures.append("unexpected_tool_call")
    if expected is True and called and result.get("tool_calls_valid") is False:
        failures.append("invalid_tool_call_schema")
    if result.get("error_only_reasoning"):
        failures.append("error_only_reasoning")
    if result.get("language_following_checked") and not result.get(
        "language_following_valid"
    ):
        failures.append("language_not_followed")
    if result.get("scenario_check_checked") and not result.get("scenario_check_valid"):
        failures.append("scenario_check_failed")
    return failures


def _failure_detail(result: Mapping[str, Any]) -> str:
    response = result.get("response")
    if isinstance(response, Mapping) and response.get("error"):
        return _truncate_message(response["error"])
    return f"finish_reason={finish_reason(result)}"


def grade_verifier_loops(
    loops: Sequence[Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    baseline: Sequence[Mapping[str, Any]],
    expected_total: int,
    thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    """Grade ``(summary, results)`` per loop on the mean of each metric.

    A loop that could not measure a metric (``None``: e.g. the scenario case
    is only checked when the model answers it in text) is left out of that
    metric's mean, as upstream does; a metric no loop measured fails.
    """
    per_loop = [
        loop_metrics(summary, results, baseline, expected_total)
        for summary, results in loops
    ]

    summary_rows: List[Dict[str, str]] = []
    metrics: Dict[str, Optional[float]] = {}
    success = bool(per_loop)
    for key, name, direction, _default in METRICS:
        value = _mean([m[key] for m in per_loop])
        threshold = thresholds[key]
        metrics[key] = value
        if value is None:
            passed = False
            detail = "not measured"
        else:
            passed = value >= threshold if direction == GE else value <= threshold
            detail = f"{_percent(value)} (needs {direction} {_percent(threshold)})"
        success = success and passed
        summary_rows.append(
            {
                "test_case": key,
                "metric": name,
                "status": "✅ PASS" if passed else "❌ FAIL",
                "summary": detail,
            }
        )

    loop_rows = [
        {"loop": i, **{key: _percent(m[key]) for key, *_ in METRICS}}
        for i, m in enumerate(per_loop, 1)
    ]

    failing: Dict[Tuple[int, str], List[str]] = {}
    for _summary, results in loops:
        for result in results:
            for check in _case_failures(result):
                failing.setdefault((result.get("data_index"), check), []).append(
                    _failure_detail(result)
                )
    case_rows = [
        {
            "data_index": index,
            "check": check,
            "loops_failed": f"{len(details)}/{len(loops)}",
            "detail": Counter(details).most_common(1)[0][0],
        }
        for (index, check), details in sorted(
            failing.items(), key=lambda item: (-len(item[1]), str(item[0]))
        )
    ]

    return {
        "loops": len(per_loop),
        "metrics": metrics,
        "parameter_conformance_summary": summary_rows,
        "loop_metrics": loop_rows,
        "failing_case_count": len(case_rows),
        "failing_cases": case_rows[:MAX_FAILING_CASE_ROWS],
        "success": success,
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# MiniMaxM3FormatCheckTest grading
# --------------------------------------------------------------------------


def parse_junit_results(junit_xml: str) -> Dict[str, List[Dict[str, str]]]:
    """``{classname: [{test_node_name, status, message}]}`` from a JUnit report.

    Failures and errors are failures; skips (including xfails) are skips.
    """
    results: Dict[str, List[Dict[str, str]]] = {}
    for case in ET.fromstring(junit_xml).iter("testcase"):
        status, message = PASSED_STATUS, ""
        for child in case:
            if child.tag in ("failure", "error"):
                status = FAILED_STATUS
                message = child.get("message") or (child.text or "")
                break
            if child.tag == "skipped":
                status = SKIPPED_STATUS
                message = child.get("message") or (child.text or "")
        results.setdefault(case.get("classname", ""), []).append(
            {
                "test_node_name": case.get("name", ""),
                "status": status,
                "message": message.strip(),
            }
        )
    return results


def count_statuses(results: Mapping[str, Sequence[Mapping[str, Any]]]) -> Counter:
    return Counter(t.get("status") for tests in results.values() for t in tests)


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


class _VerifierMixin:
    """Locating the verifier, running its children, keeping their output."""

    config: Mapping[str, Any]
    ctx: Any
    logs: list
    KIND: str

    def _resolve_verifier(self) -> Tuple[str, Path]:
        """``(python, checkout)``: an explicit override, else the provisioned venv."""
        checkout = os.environ.get(VERIFIER_DIR_ENV)
        if checkout:
            python = os.environ.get(VERIFIER_PYTHON_ENV) or shutil.which("python3")
            logger.info("Using MiniMax verifier checkout %s (%s)", checkout, python)
            return str(python), Path(checkout)

        from workflow_module.engine_types import WorkflowVenvType
        from workflow_module.venv_provisioner import get_venv_provisioner
        from workflows.workflow_venvs import MINIMAX_PROVIDER_VERIFIER_DIR_NAME

        provisioner = get_venv_provisioner()
        venv_type = WorkflowVenvType.MINIMAX_PROVIDER_VERIFIER
        model_spec = getattr(self.ctx, "model_spec", None)
        if not provisioner.provision(venv_type, model_spec):
            raise RuntimeError("Failed to provision the MiniMax provider verifier venv")
        return (
            provisioner.venv_python(venv_type),
            Path(provisioner.venv_path(venv_type)) / MINIMAX_PROVIDER_VERIFIER_DIR_NAME,
        )

    def _artifact_dir(self) -> Path:
        """Where this run keeps the verifier's outputs (kept after the run)."""
        root = (
            getattr(self, "output_root", None)
            or getattr(self.ctx, "output_path", None)
            or tempfile.gettempdir()
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = Path(root) / "minimax_provider_verifier" / f"{self.KIND}_{stamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _api_key() -> str:
        """The key the workflow set up for test children (see setup_tests_auth)."""
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""

    async def _run_child(
        self,
        command: List[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run *command*, write its output to *log_path*, keep a tail in the logs."""
        logger.info("Running %s", " ".join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        tail: deque = deque(maxlen=LOG_TAIL_LINES)
        try:
            with log_path.open("wb") as log:
                pending = b""
                while True:
                    chunk = await process.stdout.read(65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    *lines, pending = (pending + chunk).split(b"\n")
                    tail.extend(line.decode(errors="replace") for line in lines)
                if pending:
                    tail.append(pending.decode(errors="replace"))
            await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        self.logs.extend(tail)
        logger.info("%s exited with code %s", command[1], process.returncode)
        return process.returncode


class MiniMaxProviderVerifierTest(_VerifierMixin, VLLMParamConformanceTest):
    """pass@N of the verifier's ``sample.jsonl``, graded per metric."""

    KIND = "minimax_provider_verifier"
    # "functional" so acceptance_criteria counts it (see VLLMParamConformanceTest).
    TASK_TYPE = "functional"
    ENDPOINT_PATH = "/v1"

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        python, checkout = self._resolve_verifier()
        loops = _int_setting(self.config, "loops", 10)
        thresholds = self._resolve_thresholds()
        base_url = f"{self.base_url}{self.ENDPOINT_PATH}"
        model_name = self._resolve_model_name()
        artifacts = self._artifact_dir()

        sample_path = checkout / SAMPLE_FILE
        expected_total = expected_tool_call_total(_read_jsonl(sample_path))
        baseline_path = checkout / str(
            _setting(self.config, "baseline_results") or DEFAULT_BASELINE_RESULTS
        )
        baseline = _read_jsonl(baseline_path)

        loop_outputs = []
        # provider.json carries the API key, so it stays out of the artifacts.
        with tempfile.TemporaryDirectory(prefix="minimax_verifier_") as tmp:
            provider_file = Path(tmp) / "provider.json"
            provider_file.write_text(
                json.dumps(
                    [
                        {
                            "name": PROVIDER_NAME,
                            "model": model_name,
                            "base_url": base_url,
                            "api_key": self._api_key() or "EMPTY",
                        }
                    ]
                )
            )
            for loop in range(1, loops + 1):
                loop_dir = artifacts / f"loop_{loop:02d}"
                loop_dir.mkdir(parents=True, exist_ok=True)
                logger.info("MiniMax provider verifier loop %d/%d", loop, loops)
                rc = await self._run_child(
                    self._batch_verify_command(
                        python, checkout, sample_path, provider_file, loop_dir
                    ),
                    cwd=checkout,
                    env=self._child_env(checkout),
                    log_path=loop_dir / "verifier.log",
                )
                loop_outputs.append(self._load_loop(loop_dir, rc))

        self._raise_if_not_graded(loop_outputs)
        graded = grade_verifier_loops(
            loop_outputs, baseline, expected_total, thresholds
        )
        logger.info(
            "MiniMax provider verifier over %d loop(s): %s -> %s",
            graded["loops"],
            ", ".join(
                f"{key}={_percent(value)}" for key, value in graded["metrics"].items()
            ),
            "PASS" if graded["success"] else "FAIL",
        )
        self.targets = {**self.targets, **thresholds}
        return {
            "endpoint_url": base_url,
            "model_name": model_name,
            "task_name": self.KIND,
            "artifacts_dir": str(artifacts),
            "settings": self._settings_summary(loops),
            **graded,
        }

    def _batch_verify_command(
        self,
        python: str,
        checkout: Path,
        sample_path: Path,
        provider_file: Path,
        output_dir: Path,
    ) -> List[str]:
        command = [
            python,
            str(checkout / "scripts" / "batch_verify.py"),
            str(sample_path),
            "--provider-file",
            str(provider_file),
            "--output-dir",
            str(output_dir),
            "--max-workers",
            str(_int_setting(self.config, "max_workers", 10)),
            "--timeout",
            str(_int_setting(self.config, "request_timeout", 1800)),
            "--retries",
            str(_int_setting(self.config, "retries", 3)),
        ]
        if _setting(self.config, "stream", "flag"):
            command.append("--stream")
        extra_body = _setting(self.config, "extra_body", "json")
        if extra_body:
            if not isinstance(extra_body, dict):
                raise ValueError(
                    f"extra_body must be a JSON object, got {extra_body!r}"
                )
            command.extend(["--extra-body", json.dumps(extra_body)])
        return command

    @staticmethod
    def _child_env(checkout: Path) -> Dict[str, str]:
        # scripts/batch_verify.py imports verify.py from the repository root,
        # which is not on the path of a script run from scripts/.
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(checkout)] + ([existing] if existing else [])
        )
        return env

    @staticmethod
    def _load_loop(
        loop_dir: Path, rc: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        summary_path = loop_dir / f"{PROVIDER_NAME}_summary.json"
        results_path = loop_dir / f"{PROVIDER_NAME}_results.jsonl"
        if not summary_path.exists() or not results_path.exists():
            raise RuntimeError(
                f"batch_verify.py wrote no summary in {loop_dir} (exit code {rc}); "
                f"see {loop_dir / 'verifier.log'}"
            )
        return json.loads(summary_path.read_text()), _read_jsonl(results_path)

    @staticmethod
    def _raise_if_not_graded(
        loops: Sequence[Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    ) -> None:
        """No verdict (-> ERROR) when no request ever succeeded."""
        if not any(summary.get("success_count") for summary, _ in loops):
            errors = [
                (r.get("response") or {}).get("error")
                for _, results in loops
                for r in results
            ]
            example = next((e for e in errors if e), "no results")
            raise RuntimeError(
                f"server unreachable: no verifier request succeeded, e.g. {example}"
            )

    def _resolve_thresholds(self) -> Dict[str, float]:
        """targets.<metric> (per-model override) > the README default."""
        thresholds = {}
        for key, _name, _direction, default in METRICS:
            raw = self.targets.get(key, default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"targets.{key} must be a number, got {raw!r}") from e
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"targets.{key} must be in [0, 1], got {value}")
            thresholds[key] = value
        return thresholds

    def _settings_summary(self, loops: int) -> Dict[str, Any]:
        return {
            "loops": loops,
            "max_workers": _int_setting(self.config, "max_workers", 10),
            "request_timeout": _int_setting(self.config, "request_timeout", 1800),
            "retries": _int_setting(self.config, "retries", 3),
            "stream": bool(_setting(self.config, "stream", "flag")),
            "extra_body": _setting(self.config, "extra_body", "json") or {},
        }


class MiniMaxM3FormatCheckTest(_VerifierMixin, VLLMParamConformanceTest):
    """The verifier's M3 chat-completions format suite (text + stream)."""

    KIND = "minimax_m3_format_check"
    TASK_TYPE = "functional"
    REPORT_TASK_NAME = "minimax_m3_format_check"

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        python, checkout = self._resolve_verifier()
        suite_dir = checkout / FORMAT_CHECK_DIR
        model_name = self._resolve_model_name()
        artifacts = self._artifact_dir()
        junit_path = artifacts / "junit.xml"

        started = time.time()
        rc = await self._run_child(
            self._pytest_command(python, junit_path),
            cwd=suite_dir,
            env=self._suite_env(model_name, artifacts),
            log_path=artifacts / "pytest.log",
        )
        self._collect_worker_logs(suite_dir, artifacts, started)
        if not junit_path.exists():
            raise RuntimeError(
                f"m3_format_check produced no report (exit code {rc}); "
                f"see {artifacts / 'pytest.log'}"
            )

        results = parse_junit_results(junit_path.read_text(encoding="utf-8"))
        counts = count_statuses(results)
        if not results:
            raise RuntimeError("m3_format_check ran no cases")
        if counts[FAILED_STATUS] and not counts[PASSED_STATUS]:
            raise RuntimeError(
                "server unreachable or misconfigured: every m3_format_check case "
                "failed; see " + str(artifacts / "pytest.log")
            )
        logger.info(
            "m3_format_check: %d passed, %d failed, %d skipped",
            counts[PASSED_STATUS],
            counts[FAILED_STATUS],
            counts[SKIPPED_STATUS],
        )
        return {
            "endpoint_url": f"{self.base_url}/v1/chat/completions",
            "model_name": model_name,
            "task_name": self.REPORT_TASK_NAME,
            "artifacts_dir": str(artifacts),
            "passed": counts[PASSED_STATUS],
            "failed": counts[FAILED_STATUS],
            "skipped": counts[SKIPPED_STATUS],
            "parameter_conformance_summary": self._build_conformance_summary(results),
            "detailed_test_results": self._build_detailed_results(results),
            "success": self._all_passed(results),
        }

    def _pytest_command(self, python: str, junit_path: Path) -> List[str]:
        suites = _setting(self.config, "suites", "list") or list(DEFAULT_FORMAT_SUITES)
        command = [
            python,
            "-m",
            "pytest",
            *suites,
            f"--junitxml={junit_path}",
            "-p",
            "no:cacheprovider",
            "--tb=line",
        ]
        workers = _int_setting(self.config, "workers", 8)
        if workers > 1:
            command.extend(["-n", str(workers)])
        case_timeout = _setting(self.config, "case_timeout")
        if case_timeout is not None:
            command.append(f"--timeout={int(case_timeout)}")
        for node_id in _setting(self.config, "deselect", "list") or []:
            command.extend(["--deselect", node_id])
        return command

    def _suite_env(self, model_name: str, artifacts: Path) -> Dict[str, str]:
        env = dict(os.environ)
        env["M3_BASE_URL"] = self.base_url
        env["M3_MODEL"] = model_name
        key = self._api_key()
        if key:
            env["M3_API_KEY"] = key
            env["M3_AUTH_TYPE"] = "bearer"
        else:
            env["M3_AUTH_TYPE"] = "none"
        if _setting(self.config, "skip_reasoning_split", "flag"):
            env["M3_SKIP_REASONING_SPLIT"] = "1"
        max_tokens = _setting(self.config, "max_tokens")
        if max_tokens is not None:
            env["M3_MAX_TOKENS"] = str(int(max_tokens))
        if _int_setting(self.config, "workers", 8) <= 1:
            # Serial runs write one request log; xdist workers each write their
            # own under the suite's logs/ (collected afterwards).
            env["M3_RUN_LOG"] = str(artifacts / "requests.jsonl")
        return env

    @staticmethod
    def _collect_worker_logs(suite_dir: Path, artifacts: Path, started: float) -> None:
        """Move this run's per-worker request logs next to its other artifacts."""
        log_dir = suite_dir / "logs"
        if not log_dir.is_dir():
            return
        for path in log_dir.glob("run_*.jsonl"):
            if path.stat().st_mtime >= started:
                shutil.move(str(path), str(artifacts / path.name))


# --------------------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------------------

_TESTS = {
    "verifier": MiniMaxProviderVerifierTest,
    "format": MiniMaxM3FormatCheckTest,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the verifier tests against ``--server-url``; exit 0 when all pass."""
    import argparse

    from test_module._test_common import TestConfig
    from test_module.test_categorization_system.suite_loader import (
        load_server_tests_config,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server-url", required=True, help="e.g. http://host:8000")
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-M3")
    parser.add_argument(
        "--tests", default="verifier,format", help="comma-separated: verifier, format"
    )
    parser.add_argument("--output-dir", default=None, help="artifacts root")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="test_config override, e.g. --set loops=1 (repeatable)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    templates = load_server_tests_config()["test_templates"]

    status = 0
    for name in [t.strip() for t in args.tests.split(",") if t.strip()]:
        cls = _TESTS[name]
        config = {
            **templates[cls.__name__]["test_config"],
            **overrides,
            "model": args.model,
        }
        test = cls(TestConfig(config), {})
        test.base_url = args.server_url.rstrip("/")
        test.output_root = args.output_dir
        data = test.run_tests().data
        print(f"\n== {cls.__name__}: {data['status'].upper()}")
        for row in data.get("parameter_conformance_summary") or []:
            print(f"  {row['status']}  {row['test_case']}: {row['summary']}")
        if data.get("error"):
            print(f"  error: {data['error']['message']}")
        if data.get("artifacts_dir"):
            print(f"  artifacts: {data['artifacts_dir']}")
        status = status or (0 if data.get("success") else 1)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

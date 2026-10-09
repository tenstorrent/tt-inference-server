# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec-test wrapper for MiniMax-Provider-Verifier.

The verifier is not vendored: the MINIMAX_VERIFIER workflow venv exports it
from a branch (``workflows/workflow_venvs.py``), without git metadata. This test runs one of its
suites with that venv's python and grades the raw output itself
(``minimax_verifier_grading``), so thresholds live in the suite files like the
other spec tests'. One test case runs one suite and becomes one Block:

* ``verify``: ``verify.py`` replays 102 recorded agentic conversations; the
  tool-call metrics (match rate, trigger similarity against MiniMax's official
  deployment, schema accuracy, reasoning-only responses, language following,
  scenario check) are graded against the verifier README's thresholds, which
  ``targets`` override per metric (e.g. ``tool_calls_schema_accuracy``).
* ``text`` / ``stream`` / ``image`` / ``video`` / ``reasoning_effort``: an
  ``m3_format_check`` pytest suite for the chat-completions contract, graded on
  its pass rate (``pass_rate_threshold``, 100% by default) with skipped and
  xfailed tests excluded and waived failures taken out.

Status: PASS / FAIL from the grading; ERROR when the suite produced no verdict
(no output, nothing ran, or the server was never reached).

``non_blocking`` (with an optional ``non_blocking_reason``) marks the run as
informational: acceptance waives its FAIL. The suites use it for the format
checks.

Settings come from ``test_config``; an environment variable
``MINIMAX_VERIFIER_<KEY>`` overrides the key of the same name (e.g.
``MINIMAX_VERIFIER_VERIFY_LIMIT=10``), so a CI dispatch can change a run
without editing the suite files. ``suite`` and ``non_blocking`` are
``test_config`` only: an override would reach every case.

``MINIMAX_VERIFIER_SUITES`` selects which suites run (comma-separated, e.g.
``verify``); the other cases report SKIP. Unset, every enrolled suite runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import tempfile
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.schema import Block

from .._test_common import SkipTest

from . import minimax_verifier_grading as grading
from .vllm_param_conformance_test import VLLMParamConformanceTest, _truncate_message

logger = logging.getLogger(__name__)

VERIFY_SUITE = "verify"
# m3_format_check suite -> pytest file (run from that directory, so its
# pytest.ini and conftest.py apply).
PYTEST_SUITES = {
    "text": "m3_text_tests.py",
    "stream": "m3_stream_tests.py",
    "image": "m3_image_tests.py",
    "video": "m3_video_tests.py",
    "reasoning_effort": "m3_a_reasoning_effort_tests.py",
}
SUITES = (VERIFY_SUITE, *PYTEST_SUITES)
FORMAT_CHECK_DIR = "m3_format_check"
VERIFY_SAMPLE = "sample.jsonl"

THRESHOLD_KEY = "pass_rate_threshold"
DEFAULT_PASS_RATE_THRESHOLD = 1.0
DEFAULT_WORKERS = 20
# Placeholder key for an unauthenticated server: the clients require one.
NO_AUTH_KEY = "no-auth"
# Without a key the endpoint is tested unauthenticated, so the checks that a
# missing / invalid key gets HTTP 401 cannot pass.
NO_AUTH_WAIVERS = {
    r"TestErrorCodes::test_20_0[57]_": "tested without authentication (no API key), "
    "so the 401 checks cannot pass",
}
# Where each suite's raw output is kept under the context's output path.
ARTIFACT_DIR_NAME = "minimax_verifier"
# Block logs keep the tail of the child's output; the full output is in the
# suite's log file next to the other artifacts.
LOG_TAIL_LINES = 200
# Must match report_module.acceptance_criteria.NON_BLOCKING_KEY / _REASON_KEY.
NON_BLOCKING_KEY = "non_blocking"
NON_BLOCKING_REASON_KEY = "non_blocking_reason"

ENV_PREFIX = "MINIMAX_VERIFIER_"
SUITES_ENV = "MINIMAX_VERIFIER_SUITES"
_FALSE_STRINGS = ("", "0", "false", "no", "off")
# tqdm (verify.py) and pytest -q ([ 45%]) progress, relayed at most once per
# PROGRESS_STEP percent.
_PROGRESS_RE = re.compile(r"(\d+)%\|.*\| *(\d+/\d+)|\[ *(\d+)%\]$")
PROGRESS_STEP = 10


def _setting(config: Mapping[str, Any], key: str, kind: str = "value") -> Any:
    """``MINIMAX_VERIFIER_<KEY>`` from the environment, else ``config[key]``."""
    raw = os.environ.get(ENV_PREFIX + key.upper())
    if raw is None:
        return config.get(key)
    logger.info(
        "MiniMax verifier setting %s overridden by %s%s", key, ENV_PREFIX, key.upper()
    )
    if kind == "bool":
        return raw.strip().lower() not in _FALSE_STRINGS
    if kind == "json":
        return json.loads(raw) if raw.strip() else None
    return raw


def _non_negative_int(config: Mapping[str, Any], key: str, default: int) -> int:
    raw = _setting(config, key)
    value = default if raw in (None, "") else int(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


class MiniMaxProviderVerifierTest(VLLMParamConformanceTest):
    """Run one MiniMax-Provider-Verifier suite and grade it here.

    Subclasses :class:`VLLMParamConformanceTest` for its model-name resolution;
    the children are the verifier's ``verify.py`` / pytest suites rather than
    one of this repo's pytest suites.
    """

    KIND = "minimax_provider_verifier"
    # "functional" so acceptance_criteria counts it (see VLLMParamConformanceTest).
    TASK_TYPE = "functional"

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        suite = self._suite()
        self._skip_unless_selected(suite)
        python, verifier_dir = self._provision_verifier()
        model_name = self._resolve_model_name()
        token = self._auth_token()

        artifacts = self._artifact_dir(suite)
        with ExitStack() as stack:
            out_dir = artifacts or Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="minimax_"))
            )
            if suite == VERIFY_SUITE:
                result = await self._run_verify(
                    python, verifier_dir, model_name, token, out_dir
                )
            else:
                result = await self._run_pytest(
                    suite, python, verifier_dir, model_name, token, out_dir
                )

        return {
            "suite": suite,
            "base_url": f"{self.base_url}/v1",
            "model_name": model_name,
            "verifier_commit": self._verifier_commit(verifier_dir),
            "artifacts_dir": str(artifacts) if artifacts else None,
            **result,
            **self._non_blocking_fields(),
        }

    # -- verify.py ---------------------------------------------------------

    async def _run_verify(
        self,
        python: str,
        verifier_dir: Path,
        model_name: str,
        token: Optional[str],
        out_dir: Path,
    ) -> Dict[str, Any]:
        sample = self._verify_sample(verifier_dir, out_dir)
        loops = max(1, _non_negative_int(self.config, "verify_loops", 1))
        workers = _non_negative_int(self.config, "workers", DEFAULT_WORKERS)
        env = {**self._child_env(), "OPENAI_API_KEY": token or NO_AUTH_KEY}

        per_run, failures, n_cases = [], [], 0
        baseline_path = self._baseline_path(verifier_dir)
        baseline = grading.baseline_tool_calls(baseline_path) if baseline_path else None
        for run in range(1, loops + 1):
            suffix = f"_run{run:02d}" if loops > 1 else ""
            results = out_dir / f"verify_results{suffix}.jsonl"
            summary = out_dir / f"verify_summary{suffix}.json"
            command = [
                python,
                "verify.py",
                str(sample),
                "--model",
                model_name,
                "--base-url",
                f"{self.base_url}/v1",
                "--concurrency",
                str(workers),
                "--output",
                str(results),
                "--summary",
                str(summary),
            ]
            await self._run_child(
                command, verifier_dir, env, out_dir / f"verify{suffix}.log"
            )
            rows = grading.load_jsonl(results) if results.exists() else []
            if not rows:
                logger.warning("verify.py run %d produced no results", run)
                continue
            summary_data = json.loads(summary.read_text()) if summary.exists() else None
            per_run.append(grading.verify_metrics(rows, summary_data, baseline))
            failures += grading.verify_case_failures(rows, run if loops > 1 else None)
            n_cases = n_cases or len(rows)
            if not any(r.get("status") == grading.SUCCESS_STATUS for r in rows):
                raise RuntimeError(
                    "server unreachable: every verify.py request failed, e.g. "
                    f"{str((rows[0].get('response') or {}).get('error', ''))[:300]}"
                )

        if not per_run:
            raise RuntimeError("verify.py produced no results (see verify.log)")

        thresholds = {
            key: self.targets[key]
            for key in grading.VERIFY_METRICS
            if self.targets.get(key) is not None
        }
        graded = grading.grade_verify(grading.mean_metrics(per_run), thresholds)
        # Surface each gate next to its measurement on the Block's targets
        # (copied: the dict passed in belongs to the suite definition).
        self.targets = {
            **self.targets,
            **{
                m["key"]: thresholds.get(m["key"], grading.VERIFY_METRICS[m["key"]][2])
                for m in graded["metrics"]
            },
        }
        logger.info(
            "MiniMax verifier verify: %s%s",
            "PASS" if graded["success"] else "FAIL",
            f" ({', '.join(graded['failed_metrics'])})"
            if graded["failed_metrics"]
            else "",
        )
        return {
            "verify_cases": n_cases,
            "verify_runs": len(per_run),
            "verify_baseline": (
                str(baseline_path.relative_to(verifier_dir)) if baseline_path else None
            ),
            "verify_metrics": graded["metrics"],
            "failed_metrics": graded["failed_metrics"],
            "case_failures": [
                {**f, "detail": _truncate_message(f["detail"])} for f in failures
            ],
            "success": graded["success"],
        }

    def _verify_sample(self, verifier_dir: Path, out_dir: Path) -> Path:
        """sample.jsonl, or its first ``verify_limit`` cases."""
        sample = verifier_dir / VERIFY_SAMPLE
        limit = _non_negative_int(self.config, "verify_limit", 0)
        if not limit:
            return sample
        subset = out_dir / f"verify_sample_first_{limit}.jsonl"
        with open(sample, encoding="utf-8") as src, open(
            subset, "w", encoding="utf-8"
        ) as dst:
            for i, line in enumerate(src):
                if i >= limit:
                    break
                dst.write(line)
        return subset

    def _baseline_path(self, verifier_dir: Path) -> Optional[Path]:
        """The official deployment's results for ToolCalls-Trigger-Similarity:
        the first ``*_results.jsonl`` under ``verify_baseline`` (a directory in
        the checkout, which must be in its sparse paths)."""
        folder = _setting(self.config, "verify_baseline")
        if not folder:
            return None
        files = sorted((verifier_dir / folder).rglob("*_results.jsonl"))
        if not files:
            raise RuntimeError(f"no *_results.jsonl baseline under {folder}")
        return files[0]

    # -- m3_format_check pytest suites --------------------------------------

    async def _run_pytest(
        self,
        suite: str,
        python: str,
        verifier_dir: Path,
        model_name: str,
        token: Optional[str],
        out_dir: Path,
    ) -> Dict[str, Any]:
        suite_dir = verifier_dir / FORMAT_CHECK_DIR
        junit = out_dir / f"{suite}.xml"
        workers = _non_negative_int(self.config, "workers", DEFAULT_WORKERS)
        command = [
            python,
            "-m",
            "pytest",
            PYTEST_SUITES[suite],
            "-p",
            "no:cacheprovider",
            "-q",
            "--tb=short",
            f"--junitxml={junit}",
            *(["-n", str(workers)] if workers > 1 else []),
            *([] if self._include_slow() else ["-m", "not slow"]),
            *shlex.split(str(_setting(self.config, "pytest_args") or "")),
        ]
        env = {
            **self._child_env(),
            # The suites append /v1 themselves.
            "M3_BASE_URL": self.base_url,
            "M3_API_KEY": token or NO_AUTH_KEY,
            "M3_AUTH_TYPE": "bearer" if token else "none",
            "M3_MODEL": model_name,
        }
        started = time.time()
        return_code = await self._run_child(
            command, suite_dir, env, out_dir / f"{suite}.log"
        )
        self._collect_request_logs(suite_dir / "logs", started, out_dir / "logs")

        if not junit.exists():
            raise RuntimeError(
                f"pytest wrote no JUnit report (exit code {return_code}); "
                f"see {suite}.log"
            )
        results, collection_error = grading.parse_junit(junit)
        if collection_error:
            raise RuntimeError(f"{suite} suite did not collect: {collection_error}")

        waivers = self._waivers(token)
        graded = grading.grade_pytest(results, self._resolve_threshold(), waivers)
        if not graded["passed"] + graded["failed"] + graded["errors"]:
            raise RuntimeError(f"{suite} suite executed no tests")
        self.targets = {**self.targets, THRESHOLD_KEY: graded["threshold"]}
        logger.info(
            "MiniMax verifier %s: pass rate %s (%d passed, %d failed, %d waived) "
            "vs threshold %s -> %s",
            suite,
            graded["pass_rate_percent"],
            graded["passed"],
            graded["failed"] + graded["errors"],
            graded["waived"],
            graded["threshold_percent"],
            "PASS" if graded["success"] else "FAIL",
        )
        for key in ("failures", "waived_failures"):
            graded[key] = [
                {**f, "message": _truncate_message(f["message"])} for f in graded[key]
            ]
        return graded

    def _include_slow(self) -> bool:
        """Run the cases marked slow (512k/1M-token prompts, long videos);
        on unless set false."""
        value = _setting(self.config, "include_slow", "bool")
        return True if value is None else bool(value)

    def _waivers(self, token: Optional[str]) -> List[Tuple[str, str]]:
        """``waived_tests`` ({test-id regex: reason}), plus the 401 checks when
        the server is tested without a key."""
        configured = _setting(self.config, "waived_tests", "json") or {}
        if not isinstance(configured, dict):
            raise ValueError(f"waived_tests must be an object, got {configured!r}")
        waivers = dict(configured)
        if not token:
            waivers.update(NO_AUTH_WAIVERS)
        return list(waivers.items())

    def _resolve_threshold(self) -> float:
        """targets.pass_rate_threshold (per-model override) > test_config > 1.0."""
        raw = self.targets.get(THRESHOLD_KEY)
        if raw is None:
            raw = self.config.get(THRESHOLD_KEY, DEFAULT_PASS_RATE_THRESHOLD)
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{THRESHOLD_KEY} must be in [0, 1], got {threshold}")
        return threshold

    @staticmethod
    def _collect_request_logs(source: Path, since: float, dest: Path) -> None:
        """Move the request logs the suite wrote into the checkout
        (m3_format_check/logs, one file per xdist worker) to the artifacts."""
        if not source.is_dir():
            return
        for path in source.glob("run_*.jsonl"):
            if path.stat().st_mtime >= since:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), dest / path.name)

    # -- shared ------------------------------------------------------------

    def _suite(self) -> str:
        suite = str(self.config.get("suite") or VERIFY_SUITE)
        if suite not in SUITES:
            raise ValueError(f"suite must be one of {SUITES}, got {suite!r}")
        return suite

    @staticmethod
    def _skip_unless_selected(suite: str) -> None:
        """SKIP this case when ``MINIMAX_VERIFIER_SUITES`` leaves its suite out."""
        raw = os.environ.get(SUITES_ENV)
        if raw is None or not raw.strip():
            return
        selected = raw.replace(",", " ").split()
        unknown = sorted(set(selected) - set(SUITES))
        if unknown:
            raise ValueError(f"{SUITES_ENV} names unknown suites {unknown}")
        if suite not in selected:
            raise SkipTest(f"suite {suite} not selected ({SUITES_ENV}={raw})")

    def _provision_verifier(self) -> Tuple[str, Path]:
        """The MINIMAX_VERIFIER venv's python and the verifier checkout in it."""
        from workflow_module.engine_types import WorkflowVenvType
        from workflow_module.venv_provisioner import get_venv_provisioner
        from workflows.workflow_venvs import MINIMAX_VERIFIER_DIR_NAME

        provisioner = get_venv_provisioner()
        venv_type = WorkflowVenvType.MINIMAX_VERIFIER
        model_spec = getattr(self.ctx, "model_spec", None)
        if not provisioner.provision(venv_type, model_spec):
            raise RuntimeError("Failed to provision the MiniMax verifier venv")
        verifier_dir = provisioner.venv_path(venv_type) / MINIMAX_VERIFIER_DIR_NAME
        return provisioner.venv_python(venv_type), verifier_dir

    @staticmethod
    def _verifier_commit(verifier_dir: Path) -> Optional[str]:
        """The commit the verifier files were exported from."""
        from workflows.workflow_venvs import read_source_commit

        return read_source_commit(verifier_dir)

    @staticmethod
    def _auth_token() -> Optional[str]:
        """The bearer token this repo's chat-completions suites send:
        OPENAI_API_KEY / API_KEY, else a JWT minted from JWT_SECRET."""
        from test_fixtures.conftest import _get_bearer_token

        return _get_bearer_token()

    @staticmethod
    def _child_env() -> Dict[str, str]:
        # PYTEST_* from an enclosing pytest (e.g. PYTEST_ADDOPTS) must not
        # leak into the verifier's own pytest runs.
        return {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}

    def _artifact_dir(self, suite: str) -> Optional[Path]:
        """``<output>/minimax_verifier/<suite>``, emptied for this run; None
        without an output path (then a temporary directory is used)."""
        output_path = getattr(self.ctx, "output_path", None)
        if not output_path:
            return None
        path = Path(output_path) / ARTIFACT_DIR_NAME / suite
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        return path

    async def _run_child(
        self, command: Sequence[str], cwd: Path, env: Dict[str, str], log_path: Path
    ) -> Optional[int]:
        """Run a verifier process, writing its output to ``log_path``, relaying
        progress, and keeping the tail for the Block's logs."""
        logger.info("Running MiniMax verifier: %s (in %s)", " ".join(command), cwd)
        self._progress_seen = -PROGRESS_STEP
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            with open(log_path, "wb") as log:
                output = await self._stream_child(process, log)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        lines = output.decode(errors="replace").replace("\r", "\n").splitlines()
        self.logs.extend(line for line in lines[-LOG_TAIL_LINES:] if line.strip())
        logger.info("MiniMax verifier process exited with code %s", process.returncode)
        return process.returncode

    async def _stream_child(self, process, log) -> bytes:
        chunks: List[bytes] = []
        pending = b""
        while True:
            chunk = await process.stdout.read(65536)
            if not chunk:
                break
            log.write(chunk)
            chunks.append(chunk)
            # tqdm redraws its bar with \r, so both end a progress line.
            *lines, pending = re.split(rb"[\r\n]", pending + chunk)
            for line in lines:
                self._on_child_line(line.decode(errors="replace"))
        await process.wait()
        return b"".join(chunks)

    def _on_child_line(self, line: str) -> None:
        match = _PROGRESS_RE.search(line.strip())
        if not match:
            return
        percent = int(match.group(1) or match.group(3))
        if percent >= self._progress_seen + PROGRESS_STEP or percent == 100:
            if percent != self._progress_seen:
                self._progress_seen = percent
                logger.info("[minimax-verifier progress] %s", line.strip()[-120:])

    def _block(self, data: Dict[str, Any]) -> Block:
        """Name the suite in the title: a model runs this test once per suite."""
        block = super()._block(data)
        # The raw setting, unvalidated: this also titles the ERROR block of a
        # run whose suite was rejected.
        suite = self.config.get("suite") or VERIFY_SUITE
        label = f"MiniMax Provider Verifier — {suite}"
        if self._non_blocking_fields():
            label += " (non-blocking)"
        return replace(block, title=label)

    def _non_blocking_fields(self) -> Dict[str, Any]:
        """Block-data keys acceptance reads to waive this run's FAIL. Read from
        test_config only: a MINIMAX_VERIFIER_* override would reach every run."""
        if self.config.get(NON_BLOCKING_KEY) is not True:
            return {}
        return {
            NON_BLOCKING_KEY: True,
            NON_BLOCKING_REASON_KEY: str(
                self.config.get(NON_BLOCKING_REASON_KEY) or "non-blocking run"
            ),
        }

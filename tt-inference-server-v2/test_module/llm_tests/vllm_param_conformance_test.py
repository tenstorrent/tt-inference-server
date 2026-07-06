# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec-test wrappers around the vLLM parameter-conformance suites.

Each wrapper runs its pytest suite (``test_vllm_chat_completions.py`` or
``test_vllm_responses.py``) in a child pytest process — the suites contain a
``@pytest.mark.asyncio`` test that needs its own loop, which would collide
with ``BaseTest.run_tests``' ``asyncio.run`` — then reshapes the report into
the ``parameter_conformance_summary`` / ``detailed_test_results`` list fields
on ``Block.data`` so the generic renderer draws them as sub-tables with no
dedicated renderer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from report_module.schema import Block

from .._test_common import BaseTest, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

logger = logging.getLogger(__name__)

PASSED_STATUS = "passed"
FAILED_STATUS = "failed"
DEFAULT_MODEL_NAME = "unknown-model"
MESSAGE_MAX_LEN = 250

# test_module/llm_tests/<this file> -> parents[2] is the v2 package root.
_V2_ROOT = Path(__file__).resolve().parents[2]
_LLM_MODULE_DIR = _V2_ROOT / "llm_module"


def _truncate_message(message: Any) -> str:
    """Collapse a failure message to a single, table-safe, bounded string.

    Pipe/newline escaping is handled by the markdown table builder; here we
    only cap length so a long traceback doesn't dominate the report.
    """
    text = str(message)
    if len(text) > MESSAGE_MAX_LEN:
        return text[:MESSAGE_MAX_LEN] + "..."
    return text


class VLLMParamConformanceTest(BaseTest):
    """Run the vLLM chat-completions parameter suite and wrap its report.

    Subclasses point at a different suite/endpoint by overriding the three
    ``PYTEST_*`` / ``ENDPOINT_PATH`` / ``REPORT_TASK_NAME`` attributes.
    """

    KIND = "vllm_chat_completions"
    # "functional" (not the inherited "infra") so acceptance_criteria counts
    # this block: infra task types are skipped by _check_spec_tests.
    TASK_TYPE = "functional"

    PYTEST_FILENAME = "test_vllm_chat_completions.py"
    ENDPOINT_PATH = "/v1/chat/completions"
    REPORT_TASK_NAME = "vllm_chat_completions"

    async def _run_specific_test_async(self) -> Dict[str, Any]:
        endpoint_url = f"{self.base_url}{self.ENDPOINT_PATH}"
        model_name = self._resolve_model_name()

        with tempfile.TemporaryDirectory(prefix="vllm_param_") as tmp_dir:
            report_data = await self._run_pytest_suite(
                tmp_dir, endpoint_url, model_name
            )

        results = report_data.get("results", {})
        return {
            "endpoint_url": report_data.get("endpoint_url", endpoint_url),
            "model_name": report_data.get("model_name", model_name),
            "task_name": report_data.get("task_name", self.REPORT_TASK_NAME),
            "parameter_conformance_summary": self._build_conformance_summary(results),
            "detailed_test_results": self._build_detailed_results(results),
            "success": self._all_passed(results),
        }

    async def _run_pytest_suite(
        self, output_dir: str, endpoint_url: str, model_name: str
    ) -> Dict[str, Any]:
        pytest_file = _LLM_MODULE_DIR / self.PYTEST_FILENAME
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(pytest_file),
            "--output-path",
            output_dir,
            "--task-name",
            self.REPORT_TASK_NAME,
            "--endpoint-url",
            endpoint_url,
            "--model-name",
            model_name,
            "-q",
        ]
        logger.info("Running vLLM parameter suite: %s", " ".join(command))

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(_V2_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await process.communicate()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

        self._record_pytest_output(process.returncode, stdout)

        # A nonzero return code is expected when conformance tests fail — those
        # failures are captured in the report. Only a missing report means the
        # run itself broke (e.g. missing fixtures / server unreachable).
        report_path = (
            Path(output_dir) / f"parameter_report_{self.REPORT_TASK_NAME}.json"
        )
        if not report_path.exists():
            tail = (stdout or b"").decode(errors="replace")[-2000:]
            raise RuntimeError(
                f"pytest produced no report (return code {process.returncode}). "
                f"Output tail:\n{tail}"
            )

        return json.loads(report_path.read_text())

    def _record_pytest_output(
        self, return_code: Optional[int], stdout: Optional[bytes]
    ) -> None:
        text = (stdout or b"").decode(errors="replace")
        for line in text.splitlines():
            self.logs.append(line)
        logger.info("vLLM parameter suite exited with code %s", return_code)

    def _resolve_model_name(self) -> str:
        if self.ctx is not None:
            model_name = getattr(self.ctx.model_spec, "model_name", None)
            if model_name:
                return str(model_name)
        return str(self.config.get("model") or DEFAULT_MODEL_NAME)

    @staticmethod
    def _all_passed(results: Mapping[str, List[Mapping[str, Any]]]) -> bool:
        return all(
            test.get("status") == PASSED_STATUS
            for tests in results.values()
            for test in tests
        )

    @staticmethod
    def _build_conformance_summary(
        results: Mapping[str, List[Mapping[str, Any]]],
    ) -> List[Dict[str, str]]:
        """One row per test case: pass/fail status and a "P/N passed" count."""
        rows: List[Dict[str, str]] = []
        for test_case in sorted(results):
            tests = results[test_case]
            total = len(tests)
            if not total:
                rows.append(
                    {
                        "test_case": test_case,
                        "status": "⚠️ SKIP",
                        "summary": "No tests run",
                    }
                )
                continue
            passed = sum(1 for t in tests if t.get("status") == PASSED_STATUS)
            failed = sum(1 for t in tests if t.get("status") == FAILED_STATUS)
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            rows.append(
                {
                    "test_case": test_case,
                    "status": status,
                    "summary": f"{passed}/{total} passed",
                }
            )
        return rows

    @staticmethod
    def _build_detailed_results(
        results: Mapping[str, List[Mapping[str, Any]]],
    ) -> List[Dict[str, str]]:
        """One row per parametrization; failures first, message only on failure."""
        rows: List[Dict[str, str]] = []
        for test_case in sorted(results):
            ordered = sorted(
                results[test_case],
                key=lambda t: (
                    t.get("status") == PASSED_STATUS,
                    t.get("test_node_name", ""),
                ),
            )
            for test in ordered:
                passed = test.get("status") == PASSED_STATUS
                rows.append(
                    {
                        "test_case": test_case,
                        "parametrization": str(test.get("test_node_name", "")),
                        "status": "✅ PASSED" if passed else "❌ FAILED",
                        "message": ""
                        if passed
                        else _truncate_message(test.get("message", "")),
                    }
                )
        return rows


class VLLMResponsesParamConformanceTest(VLLMParamConformanceTest):
    """Run the vLLM responses (``/v1/responses``) parameter suite."""

    KIND = "vllm_responses"
    PYTEST_FILENAME = "test_vllm_responses.py"
    ENDPOINT_PATH = "/v1/responses"
    REPORT_TASK_NAME = "vllm_responses"


def run_vllm_param_conformance(
    ctx: "MediaContext",
    targets: Optional[dict] = None,
    test_cls: type[VLLMParamConformanceTest] = VLLMParamConformanceTest,
) -> Block:
    """Run a vLLM parameter-conformance test under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 3600,
            "retry_attempts": 0,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    return test_cls(test_config, targets or {}, ctx=ctx).run_tests()

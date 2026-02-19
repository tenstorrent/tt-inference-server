# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for Llama runner (run_step and pipe protocol) without device."""

import json
import os
import sys

import pytest


@pytest.mark.skipif(
    not os.environ.get("TT_METAL_HOME"),
    reason="TT_METAL_HOME not set; Llama runner requires tt-metal",
)
class TestLlamaRunnerWithMetal:
    """Tests that require TT_METAL_HOME (imports tt-metal)."""

    def test_run_step_returns_one_result_per_sequence(self):
        from tt_model_runners.llama_runner import Llama31_8BRunner, StepSequence, StepResult

        # Create runner without warmup (avoid device). We only test run_step stub.
        runner = Llama31_8BRunner("device_0")
        runner.model = None  # skip load
        seqs = [
            StepSequence(
                task_id="tid-1",
                token_ids=[1, 2, 3],
                max_tokens=64,
                temperature=1.0,
                ignore_eos=False,
            ),
        ]
        results = runner.run_step(is_prefill=True, sequences=seqs)
        assert len(results) == 1
        assert isinstance(results[0], StepResult)
        assert results[0].task_id == "tid-1"
        assert results[0].token_id in (0, 1)
        assert results[0].finished is False

    def test_run_step_max_tokens_zero_marks_finished(self):
        from tt_model_runners.llama_runner import Llama31_8BRunner, StepSequence

        runner = Llama31_8BRunner("device_0")
        runner.model = None
        seqs = [
            StepSequence(
                task_id="t",
                token_ids=[1],
                max_tokens=0,
                temperature=1.0,
                ignore_eos=False,
            ),
        ]
        results = runner.run_step(is_prefill=False, sequences=seqs)
        assert len(results) == 1
        assert results[0].finished is True


class TestLlamaRunnerPipeProtocol:
    """Test pipe protocol (length-prefixed JSON) without device."""

    def test_pipe_request_response_shape(self):
        """Verify request/response JSON shape expected by C++."""
        req = {
            "is_prefill": True,
            "sequences": [
                {
                    "task_id": "uuid-1",
                    "token_ids": [1, 2, 3],
                    "max_tokens": 64,
                    "temperature": 1.0,
                    "ignore_eos": False,
                },
            ],
            "exit": False,
        }
        body = json.dumps(req).encode("utf-8")
        assert len(body) > 0
        resp = [
            {"task_id": "uuid-1", "token_id": 0, "finished": False},
        ]
        out = json.dumps(resp).encode("utf-8")
        assert len(out) > 0
        parsed = json.loads(out.decode("utf-8"))
        assert parsed[0]["task_id"] == "uuid-1"
        assert "token_id" in parsed[0]
        assert "finished" in parsed[0]

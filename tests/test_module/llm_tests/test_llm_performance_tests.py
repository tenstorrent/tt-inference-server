# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``run_llm_performance`` hands every sweep point to the accumulator as it lands.

Checkpointing is the accumulator hook's job (``WorkflowExecution.run``): it
renders with the workflow's metadata. A second, direct render from the runner
would overwrite that checkpoint with one missing ``workflow``/``run_command``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_module.runner import RunnerResult
from report_module.schema import Block
from test_module.llm_tests import llm_performance_tests as lpt
from workflow_module import get_default_accumulator


def _block(isl: int) -> Block:
    return Block(kind="benchmarks", title="B", data={"input_sequence_length": isl})


class _FakeRunner:
    """Emits two sweep points through ``on_block``, like the real runner."""

    def __init__(self, *_a, **_k):
        pass

    def run(self, configs, server, context, *, on_block=None, **_k):
        result = RunnerResult()
        for isl in (128, 1024):
            block = _block(isl)
            result.blocks.append(block)
            on_block(block)
        return result


def _ctx(tmp_path: Path) -> MagicMock:
    ctx = MagicMock()
    ctx.remote_server = False
    ctx.server_host = "http://127.0.0.1"
    ctx.server_port = 8000
    ctx.output_path = str(tmp_path / "reports_output" / "benchmarks" / "out")
    ctx.device.name = "SUPER_CLUSTER"
    ctx.model_spec.metadata = {}
    ctx.model_spec.hf_model_repo = "zai-org/GLM-5.3"
    ctx.model_spec.model_name = "GLM-5.3"
    return ctx


@pytest.fixture
def accumulator():
    acc = get_default_accumulator()
    acc.clear()
    yield acc
    acc.clear()


def test_each_sweep_point_is_accepted_and_the_runner_renders_nothing(
    tmp_path, monkeypatch, accumulator
):
    monkeypatch.setattr(lpt, "LLMPerformanceRunner", _FakeRunner)

    lpt.run_llm_performance(
        _ctx(tmp_path), driver=MagicMock(), configs=[], server_controller=MagicMock()
    )

    assert [b.data["input_sequence_length"] for b in accumulator.blocks] == [128, 1024]
    # No hook installed here, so any report on disk was rendered by the runner.
    assert not list(tmp_path.rglob("report_*"))

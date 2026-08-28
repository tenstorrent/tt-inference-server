# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

import pytest

from scripts.release.plan_agentic_external_run import (
    ContractError,
    build_contract,
    write_plan,
)


def _gpt120(**overrides):
    values = {
        "model": "gpt-oss-120b",
        "device": "p300x2",
        "task_name": "swe_bench_verified",
        "limit_samples_mode": "ci-nightly",
        "admitted_max_input_tokens": 92 * 1024,
        "admitted_max_context_tokens": 128 * 1024,
        "server_url": "http://qb2-120-p06t07",
        "service_port": 18091,
    }
    values.update(overrides)
    return build_contract(**values)


def test_gpt120_plan_pins_exact_swe_shape_and_writes_argv(tmp_path):
    contract, model_spec = _gpt120()
    assert contract.concurrency == 1
    assert contract.max_input_tokens == 92 * 1024
    assert contract.max_output_tokens == 32 * 1024
    assert contract.required_context_tokens == 124 * 1024
    assert contract.catalog_max_context_tokens == 128 * 1024
    assert len(contract.instance_ids) == 5

    plan_path, runtime_path, command = write_plan(contract, model_spec, tmp_path)
    plan = json.loads(plan_path.read_text())
    runtime = json.loads(runtime_path.read_text())
    assert plan["argv"] == command
    assert command[command.index("--model") + 1] == "gpt-oss-120b"
    assert command[command.index("--server-url") + 1] == "http://qb2-120-p06t07"
    assert runtime["runtime_config"]["limit_samples_mode"] == "ci-nightly"
    assert runtime["runtime_config"]["workflow"] == "agentic"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("admitted_max_input_tokens", 92 * 1024 - 1, "input tokens"),
        ("admitted_max_context_tokens", 124 * 1024 - 1, "total tokens"),
    ],
)
def test_gpt120_plan_rejects_under_admitted_artifact(field, value, match):
    with pytest.raises(ContractError, match=match):
        _gpt120(**{field: value})


def test_plan_rejects_non_agentic_task():
    with pytest.raises(ContractError, match="not an agentic eval"):
        _gpt120(task_name="mmlu_generative")

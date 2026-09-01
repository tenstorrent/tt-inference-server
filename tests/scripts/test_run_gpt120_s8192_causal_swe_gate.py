# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/release/run_gpt120_s8192_causal_swe_gate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("gpt_causal_swe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_causal_arms_differ_only_by_reasoning_effort():
    module = _module()
    default = module.completion_kwargs("default")
    high = module.completion_kwargs("high")
    assert default == {"seed": 42}
    assert high == {"seed": 42, "reasoning_effort": "high"}
    assert {
        key: value for key, value in high.items() if key != "reasoning_effort"
    } == default
    assert module.MAX_INPUT + module.MAX_OUTPUT == module.MAX_CONTEXT == 8192
    assert module.STEP_LIMIT == 16
    assert module.OBSERVATION_RETAINED_PAYLOAD_CHARS == 2048
    assert module.INSTANCE_IDS == ["django__django-11299"]
    assert len(module.DATASET_REVISION) == 40


def test_bounded_workflow_is_generic_and_enforces_preregistered_discipline():
    module = _module()
    template = module.BOUNDED_INSTANCE_TEMPLATE
    assert "{{task}}" in template
    assert "do not\nrun recursive repository listings" in template
    assert "Avoid rereading the same file range" in template
    assert "By turn 8" in template
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in template
    assert "constraints.py" not in template
    assert "CheckConstraint" not in template

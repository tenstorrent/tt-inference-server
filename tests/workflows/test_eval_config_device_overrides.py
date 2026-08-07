# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for EvalTask per-device overrides (3-tier device-variance schema)."""

import pytest

from reference_config.evals.eval_config import EvalTask, resolve_task_for_device


class _FakeDevice:
    def __init__(self, name):
        self.name = name


def test_no_overrides_returns_task_unchanged():
    task = EvalTask(task_name="t")
    assert resolve_task_for_device(task, "GALAXY") is task
    assert resolve_task_for_device(task, None) is task


def test_matching_device_applies_tier2_overrides():
    task = EvalTask(
        task_name="t",
        max_concurrent=32,
        device_overrides={
            "SUPER_CLUSTER": {
                "max_concurrent": 38,
                "use_chat_api": True,
                "gen_kwargs": {"stream": "True"},
            }
        },
    )
    resolved = resolve_task_for_device(task, "SUPER_CLUSTER")
    assert resolved.max_concurrent == 38
    assert resolved.use_chat_api is True
    assert resolved.gen_kwargs == {"stream": "True"}
    # tier-1 identity untouched
    assert resolved.task_name == "t"
    # base task unchanged (frozen dataclass)
    assert task.max_concurrent == 32


def test_device_enum_and_case_insensitive_matching():
    task = EvalTask(
        task_name="t",
        device_overrides={"galaxy": {"max_concurrent": 16}},
    )
    assert resolve_task_for_device(task, _FakeDevice("GALAXY")).max_concurrent == 16
    assert resolve_task_for_device(task, "Galaxy").max_concurrent == 16


def test_non_matching_device_returns_task_unchanged():
    task = EvalTask(
        task_name="t",
        device_overrides={"SUPER_CLUSTER": {"max_concurrent": 38}},
    )
    assert resolve_task_for_device(task, "GALAXY") is task


def test_tier1_override_rejected():
    with pytest.raises(ValueError, match="device-invariant"):
        EvalTask(
            task_name="t",
            device_overrides={"GALAXY": {"task_name": "other"}},
        )


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown EvalTask fields"):
        EvalTask(
            task_name="t",
            device_overrides={"GALAXY": {"not_a_field": 1}},
        )


def test_chat_api_override_reinfers_eval_class():
    task = EvalTask(
        task_name="t",
        device_overrides={"SUPER_CLUSTER": {"use_chat_api": True}},
    )
    assert task.eval_class == "local-completions"
    resolved = resolve_task_for_device(task, "SUPER_CLUSTER")
    assert resolved.eval_class == "local-chat-completions"

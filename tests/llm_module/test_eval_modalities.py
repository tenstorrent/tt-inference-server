# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Vision eval tasks run only on entries that serve images."""

from types import SimpleNamespace

from llm_module.eval_configs import tasks_for_modalities


def _task(name, modalities=None):
    return SimpleNamespace(task_name=name, modalities=modalities)


TEXT = _task("mmlu_pro")
VISION = _task("chartqa", ["image"])


def test_text_only_entry_skips_vision_tasks():
    assert tasks_for_modalities([TEXT, VISION], ["text"]) == [TEXT]


def test_entry_without_declared_modalities_is_text_only():
    assert tasks_for_modalities([TEXT, VISION], None) == [TEXT]


def test_vision_entry_keeps_both():
    assert tasks_for_modalities([TEXT, VISION], ["text", "image"]) == [TEXT, VISION]


def test_task_without_the_attribute_is_text():
    legacy = SimpleNamespace(task_name="ifeval")
    assert tasks_for_modalities([legacy], ["text"]) == [legacy]

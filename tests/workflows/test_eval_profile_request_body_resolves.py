# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Every dev catalog profile must resolve the request body of every eval task it
runs: a profile's ``eval_task_request_body_overrides`` and the task's own
``thinking`` pin may not disagree (``resolve_request_body`` raises when they do,
which otherwise surfaces only when the server boots). Two implementations of
one checkpoint share the model's eval tasks, so a pin that is right for one
profile has to live in that profile's override rather than on the task."""

import pytest

from llm_module.eval_configs import _apply_profile_request_body_overrides
from llm_module.request_overrides import resolve_request_body
from reference_config.evals.eval_config import _eval_config_map
from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path


def _dev_specs():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [spec for template in templates for spec in template.expand_to_specs()]


def _profile_bodies(spec):
    tasks = _eval_config_map[spec.hf_model_repo].tasks
    tasks = _apply_profile_request_body_overrides(tasks, spec)
    return {task.task_name: resolve_request_body(task) for task in tasks}


def test_every_dev_profile_resolves_every_eval_request_body():
    specs = [spec for spec in _dev_specs() if spec.hf_model_repo in _eval_config_map]
    assert specs
    failures = []
    for spec in specs:
        try:
            _profile_bodies(spec)
        except ValueError as exc:
            failures.append(
                f"{spec.hf_model_repo} / {spec.impl.impl_name} / {spec.device_type}: {exc}"
            )
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("impl", "device", "thinking"),
    [
        ("gemma4-26b-a4b-qb2", "p300x2", False),
        ("tt-transformers", "p300x2", True),
        ("tt-transformers", "blackhole_galaxy", True),
    ],
)
def test_gemma4_26b_gpqa_thinking_mode_is_pinned_per_profile(impl, device, thinking):
    """The dedicated QB2 profile runs GPQA Diamond thinking off (its validated path);
    the tt-transformers profiles run it thinking on (the mode of the published and
    GPU references). The shared task carries no pin of its own."""
    specs = [s for s in _dev_specs() if s.hf_model_repo == "google/gemma-4-26B-A4B-it"]
    spec = resolve_model_spec(
        specs,
        model="google/gemma-4-26B-A4B-it",
        device=device,
        impl=impl,
        catalog_name="Shield",
    )
    gpqa = next(
        t
        for t in _eval_config_map["google/gemma-4-26B-A4B-it"].tasks
        if t.task_name == "r1_gpqa_diamond"
    )
    assert gpqa.thinking is None
    body = _profile_bodies(spec)["r1_gpqa_diamond"]
    assert body["chat_template_kwargs"]["enable_thinking"] is thinking

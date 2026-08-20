# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The LLM benchmark runner is text-only and drops every non-``text`` param,
while VLM image/text perf is driven separately by the guidellm ``omni_modal_image`` scenario over a real HF dataset (see ``test_guidellm_scenarios.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reference_config.benchmarking.benchmark_config import get_benchmark_config
from llm_module.benchmark_configs import get_llm_configs
from workflows.model_spec import MODEL_SPECS, load_templates_from_yaml
from workflows.utils import get_repo_root_path
from workflows.workflow_types import ModelType


def _text_keys(params):
    return {
        (p.isl, p.osl, p.max_concurrency, p.num_prompts)
        for p in params
        if p.task_type == "text"
    }


def _params_for(spec):
    """All benchmark params configured for ``spec`` on its own device."""
    bc = get_benchmark_config(spec)
    dev = spec.device_type
    return dev, [p for task in bc.tasks for p in task.param_map.get(dev, [])]


def _vlm_specs():
    seen = {}
    for spec in MODEL_SPECS.values():
        if spec.model_type != ModelType.VLM:
            continue
        if "image" not in spec.supported_modalities:
            continue
        # one spec per (model_name, device) is plenty for the invariant
        seen.setdefault((spec.model_name, spec.device_type.name), spec)
    return list(seen.values())


def _cfg_keys(configs):
    return {(c.isl, c.osl, c.max_concurrency, c.num_prompts) for c in configs}


def test_vlm_benchmark_keeps_only_text_params():
    """A representative VLM spec: text params survive; vlm/structured dropped."""
    spec = next(
        s
        for s in MODEL_SPECS.values()
        if s.model_type == ModelType.VLM and "image" in s.supported_modalities
    )
    dev, params = _params_for(spec)
    task_types = {p.task_type for p in params}

    # Guard: the spec really does carry the params that must be dropped, so the
    # filtering below is meaningful rather than vacuously true.
    assert "text" in task_types
    assert "vlm" in task_types, "expected image (vlm) benchmark params to drop"
    assert "structured_output" in task_types

    configs = get_llm_configs(spec, dev)

    # Exactly the (deduped) text params survive — nothing else leaks in.
    assert _cfg_keys(configs) == _text_keys(params)


@pytest.mark.parametrize(
    "spec", _vlm_specs(), ids=lambda s: f"{s.model_name}-{s.device_type.name}"
)
def test_vlm_configs_never_leak_non_text_params(spec):
    """Catalog-wide invariant: no VLM spec leaks non-text params into the sweep."""
    dev, params = _params_for(spec)
    non_text_only = {
        (p.isl, p.osl, p.max_concurrency, p.num_prompts)
        for p in params
        if p.task_type != "text"
    } - _text_keys(params)

    cfg_keys = _cfg_keys(get_llm_configs(spec, dev))

    assert cfg_keys <= _text_keys(params)
    assert cfg_keys.isdisjoint(non_text_only)


def _specs_with_text_targets():
    """One LLM spec per model that defines tiered targets for a text config."""
    seen = {}
    for spec in MODEL_SPECS.values():
        params = [
            p
            for task in get_benchmark_config(spec).tasks
            for p in task.param_map.get(spec.device_type, [])
            if p.task_type == "text" and p.targets
        ]
        if params:
            seen.setdefault(spec.model_name, (spec, params))
    return list(seen.values())[:5]


@pytest.mark.parametrize(
    "spec,targeted", _specs_with_text_targets(), ids=lambda v: getattr(v, "id", "")
)
def test_perf_reference_targets_reach_the_sweep_point(spec, targeted):
    """Targets ride along on the config so the runner can grade the run.

    They live on the spec's ``perf_reference`` params only; dropping them
    here is what left LLM benchmark blocks with no ``target_checks``.
    """
    configs = get_llm_configs(spec, spec.device_type)
    graded = {
        (c.isl, c.osl, c.max_concurrency): c.targets for c in configs if c.targets
    }

    for params in targeted:
        shape = (params.isl, params.osl, params.max_concurrency)
        assert shape in graded, f"targets lost for {shape}"
        assert graded[shape] == params.targets


def test_sweep_points_without_a_perf_reference_carry_no_targets():
    """Only configs the spec actually defines targets for are graded."""
    spec, targeted = _specs_with_text_targets()[0]
    shapes_with_targets = {(p.isl, p.osl, p.max_concurrency) for p in targeted}

    for config in get_llm_configs(spec, spec.device_type):
        shape = (config.isl, config.osl, config.max_concurrency)
        assert bool(config.targets) == (shape in shapes_with_targets)


def test_configs_stay_hashable_with_targets_attached():
    """``targets`` is compare=False; sweep-point identity is the numbers."""
    spec, _ = _specs_with_text_targets()[0]
    configs = get_llm_configs(spec, spec.device_type)
    assert len({hash(c) for c in configs}) == len(
        {(c.isl, c.osl, c.max_concurrency, c.num_prompts) for c in configs}
    )


def test_block_granular_spec_selects_custom_dataset_from_model_spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    spec = next(
        t for t in templates if t.weights == ["google/diffusiongemma-26B-A4B-it"]
    ).expand_to_specs()[0]
    configs = get_llm_configs(spec, spec.device_type)

    assert configs
    for config in configs:
        assert config.output_block_size == 256
        assert config.custom_dataset_path == Path(
            f"speed_bench_prompts_isl-{config.isl}_n-{config.num_prompts}.jsonl"
        )


def test_token_granular_specs_keep_random_dataset():
    spec, _ = _specs_with_text_targets()[0]
    for config in get_llm_configs(spec, spec.device_type):
        assert config.output_block_size == 1
        assert config.custom_dataset_path is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

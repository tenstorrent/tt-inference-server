# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""VLM handling in the v2 LLM benchmark-config builder (``get_llm_configs``).

Context: in v1, VLM benchmarks flowed through ``benchmark_config``'s
concurrency sweeps, which fold image ("vision") tokens into ``total_seq_len``
and cap ``max_concurrency`` accordingly (v1 tests
``tests/test_benchmark_concurrency_sweeps.py::test_expand_concurrency_sweeps_image_accounts_for_vision_tokens``
and ``tests/test_benchmark_config.py``). v2 does NOT reproduce that path: the
LLM benchmark runner is text-only and drops every non-``text`` param
(``llm_module/benchmark_configs.py`` — ``params.task_type != "text"``), while
VLM image/text perf is driven separately by the guidellm ``omni_modal_image``
scenario over a real HF dataset (see ``test_guidellm_scenarios.py``).

These tests lock in that v2-side contract: a VLM spec's ``vlm`` (image) and
``structured_output`` params must be excluded from the text sweep so uncapped
image params never leak into the text driver — only the ``text`` params survive.
"""

from __future__ import annotations

import pytest

from benchmarking.benchmark_config import get_benchmark_config
from llm_module.benchmark_configs import get_llm_configs
from workflows.model_spec import MODEL_SPECS
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path

DECODE_ROWS = 32
# slots -> n-gram draft length; absent means speculation off (see the profile notes in dev/llm.yaml)
SPECULATIVE_IMPLS = {"gemma4-31b-qb2": 5}  # impl name -> n-gram draft length; absent means speculation off


def _profiles():
    templates = load_templates_from_yaml(get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml")
    return [
        spec
        for template in templates
        if template.weights == ["google/gemma-4-31B-it"] and template.impl.code_path == "models/demos/gemma4_31b_qb2"
        for spec in template.expand_to_specs()
    ]


def test_both_profiles_serve_32_slots_with_full_context_and_prefix_caching():
    profiles = {spec.impl.impl_name: spec for spec in _profiles()}
    assert sorted(profiles) == ["gemma4-31b-qb2", "gemma4-31b-qb2-c32"]
    for spec in profiles.values():
        device = spec.device_model_spec
        assert device.max_concurrency == 32 and device.max_context == 262144
        assert device.vllm_args["enable-prefix-caching"] is True


def test_the_profiles_differ_only_in_speculation_and_scheduling():
    profiles = {spec.impl.impl_name: spec for spec in _profiles()}
    for name, spec in profiles.items():
        args = spec.device_model_spec.vllm_args
        if name not in SPECULATIVE_IMPLS:
            assert "speculative_config" not in args and args["async-scheduling"] is True
            continue
        config = json.loads(args["speculative_config"])
        assert config["method"] == "ngram" and config["num_speculative_tokens"] == SPECULATIVE_IMPLS[name]
        assert args["async-scheduling"] is False
        assert spec.device_model_spec.env_vars["NUMBA_NUM_THREADS"] == "1"
    speculative, plain = (profiles[n].device_model_spec.vllm_args for n in ("gemma4-31b-qb2", "gemma4-31b-qb2-c32"))
    other = lambda args: {k: v for k, v in args.items() if k not in ("speculative_config", "async-scheduling")}
    assert other(speculative) == other(plain)


def test_each_profile_is_selectable_by_its_impl_name():
    specs = _profiles()
    for impl in ("gemma4-31b-qb2", "gemma4-31b-qb2-c32"):
        spec = resolve_model_spec(specs, model="google/gemma-4-31B-it", device="p300x2", impl=impl, catalog_name="Shield")
        assert spec.impl.impl_name == impl

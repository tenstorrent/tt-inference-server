# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path

DECODE_ROWS = 32
# slots -> n-gram draft length; absent means speculation off (see the profile table in dev/llm.yaml)
SPECULATIVE_K = {1: 5}


def _profiles():
    templates = load_templates_from_yaml(get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml")
    return [
        spec
        for template in templates
        if template.weights == ["google/gemma-4-31B-it"] and template.impl.code_path == "models/demos/gemma4_31b_qb2"
        for spec in template.expand_to_specs()
    ]


def test_every_slot_profile_is_defined_with_prefix_caching():
    profiles = {spec.device_model_spec.max_concurrency: spec for spec in _profiles()}
    assert sorted(profiles) == [1, 4, 8, 16, 32]
    for spec in profiles.values():
        assert spec.device_model_spec.vllm_args["enable-prefix-caching"] is True


def test_speculative_profiles_follow_the_table_and_fit_the_decode_trace():
    for spec in _profiles():
        slots = spec.device_model_spec.max_concurrency
        args = spec.device_model_spec.vllm_args
        if slots not in SPECULATIVE_K:
            assert "speculative_config" not in args and args["async-scheduling"] is True
            continue
        config = json.loads(args["speculative_config"])
        assert config["method"] == "ngram" and config["num_speculative_tokens"] == SPECULATIVE_K[slots]
        assert slots * (1 + config["num_speculative_tokens"]) <= DECODE_ROWS
        assert args["async-scheduling"] is False


def test_each_profile_is_selectable_by_its_impl_name():
    specs = _profiles()
    for impl in ("gemma4-31b-qb2", "gemma4-31b-qb2-c4", "gemma4-31b-qb2-c8", "gemma4-31b-qb2-c16", "gemma4-31b-qb2-c32"):
        spec = resolve_model_spec(specs, model="google/gemma-4-31B-it", device="p300x2", impl=impl, catalog_name="Shield")
        assert spec.impl.impl_name == impl

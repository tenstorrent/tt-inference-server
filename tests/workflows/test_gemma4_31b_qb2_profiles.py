# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path


def _profiles():
    templates = load_templates_from_yaml(get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml")
    return [
        spec
        for template in templates
        if template.weights == ["google/gemma-4-31B-it"] and template.impl.code_path == "models/demos/gemma4_31b_qb2"
        for spec in template.expand_to_specs()
    ]


def test_one_profile_with_32_slots_full_context_and_prefix_caching():
    (spec,) = _profiles()
    device = spec.device_model_spec
    assert device.max_concurrency == 32 and device.max_context == 262144
    assert device.vllm_args["enable-prefix-caching"] is True


def test_speculation_is_ngram_k5_with_synchronous_scheduling_and_one_numba_thread():
    (spec,) = _profiles()
    args = spec.device_model_spec.vllm_args
    config = json.loads(args["speculative_config"])
    assert config["method"] == "ngram" and config["num_speculative_tokens"] == 5
    assert args["async-scheduling"] is False
    assert spec.device_model_spec.env_vars["NUMBA_NUM_THREADS"] == "1"
    assert spec.device_model_spec.env_vars["MESH_DEVICE"] == "P300x2"


def test_profile_is_selectable_by_its_impl_name():
    spec = resolve_model_spec(_profiles(), model="google/gemma-4-31B-it", device="p300x2", impl="gemma4-31b-qb2", catalog_name="Shield")
    assert spec.impl.impl_name == "gemma4-31b-qb2"

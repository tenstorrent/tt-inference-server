# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GLM-5.3-Flash Blackhole Galaxy profiles (dev catalog) agree with the stack adapter's contract.

The adapter (tt-metal models/demos/glm53_flash/tt/generator_vllm_stack.py) builds the device stack with
GLM53_STACK_BUILD_B rows (at least max_num_seqs, default max_num_seqs), every admitted user owns a full context on
device, and the speculative profiles' plugin lookahead must reach the adapter's draft length.
"""

import json

import pytest

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes

GLM53_REPO = "zai-org/GLM-5.3-Flash"
GLM53_IMPLS = (
    "glm53_flash",
    "glm53_flash_mtp",
    "glm53_flash_mtp_mu",
    "glm53_flash_1m",
    "glm53_flash_128k",
    "glm53_flash_128k_mm",
)
# The multi-user MTP steps verify G (K + 1) rows on the B 16 build (tt-metal spec_mtp.py asserts the verify width
# fits the build's rows; generator_vllm_stack.spec_plan_for plans them for GLM53_SPEC_MULTI_W = 16).
SPEC_MULTI_BUILD_ROWS = 16


def _glm53_specs():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [
        spec
        for template in templates
        if template.weights == [GLM53_REPO]
        for spec in template.expand_to_specs()
        if spec.device_type == DeviceTypes.BLACKHOLE_GALAXY
    ]


def _spec(impl_id):
    (spec,) = [s for s in _glm53_specs() if s.impl.impl_id == impl_id]
    return spec


def test_every_glm53_profile_is_listed_once_with_one_default():
    specs = _glm53_specs()
    assert sorted(s.impl.impl_id for s in specs) == sorted(GLM53_IMPLS)
    defaults = [s.impl.impl_id for s in specs if s.device_model_spec.default_impl]
    assert len(defaults) == 1
    resolved = resolve_model_spec(
        specs, model=GLM53_REPO, device="blackhole_galaxy", catalog_name="dev"
    )
    assert resolved.impl.impl_id == defaults[0]


@pytest.mark.parametrize("impl_id", GLM53_IMPLS)
def test_admission_is_resident_and_the_pool_is_one_context_per_user(impl_id):
    dms = _spec(impl_id).device_model_spec
    max_num_seqs = int(dms.vllm_args["max_num_seqs"])
    # every admitted user is resident: the clients' concurrency is the engine's admission
    assert dms.max_concurrency == max_num_seqs
    # the logical KV pool: each admitted user owns a full context (the adapter's get_max_tokens_all_users)
    assert int(dms.vllm_args["max_model_len"]) == dms.max_context
    assert dms.max_tokens_all_users_override == max_num_seqs * dms.max_context
    assert dms.max_tokens_all_users == dms.max_tokens_all_users_override


@pytest.mark.parametrize("impl_id", GLM53_IMPLS)
def test_the_build_holds_every_user_and_every_speculative_step(impl_id):
    dms = _spec(impl_id).device_model_spec
    env = dms.env_vars
    max_num_seqs = int(dms.vllm_args["max_num_seqs"])
    build_b = int(env.get("GLM53_STACK_BUILD_B", max_num_seqs))
    assert build_b >= max_num_seqs

    if "GLM53_SPEC_K" not in env:
        assert "speculative_config" not in dms.vllm_args
        return
    k = int(env["GLM53_SPEC_K"])
    m = int(env.get("GLM53_SPEC_STEPS_PER_CALL", "1"))
    spec_config = json.loads(dms.vllm_args["speculative_config"])
    # the plugin's draft length for M device steps per verify is M (K + 1) - 1 (generator_vllm_stack.spec_plugin_k)
    assert spec_config["num_speculative_tokens"] >= m * (k + 1) - 1

    multi_user = env.get("GLM53_SPEC_MULTI") == "1" or any(
        key in env for key in ("GLM53_SPEC_MULTI_USERS", "GLM53_SPEC_MULTI_ALT")
    )
    if multi_user:
        rows = int(env.get("GLM53_SPEC_MULTI_USERS", "0")) * (k + 1)
        if "GLM53_SPEC_MULTI_ALT" in env:
            alt_users, alt_k = (int(v) for v in env["GLM53_SPEC_MULTI_ALT"].split(":"))
            rows = max(rows, alt_users * (alt_k + 1))
        assert 0 < rows <= SPEC_MULTI_BUILD_ROWS
        assert build_b >= SPEC_MULTI_BUILD_ROWS

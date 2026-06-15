# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

from workflows.helm_generator.cli import assert_single_default_impl, assert_unique
from workflows.helm_generator.errors import GenerateHelmValuesError


def test_unique_specs_pass(vllm_spec, media_spec, forge_spec):
    assert_unique([vllm_spec, media_spec, forge_spec])


def test_duplicate_quadruple_raises(vllm_spec):
    with pytest.raises(GenerateHelmValuesError) as excinfo:
        assert_unique([vllm_spec, vllm_spec])
    assert "Duplicate" in str(excinfo.value)


def test_assert_single_default_impl_passes_for_single_default(vllm_spec):
    assert_single_default_impl([vllm_spec])


def test_assert_single_default_impl_raises_on_collision(sample_impl):
    from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpecTemplate
    from workflows.workflow_types import DeviceTypes, InferenceEngine

    other_impl = ImplSpec(
        impl_id="other_impl",
        impl_name="other",
        repo_url="https://example.com",
        code_path="x",
    )

    def make(impl):
        return ModelSpecTemplate(
            weights=["acme/Collide-7B"],
            impl=impl,
            tt_metal_commit="aaa",
            vllm_commit="bbb",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.GALAXY,
                    max_concurrency=1,
                    max_context=1024,
                    default_impl=True,
                )
            ],
            docker_image="ghcr.io/acme/img:1.0",
        )

    spec_a = make(sample_impl).expand_to_specs()[0]
    spec_b = make(other_impl).expand_to_specs()[0]
    with pytest.raises(GenerateHelmValuesError) as excinfo:
        assert_single_default_impl([spec_a, spec_b])
    assert "2 marked default_impl=True" in str(excinfo.value)

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import re
import pytest

from workflows.model_spec import (
    InferenceEngine,
    ModelSpec,
    ModelSpecTemplate,
    ImplSpec,
    DeviceModelSpec,
    get_model_spec_map,
    MODEL_SPECS,
    spec_templates,
)
from workflows.workflow_types import DeviceTypes, ModelStatusTypes


@pytest.fixture
def sample_impl():
    """Sample implementation spec for testing."""
    return ImplSpec(
        impl_id="test-impl",
        impl_name="test-impl",
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


@pytest.fixture
def sample_device_model_spec():
    """Sample device model spec for testing."""
    return DeviceModelSpec(
        device=DeviceTypes.N150,
        max_concurrency=16,
        max_context=64 * 1024,
        default_impl=True,
    )


class TestModelSpecTemplateSystem:
    """Tests for the ModelSpecTemplate system."""

    def test_template_creation_and_expansion(self, sample_impl):
        """Test template creation and expansion."""
        template = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=16,
                    max_context=64 * 1024,
                    default_impl=True,
                ),
                DeviceModelSpec(
                    device=DeviceTypes.N300,
                    max_concurrency=32,
                    max_context=128 * 1024,
                    default_impl=False,
                ),
            ],
            weights=["test/model-7B", "test/model-7B-Instruct"],
            status="testing",
        )

        # Verify template properties
        assert template.impl.impl_name == "test-impl"
        assert template.tt_metal_commit == "v1.0.0"
        assert len(template.weights) == 2
        assert template.status == "testing"

        # Test template expansion
        specs = template.expand_to_specs()
        assert len(specs) == 4  # 2 weights × 2 devices

        # Verify all specs are ModelSpec instances
        for spec in specs:
            assert isinstance(spec, ModelSpec)
            assert spec.impl == template.impl
            assert spec.status == "testing"

    def test_template_defaults(self, sample_impl):
        """Test template creation with defaults."""
        template = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=32,
                    max_context=128 * 1024,
                )
            ],
            weights=["test/model"],
        )
        assert template.repacked == 0
        assert template.version == "0.4.0"
        assert template.status == ModelStatusTypes.EXPERIMENTAL
        assert template.docker_image is None


class TestModelSpecSystem:
    """Tests for the ModelSpec system."""

    def test_model_spec_creation(self, sample_impl, sample_device_model_spec):
        """Test ModelSpec creation and basic properties."""
        spec = ModelSpec(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-7B",
            model_id="id_test-impl_TestModel-7B_n150",
            model_name="TestModel-7B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_spec=sample_device_model_spec,
        )

        # Verify basic properties
        assert spec.device_type == DeviceTypes.N150
        assert spec.model_name == "TestModel-7B"
        assert spec.param_count == 7  # Inferred from model name
        assert spec.docker_image is not None

    def test_model_spec_validation(self, sample_impl, sample_device_model_spec):
        """Test ModelSpec validation."""
        # Test validation success
        spec = ModelSpec(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/model",
            model_id="test_id",
            model_name="model",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_spec=sample_device_model_spec,
        )
        spec._validate_data()  # Should not raise

        # Test validation failures
        with pytest.raises(AssertionError, match="hf_model_repo must be set"):
            ModelSpec(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                inference_engine=InferenceEngine.VLLM.value,
                device_model_spec=sample_device_model_spec,
            )

        with pytest.raises(
            AssertionError,
            match=re.escape(
                f"inference_engine must be one of {[e.value for e in InferenceEngine]}"
            ),
        ):
            ModelSpec(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="test/model",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                inference_engine="my_custom_inference_engine",
                device_model_spec=sample_device_model_spec,
            )

    def test_parameter_count_inference(self):
        """Test parameter count inference from model names."""
        test_cases = [
            ("meta-llama/Llama-3.1-8B", 8),
            ("meta-llama/Llama-3.1-70B-Instruct", 70),
            ("Qwen/Qwen2.5-72B", 72),
            ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 70),
            ("mistralai/Mistral-7B-Instruct-v0.3", 7),
            ("Qwen/Qwen2.5-3.5B", 3),
            ("test/model-without-param-count", None),
        ]

        for model_repo, expected_count in test_cases:
            result = ModelSpec.infer_param_count(model_repo)
            assert result == expected_count, f"Failed for {model_repo}"

    def test_json_serialization(self, sample_impl, sample_device_model_spec, tmp_path):
        """Test JSON serialization and deserialization."""
        original_spec = ModelSpec(
            device_type=DeviceTypes.T3K,
            impl=sample_impl,
            hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
            model_id="test-llama-70b-t3k",
            model_name="Llama-3.1-70B-Instruct",
            tt_metal_commit="v1.2.3-rc45",
            vllm_commit="abcdef123456",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_spec=sample_device_model_spec,
            status=ModelStatusTypes.FUNCTIONAL,
        )

        # Test JSON export and import
        json_file = original_spec.to_json(run_id="test_run", output_dir=str(tmp_path))
        loaded_spec = ModelSpec.from_json(json_file)

        # Compare key fields
        assert loaded_spec.device_type == original_spec.device_type
        assert loaded_spec.model_id == original_spec.model_id
        assert loaded_spec.model_name == original_spec.model_name
        assert loaded_spec.status == original_spec.status


class TestSystemIntegration:
    """Integration tests for the model spec system."""

    def test_model_spec_map_generation(self, sample_impl):
        """Test spec map generation from templates."""
        templates = [
            ModelSpecTemplate(
                impl=sample_impl,
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                inference_engine=InferenceEngine.VLLM.value,
                device_model_specs=[
                    DeviceModelSpec(
                        device=DeviceTypes.N150,
                        max_concurrency=16,
                        max_context=64 * 1024,
                    ),
                ],
                weights=["test/model-A", "test/model-B"],
            )
        ]

        spec_map = get_model_spec_map(templates)
        assert len(spec_map) == 2  # 2 weights × 1 device

        for model_id, spec in spec_map.items():
            assert isinstance(spec, ModelSpec)
            assert model_id.startswith("id_")
            assert spec.model_id == model_id

    def test_real_spec_templates(self):
        """Test that real spec templates generate valid specs."""
        real_spec_map = get_model_spec_map(spec_templates)
        assert len(real_spec_map) > 20

        for spec in real_spec_map.values():
            assert isinstance(spec, ModelSpec)
            assert isinstance(spec.device_type, DeviceTypes)
            assert spec.hf_model_repo
            assert spec.model_name


class TestModelSpecsStructure:
    """Test MODEL_SPECS structure and compatibility."""

    def test_model_specs_structure(self):
        """Test MODEL_SPECS has proper structure."""
        assert isinstance(MODEL_SPECS, dict)
        assert len(MODEL_SPECS) > 20

        for model_id, spec in MODEL_SPECS.items():
            assert isinstance(model_id, str)
            assert isinstance(spec, ModelSpec)
            assert spec.model_id == model_id
            assert isinstance(spec.device_type, DeviceTypes)

    def test_template_vs_spec_distinction(self):
        """Test that templates and specs have different structures."""
        # Templates should have these attributes
        for template in spec_templates:
            assert hasattr(template, "device_model_specs")
            assert hasattr(template, "weights")

        # Final specs should have different attributes
        for spec in MODEL_SPECS.values():
            assert hasattr(spec, "device_type")
            assert hasattr(spec, "device_model_spec")
            # Should NOT have template attributes
            assert not hasattr(spec, "device_model_specs")
            assert not hasattr(spec, "weights")


if __name__ == "__main__":
    pytest.main([__file__])

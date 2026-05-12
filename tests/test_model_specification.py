#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import re

import pytest

from workflows.model_spec import (
    MODEL_SPECS,
    MODEL_SPECS_SCHEMA_VERSION,
    VERSION,
    DeviceModelSpec,
    ImplSpec,
    ModelSpec,
    ModelSpecTemplate,
    VersionRequirement,
    export_model_specs_json,
    get_model_spec_map,
    spec_templates,
    SystemRequirements,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    VersionMode,
)


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
        tensor_cache_timeout=2400.0,
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
                    tensor_cache_timeout=1800.0,
                ),
                DeviceModelSpec(
                    device=DeviceTypes.N300,
                    max_concurrency=32,
                    max_context=128 * 1024,
                    default_impl=False,
                    tensor_cache_timeout=3600.0,
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
            expected_timeout = (
                1800.0 if spec.device_type == DeviceTypes.N150 else 3600.0
            )
            assert spec.device_model_spec.tensor_cache_timeout == expected_timeout

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
        assert template.version == VERSION
        assert template.status == ModelStatusTypes.EXPERIMENTAL
        assert template.docker_image is None

    def test_system_requirement_template_level(self, sample_impl):
        """Test that SystemRequirements propagate correctly from templates and device specs."""
        template1 = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            system_requirements=SystemRequirements(
                firmware=VersionRequirement(
                    specifier=">=18.6.0",
                    mode=VersionMode.STRICT,
                ),
                kmd=VersionRequirement(
                    specifier=">=2.1.0",
                    mode=VersionMode.STRICT,
                ),
            ),
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=32,
                    max_context=128 * 1024,
                )
            ],
            weights=["test/model-1"],
        )

        specs1 = template1.expand_to_specs()
        assert len(specs1) == 1
        assert specs1[0].system_requirements is not None
        assert specs1[0].system_requirements.firmware.specifier == ">=18.6.0"
        assert specs1[0].system_requirements.firmware.mode == VersionMode.STRICT
        assert specs1[0].system_requirements.kmd.specifier == ">=2.1.0"
        assert specs1[0].system_requirements.kmd.mode == VersionMode.STRICT

    def test_system_requirements_device_model_spec_level(self, sample_impl):
        template2 = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.1.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=32,
                    max_context=128 * 1024,
                    system_requirements=SystemRequirements(
                        firmware=VersionRequirement(
                            specifier=">=18.8.0",
                            mode=VersionMode.SUGGESTED,
                        ),
                        kmd=VersionRequirement(
                            specifier=">=2.2.0",
                            mode=VersionMode.SUGGESTED,
                        ),
                    ),
                ),
            ],
            weights=["test/model-2"],
        )

        specs2 = template2.expand_to_specs()
        assert len(specs2) == 1
        assert specs2[0].system_requirements is not None
        assert specs2[0].system_requirements.firmware.specifier == ">=18.8.0"
        assert specs2[0].system_requirements.firmware.mode == VersionMode.SUGGESTED
        assert specs2[0].system_requirements.kmd.specifier == ">=2.2.0"
        assert specs2[0].system_requirements.kmd.mode == VersionMode.SUGGESTED

    def test_system_requirements_both_levels(self, sample_impl):
        template3 = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.2.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            system_requirements=SystemRequirements(
                firmware=VersionRequirement(
                    specifier=">=18.0.0",
                    mode=VersionMode.STRICT,
                ),
                kmd=VersionRequirement(
                    specifier=">=2.0.0",
                    mode=VersionMode.STRICT,
                ),
            ),
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=32,
                    max_context=128 * 1024,
                    system_requirements=SystemRequirements(
                        firmware=VersionRequirement(
                            specifier=">=18.12.0",
                            mode=VersionMode.SUGGESTED,
                        ),
                        kmd=VersionRequirement(
                            specifier=">=2.4.1",
                            mode=VersionMode.SUGGESTED,
                        ),
                    ),
                ),
            ],
            weights=["test/model-3"],
        )

        specs3 = template3.expand_to_specs()
        assert len(specs3) == 1
        assert specs3[0].system_requirements is not None
        assert specs3[0].system_requirements.firmware.specifier == ">=18.12.0"
        assert specs3[0].system_requirements.firmware.mode == VersionMode.SUGGESTED
        assert specs3[0].system_requirements.kmd.specifier == ">=2.4.1"
        assert specs3[0].system_requirements.kmd.mode == VersionMode.SUGGESTED

    def test_metadata_per_weight(self, sample_impl):
        """Test that metadata is correctly assigned per weight during expansion."""
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
            ],
            weights=["test/model-A", "test/model-B"],
            metadata={
                "test/model-A": {"license": "apache-2.0", "custom_key": 42},
                "test/model-B": {"license": "mit"},
            },
        )

        specs = template.expand_to_specs()
        assert len(specs) == 2

        spec_a = [s for s in specs if s.hf_model_repo == "test/model-A"][0]
        spec_b = [s for s in specs if s.hf_model_repo == "test/model-B"][0]

        assert spec_a.metadata == {"license": "apache-2.0", "custom_key": 42}
        assert spec_b.metadata == {"license": "mit"}

    def test_metadata_defaults_empty(self, sample_impl):
        """Test that metadata defaults to empty dict when not provided."""
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
                ),
            ],
            weights=["test/model"],
        )

        assert template.metadata == {}
        specs = template.expand_to_specs()
        assert specs[0].metadata == {}

    def test_metadata_partial_weights(self, sample_impl):
        """Test that weights without metadata entries get empty dict."""
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
                ),
            ],
            weights=["test/model-A", "test/model-B"],
            metadata={
                "test/model-A": {"some": "data"},
            },
        )

        specs = template.expand_to_specs()
        spec_a = [s for s in specs if s.hf_model_repo == "test/model-A"][0]
        spec_b = [s for s in specs if s.hf_model_repo == "test/model-B"][0]

        assert spec_a.metadata == {"some": "data"}
        assert spec_b.metadata == {}

    def test_metadata_invalid_key_raises(self, sample_impl):
        """Test that metadata with keys not in weights raises an error."""
        with pytest.raises(AssertionError, match="These keys do not exist as weights"):
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
                weights=["test/model-A"],
                metadata={
                    "test/model-A": {"ok": True},
                    "test/nonexistent-model": {"bad": True},
                },
            )


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
        assert (
            loaded_spec.device_model_spec.tensor_cache_timeout
            == original_spec.device_model_spec.tensor_cache_timeout
        )

    def test_apply_overrides_commits_from_docker_image(
        self, sample_impl, sample_device_model_spec
    ):
        """Test that apply_overrides updates commits from docker image tag."""
        from workflows.runtime_config import RuntimeConfig

        default_tt_metal_commit = "default-tt-metal-commit-1234567890"
        default_vllm_commit = "default-vllm"
        spec = ModelSpec(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-7B",
            model_id="id_test-impl_TestModel-7B_n150",
            model_name="TestModel-7B",
            tt_metal_commit=default_tt_metal_commit,
            vllm_commit=default_vllm_commit,
            inference_engine=InferenceEngine.VLLM.value,
            device_model_spec=sample_device_model_spec,
        )

        assert spec.tt_metal_commit == default_tt_metal_commit
        assert spec.vllm_commit == default_vllm_commit

        new_tt_metal_commit = "fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd"
        new_vllm_commit = "7a9b86f"
        docker_image_with_commits = f"ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.4.0-{new_tt_metal_commit}-{new_vllm_commit}-58111263717"
        rc = RuntimeConfig(
            model="TestModel-7B",
            workflow="benchmarks",
            device="n150",
            override_docker_image=docker_image_with_commits,
        )

        spec.apply_overrides(rc)

        assert spec.tt_metal_commit == new_tt_metal_commit
        assert spec.vllm_commit == new_vllm_commit
        assert spec.docker_image == docker_image_with_commits


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

    def test_export_model_specs_json_includes_metadata(
        self, sample_impl, sample_device_model_spec, tmp_path
    ):
        """Test exported model spec JSON includes top-level metadata."""
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
        output_path = tmp_path / "model_spec.json"

        num_specs = export_model_specs_json({spec.model_id: spec}, output_path)

        assert num_specs == 1
        data = json.loads(output_path.read_text())
        assert data["schema_version"] == MODEL_SPECS_SCHEMA_VERSION
        assert data["release_version"] == VERSION
        assert (
            data["model_specs"][spec.hf_model_repo][spec.device_type.to_string()][
                spec.inference_engine
            ][spec.impl.impl_id]["model_id"]
            == spec.model_id
        )
        assert (
            data["model_specs"][spec.hf_model_repo][spec.device_type.to_string()][
                spec.inference_engine
            ][spec.impl.impl_id]["device_model_spec"]["tensor_cache_timeout"]
            == spec.device_model_spec.tensor_cache_timeout
        )

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

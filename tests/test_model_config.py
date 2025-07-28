#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import patch

from workflows.model_config import (
    ModelConfig,
    ModelConfigTemplate,
    ImplConfig,
    DeviceModelSpec,
    get_model_config_map,
    MODEL_CONFIGS,
    config_templates,
)
from workflows.workflow_types import DeviceTypes, ModelStatusTypes


@pytest.fixture
def sample_impl():
    """Sample implementation config for testing."""
    return ImplConfig(
        impl_id="test-impl",
        impl_name="test-impl",
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


@pytest.fixture
def sample_device_model_spec():
    """Sample device model spec for testing."""
    return DeviceModelSpec(
        max_concurrency=16,
        max_context=64 * 1024,
        default_impl=True,
    )


class TestModelConfigTemplateSystem:
    """Comprehensive tests for the ModelConfigTemplate system."""

    def test_template_creation_and_expansion(self, sample_impl):
        """Test template creation, defaults, and expansion in one comprehensive test."""
        # Test basic template creation
        template = ModelConfigTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            device_model_spec_map={
                DeviceTypes.N150: DeviceModelSpec(
                    max_concurrency=16,
                    max_context=64 * 1024,
                    default_impl=True,
                    override_tt_config={"test_param": "test_value"},
                ),
                DeviceTypes.N300: DeviceModelSpec(
                    max_concurrency=32,
                    max_context=128 * 1024,
                    default_impl=False,
                ),
            },
            weights=["test/model-7B", "test/model-7B-Instruct"],
            status="testing",
        )

        # Verify template properties
        assert template.impl.impl_name == "test-impl"
        assert template.tt_metal_commit == "v1.0.0"
        assert len(template.weights) == 2
        assert DeviceTypes.N150 in template.device_model_spec_map
        assert template.status == "testing"

        # Test template with defaults
        minimal_template = ModelConfigTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            device_model_spec_map={
                DeviceTypes.N150: DeviceModelSpec(
                    max_concurrency=32,
                    max_context=128 * 1024,
                )
            },
            weights=["test/model"],
        )
        assert minimal_template.repacked == 0
        assert minimal_template.version == "0.0.1"
        assert minimal_template.status == ModelStatusTypes.EXPERIMENTAL
        assert minimal_template.docker_image is None

        # Test template expansion
        configs = template.expand_to_configs()

        # Should create configs for 2 weights × 2 devices = 4 configs
        assert len(configs) == 4

        # Verify all configs are ModelConfig instances with correct properties
        for config in configs:
            assert isinstance(config, ModelConfig)
            assert config.impl == template.impl
            assert config.tt_metal_commit == template.tt_metal_commit
            assert config.vllm_commit == template.vllm_commit
            assert config.status == "testing"

        # Test device-specific values are correctly extracted
        n150_configs = [c for c in configs if c.device_type == DeviceTypes.N150]
        n300_configs = [c for c in configs if c.device_type == DeviceTypes.N300]

        assert len(n150_configs) == 2
        assert len(n300_configs) == 2

        for config in n150_configs:
            assert config.device_model_spec.max_concurrency == 16
            assert config.device_model_spec.max_context == 64 * 1024
            assert config.override_tt_config == {"test_param": "test_value"}

        for config in n300_configs:
            assert config.device_model_spec.max_concurrency == 32
            assert config.device_model_spec.max_context == 128 * 1024
            assert config.override_tt_config == {}

        # Test model naming from weight paths
        model_names = {config.model_name for config in configs}
        assert "model-7B" in model_names
        assert "model-7B-Instruct" in model_names

        # Test immutability
        with pytest.raises(AttributeError):
            template.impl = None


class TestModelConfigSystem:
    """Comprehensive tests for the ModelConfig system."""

    def test_model_config_creation_validation_and_inference(
        self, sample_impl, sample_device_model_spec
    ):
        """Test ModelConfig creation, validation, and data inference comprehensively."""
        # Test basic creation with minimal fields
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-7B",
            model_id="id_test-impl_TestModel-7B_n150",
            model_name="TestModel-7B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            device_model_spec=sample_device_model_spec,
        )

        # Verify basic properties
        assert config.device_type == DeviceTypes.N150
        assert config.impl == sample_impl
        assert config.hf_model_repo == "test/TestModel-7B"
        assert config.model_name == "TestModel-7B"

        # Test data inference - param count, disk/RAM, defaults
        assert config.param_count == 7  # Inferred from model name
        assert config.min_disk_gb == 7 * 4  # 4x for non-repacked
        assert config.min_ram_gb == 7 * 5  # 5x conservative estimate
        assert config.device_model_spec.max_concurrency == 16
        assert config.device_model_spec.max_context == 64 * 1024
        assert config.docker_image is not None
        assert config.code_link is not None

        # Test repacked model disk calculation
        repacked_config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-70B",
            model_id="test_id",
            model_name="TestModel-70B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            device_model_spec=sample_device_model_spec,
            repacked=1,
        )
        assert repacked_config.min_disk_gb == 70 * 5  # 5x for repacked

        # Test validation success
        config.validate_data()  # Should not raise

        # Test validation failures
        with pytest.raises(AssertionError, match="hf_model_repo must be set"):
            ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                device_model_spec=sample_device_model_spec,
            )

        with pytest.raises(AssertionError, match="model_name must be set"):
            ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="test/model",
                model_id="test_id",
                model_name="",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                device_model_spec=sample_device_model_spec,
            )

        # Test parameter count inference for various model names
        test_cases = [
            ("meta-llama/Llama-3.1-8B", 8),
            ("meta-llama/Llama-3.1-70B-Instruct", 70),
            ("Qwen/Qwen2.5-72B", 72),
            ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 70),
            ("mistralai/Mistral-7B-Instruct-v0.3", 7),
            ("Qwen/Qwen2.5-3.5B", 3),  # Decimal handling
            ("test/model-without-param-count", None),
        ]

        for model_repo, expected_count in test_cases:
            result = ModelConfig.infer_param_count(model_repo)
            assert result == expected_count, f"Failed for {model_repo}"

        # Test immutability
        with pytest.raises(AttributeError):
            config.device_type = DeviceTypes.N300

        # Test docker image generation
        with patch("workflows.model_config.VERSION", "1.0.0"):
            docker_config = ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="test/model",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.2.3-rc45",
                vllm_commit="abcdef123456",
                device_model_spec=sample_device_model_spec,
            )

            expected_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
            expected_tag = "1.0.0-v1.2.3-rc45-abcdef123456"
            expected_image = f"{expected_repo}:{expected_tag}"
            assert docker_config.docker_image == expected_image


class TestSystemIntegration:
    """Integration tests for the complete model config system."""

    def test_end_to_end_template_to_final_configs(self, sample_impl):
        """Test the complete flow from templates to final configurations."""
        # Create test templates
        templates = [
            ModelConfigTemplate(
                impl=sample_impl,
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                device_model_spec_map={
                    DeviceTypes.N150: DeviceModelSpec(
                        max_concurrency=16,
                        max_context=64 * 1024,
                    ),
                    DeviceTypes.N300: DeviceModelSpec(
                        max_concurrency=32,
                        max_context=128 * 1024,
                    ),
                },
                weights=["test/model-A", "test/model-B"],
            )
        ]

        # Test config map generation
        config_map = get_model_config_map(templates)

        # Should create configs for 2 weights × 2 devices = 4
        assert len(config_map) == 4

        # All values should be ModelConfig instances with proper model IDs
        for model_id, config in config_map.items():
            assert isinstance(config, ModelConfig)
            assert model_id.startswith("id_")
            assert config.model_id == model_id
            assert config.hf_model_repo in ["test/model-A", "test/model-B"]
            assert config.model_name in ["model-A", "model-B"]

        # Test with real config templates
        real_config_map = get_model_config_map(config_templates)
        assert len(real_config_map) > 20  # Adjusted expectation

        # Verify all real configs have proper structure
        for config in real_config_map.values():
            assert isinstance(config, ModelConfig)
            assert isinstance(config.device_type, DeviceTypes)
            assert config.hf_model_repo
            assert config.model_name
            assert config.model_id
            assert hasattr(config, "device_model_spec")
            assert hasattr(config.device_model_spec, "max_concurrency")
            assert hasattr(config.device_model_spec, "max_context")
            # Should not have old template attributes
            assert not hasattr(config, "device_configurations")
            assert not hasattr(config, "weights")


class TestBackwardCompatibilityAndStructure:
    """Test backward compatibility and proper system structure."""

    def test_model_configs_structure_and_compatibility(self):
        """Test MODEL_CONFIGS structure, backward compatibility, and template vs config distinction."""
        # Test MODEL_CONFIGS has expected structure and size
        assert isinstance(MODEL_CONFIGS, dict)
        assert len(MODEL_CONFIGS) > 20  # Adjusted expectation

        # Test all entries have proper structure
        for model_id, config in MODEL_CONFIGS.items():
            assert isinstance(model_id, str)
            assert isinstance(config, ModelConfig)
            assert config.model_id == model_id
            assert isinstance(config.device_type, DeviceTypes)
            assert config.device_model_spec.max_concurrency is not None
            assert config.device_model_spec.max_context is not None

        # Test GPU configs exist for reference testing
        gpu_configs = [
            config
            for config in MODEL_CONFIGS.values()
            if config.device_type == DeviceTypes.GPU
        ]
        assert len(gpu_configs) > 0  # At least some GPU configs should exist

        # Test template vs final config structure distinction
        for template in config_templates:
            # Templates should have these attributes
            assert hasattr(template, "device_model_spec_map")
            assert isinstance(template.device_model_spec_map, dict)
            assert hasattr(template, "weights")
            assert isinstance(template.weights, list)

        for config in MODEL_CONFIGS.values():
            # Final configs should have these attributes
            assert hasattr(config, "device_type")
            assert isinstance(config.device_type, DeviceTypes)
            assert hasattr(config, "device_model_spec")
            assert isinstance(config.device_model_spec, DeviceModelSpec)
            # Should NOT have template attributes
            assert not hasattr(config, "device_model_spec_map")
            assert not hasattr(config, "weights")

        # Test specific model configuration exists and works
        mistral_configs = [
            config
            for config in MODEL_CONFIGS.values()
            if "Mistral" in config.model_name
        ]
        assert len(mistral_configs) > 0

        mistral_config = mistral_configs[0]
        assert mistral_config.device_model_spec.max_concurrency is not None
        assert mistral_config.impl.impl_name == "tt-transformers"


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_system_performance_and_edge_cases(self, sample_impl):
        """Test performance, immutability, and edge cases."""
        import time

        # Test performance - config expansion should be fast
        start_time = time.time()
        config_map = get_model_config_map(config_templates)
        end_time = time.time()

        assert end_time - start_time < 1.0  # Should complete quickly
        assert 20 < len(config_map) < 200  # Reasonable number of configs

        # Test ImplConfig immutability and creation
        impl = ImplConfig(
            impl_id="test-id",
            impl_name="test-name",
            repo_url="https://example.com",
            code_path="models/test",
        )
        assert impl.impl_id == "test-id"
        with pytest.raises(AttributeError):
            impl.impl_id = "modified"

        # Test edge cases in parameter inference
        edge_cases = [
            ("no-numbers", None),
            ("multiple-123B-456B", 456),  # Should take last match
            ("decimal-3.14B", 3),
            ("", None),
        ]

        for model_repo, expected in edge_cases:
            result = ModelConfig.infer_param_count(model_repo)
            assert result == expected, f"Failed for edge case: {model_repo}"

        # Test DeviceModelSpec functionality
        device_spec = DeviceModelSpec(
            max_concurrency=16,
            max_context=64 * 1024,
            default_impl=True,
        )
        assert device_spec.max_concurrency == 16
        assert device_spec.max_context == 64 * 1024
        assert device_spec.default_impl is True

        # Test DeviceModelSpec with defaults
        minimal_spec = DeviceModelSpec(
            max_concurrency=0,  # Will be set to default
            max_context=0,  # Will be set to default
        )
        assert minimal_spec.max_concurrency == 32  # Default value
        assert minimal_spec.max_context == 128 * 1024  # Default value


if __name__ == "__main__":
    pytest.main([__file__])

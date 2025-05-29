#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from pathlib import Path
from unittest.mock import patch

from workflows.model_config import (
    ModelConfig,
    ModelConfigTemplate,
    ImplConfig,
    get_model_config_map,
    MODEL_CONFIGS,
    config_templates,
)
from workflows.workflow_types import DeviceTypes


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
def sample_template(sample_impl):
    """Sample model config template for testing."""
    return ModelConfigTemplate(
        impl=sample_impl,
        tt_metal_commit="v1.0.0",
        vllm_commit="abc123",
        device_configurations={DeviceTypes.N150, DeviceTypes.N300},
        weights=["test/model-7B", "test/model-7B-Instruct"],
        default_impl_map={
            DeviceTypes.N150: True,
            DeviceTypes.N300: False,
        },
        status="testing",
        max_concurrency_map={
            DeviceTypes.N150: 16,
            DeviceTypes.N300: 32,
        },
        max_context_map={
            DeviceTypes.N150: 64 * 1024,
            DeviceTypes.N300: 128 * 1024,
        },
        override_tt_config={
            "test_param": "test_value",
        },
    )


class TestImplConfig:
    """Tests for ImplConfig class."""

    def test_impl_config_creation(self):
        """Test ImplConfig creation with all required fields."""
        impl = ImplConfig(
            impl_id="test-id",
            impl_name="test-name", 
            repo_url="https://example.com",
            code_path="models/test"
        )
        assert impl.impl_id == "test-id"
        assert impl.impl_name == "test-name"
        assert impl.repo_url == "https://example.com"
        assert impl.code_path == "models/test"

    def test_impl_config_frozen(self):
        """Test that ImplConfig is frozen (immutable)."""
        impl = ImplConfig(
            impl_id="test", impl_name="test", repo_url="test", code_path="test"
        )
        with pytest.raises(AttributeError):
            impl.impl_id = "modified"


class TestModelConfigTemplate:
    """Tests for ModelConfigTemplate class."""

    def test_template_creation(self, sample_template):
        """Test basic template creation."""
        assert sample_template.impl.impl_name == "test-impl"
        assert sample_template.tt_metal_commit == "v1.0.0"
        assert sample_template.vllm_commit == "abc123"
        assert len(sample_template.weights) == 2
        assert DeviceTypes.N150 in sample_template.device_configurations
        assert DeviceTypes.N300 in sample_template.device_configurations

    def test_template_defaults(self, sample_impl):
        """Test template creation with default values."""
        template = ModelConfigTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            device_configurations={DeviceTypes.N150},
            weights=["test/model"],
        )
        assert template.repacked == 0
        assert template.version == "0.0.1"
        assert template.status == "preview"
        assert template.docker_image is None
        assert len(template.perf_targets_map) == 0
        assert len(template.max_concurrency_map) == 0
        assert len(template.max_context_map) == 0
        assert len(template.override_tt_config) == 0

    def test_expand_to_configs_basic(self, sample_template):
        """Test basic template expansion."""
        configs = sample_template.expand_to_configs()
        
        # Should create configs for 2 weights × 2 devices + GPU = 6 configs
        # (N150, N300, GPU) × 2 weights = 6
        assert len(configs) == 6
        
        # Check that all configs are ModelConfig instances
        for config in configs:
            assert isinstance(config, ModelConfig)
            assert config.impl == sample_template.impl
            assert config.tt_metal_commit == sample_template.tt_metal_commit
            assert config.vllm_commit == sample_template.vllm_commit

    def test_expand_to_configs_adds_gpu(self, sample_template):
        """Test that GPU device is automatically added for reference testing."""
        configs = sample_template.expand_to_configs()
        
        # Check that GPU configs are created
        gpu_configs = [c for c in configs if c.device_type == DeviceTypes.GPU]
        assert len(gpu_configs) == 2  # One for each weight
        
        for gpu_config in gpu_configs:
            assert gpu_config.device_type == DeviceTypes.GPU
            assert gpu_config.model_name in ["model-7B", "model-7B-Instruct"]

    def test_expand_to_configs_device_specific_values(self, sample_template):
        """Test that device-specific values are correctly extracted."""
        configs = sample_template.expand_to_configs()
        
        # Find N150 and N300 configs
        n150_configs = [c for c in configs if c.device_type == DeviceTypes.N150]
        n300_configs = [c for c in configs if c.device_type == DeviceTypes.N300]
        
        assert len(n150_configs) == 2
        assert len(n300_configs) == 2
        
        # Check device-specific values
        for config in n150_configs:
            assert config.max_concurrency == 16
            assert config.max_context == 64 * 1024
            
        for config in n300_configs:
            assert config.max_concurrency == 32
            assert config.max_context == 128 * 1024

    def test_expand_to_configs_model_naming(self, sample_template):
        """Test that model names are correctly extracted from weight paths."""
        configs = sample_template.expand_to_configs()
        
        model_names = {config.model_name for config in configs}
        assert "model-7B" in model_names
        assert "model-7B-Instruct" in model_names

    def test_template_frozen(self, sample_template):
        """Test that ModelConfigTemplate is frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_template.impl = None


class TestModelConfig:
    """Tests for ModelConfig class."""

    def test_model_config_creation_minimal(self, sample_impl):
        """Test ModelConfig creation with minimal required fields."""
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/model-7B",
            model_id="id_test-impl_model-7B_n150",
            model_name="model-7B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
        )
        
        assert config.device_type == DeviceTypes.N150
        assert config.impl == sample_impl
        assert config.hf_model_repo == "test/model-7B"
        assert config.model_name == "model-7B"
        assert config.tt_metal_commit == "v1.0.0"
        assert config.vllm_commit == "abc123"

    def test_model_config_data_inference(self, sample_impl):
        """Test that ModelConfig correctly infers missing data."""
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-7B",
            model_id="test_id",
            model_name="TestModel-7B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
        )
        
        # Should infer param count from model name
        assert config.param_count == 7
        
        # Should infer disk and RAM requirements
        assert config.min_disk_gb == 7 * 4  # 4x for non-repacked
        assert config.min_ram_gb == 7 * 5   # 5x conservative estimate
        
        # Should set defaults
        assert config.max_concurrency == 32
        assert config.max_context == 128 * 1024
        assert config.docker_image is not None
        assert config.code_link is not None

    def test_model_config_repacked_disk_calculation(self, sample_impl):
        """Test disk calculation for repacked models."""
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/TestModel-70B",
            model_id="test_id",
            model_name="TestModel-70B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            repacked=1,
        )
        
        # Should use 5x multiplier for repacked models
        assert config.min_disk_gb == 70 * 5

    def test_model_config_validation_success(self, sample_impl):
        """Test successful validation."""
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/model",
            model_id="test_id",
            model_name="model",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
        )
        # Should not raise any exceptions
        config.validate_data()

    def test_model_config_validation_failures(self, sample_impl):
        """Test validation failures."""
        # Missing hf_model_repo
        with pytest.raises(AssertionError, match="hf_model_repo must be set"):
            ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
            )
        
        # Missing model_name
        with pytest.raises(AssertionError, match="model_name must be set"):
            ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="test/model",
                model_id="test_id",
                model_name="",
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
            )

    @pytest.mark.parametrize(
        "model_repo,expected_param_count",
        [
            ("meta-llama/Llama-3.1-8B", 8),
            ("meta-llama/Llama-3.1-70B-Instruct", 70),
            ("Qwen/Qwen2.5-72B", 72),
            ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 70),
            ("mistralai/Mistral-7B-Instruct-v0.3", 7),
            ("Qwen/Qwen2.5-7B-Instruct", 7),
            ("test/model-without-param-count", None),
            ("Qwen/Qwen2.5-3.5B", 3),  # Test decimal handling
        ],
    )
    def test_infer_param_count(self, model_repo, expected_param_count):
        """Test parameter count inference from model repository names."""
        result = ModelConfig.infer_param_count(model_repo)
        assert result == expected_param_count

    def test_model_config_frozen(self, sample_impl):
        """Test that ModelConfig is frozen (immutable) after creation."""
        config = ModelConfig(
            device_type=DeviceTypes.N150,
            impl=sample_impl,
            hf_model_repo="test/model",
            model_id="test_id",
            model_name="model",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
        )
        
        with pytest.raises(AttributeError):
            config.device_type = DeviceTypes.N300

    def test_docker_image_generation(self, sample_impl):
        """Test docker image URL generation."""
        with patch("workflows.model_config.VERSION", "1.0.0"):
            config = ModelConfig(
                device_type=DeviceTypes.N150,
                impl=sample_impl,
                hf_model_repo="test/model",
                model_id="test_id",
                model_name="model",
                tt_metal_commit="v1.2.3-rc45",
                vllm_commit="abcdef123456",
            )
            
            expected_repo = "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64"
            expected_tag = "1.0.0-v1.2.3-rc45-abcdef123456"
            expected_image = f"{expected_repo}:{expected_tag}"
            
            assert config.docker_image == expected_image


class TestModelConfigIntegration:
    """Integration tests for the model config system."""

    def test_get_model_config_map(self, sample_impl):
        """Test the model config map generation function."""
        templates = [
            ModelConfigTemplate(
                impl=sample_impl,
                tt_metal_commit="v1.0.0",
                vllm_commit="abc123",
                device_configurations={DeviceTypes.N150, DeviceTypes.N300},
                weights=["test/model-A", "test/model-B"],
            )
        ]
        
        config_map = get_model_config_map(templates)
        
        # Should create configs for each weight-device combination + GPU
        # 2 weights × 3 devices (N150, N300, GPU) = 6 configs
        assert len(config_map) == 6
        
        # All values should be ModelConfig instances
        for config in config_map.values():
            assert isinstance(config, ModelConfig)
        
        # Keys should be model IDs
        for model_id in config_map.keys():
            assert model_id.startswith("id_")

    def test_actual_config_templates_expansion(self):
        """Test that actual config templates expand correctly."""
        # Test with the real config templates
        config_map = get_model_config_map(config_templates)
        
        assert len(config_map) > 0
        
        # All configs should be ModelConfig instances
        for config in config_map.values():
            assert isinstance(config, ModelConfig)
            assert hasattr(config, 'device_type')
            assert isinstance(config.device_type, DeviceTypes)
            assert config.hf_model_repo
            assert config.model_name
            assert config.model_id

    def test_model_configs_consistency(self):
        """Test that MODEL_CONFIGS is consistent with expectations."""
        # Should have configs for multiple models and devices
        assert len(MODEL_CONFIGS) > 50  # We expect many configurations
        
        # Should have GPU configs for reference testing
        gpu_configs = [
            config for config in MODEL_CONFIGS.values() 
            if config.device_type == DeviceTypes.GPU
        ]
        assert len(gpu_configs) > 0
        
        # All configs should have single device types
        for config in MODEL_CONFIGS.values():
            assert isinstance(config.device_type, DeviceTypes)
            assert hasattr(config, 'max_concurrency')
            assert hasattr(config, 'max_context')
            # Should not have the old device_configurations attribute
            assert not hasattr(config, 'device_configurations')

    def test_template_vs_final_config_structure(self):
        """Test that templates and final configs have correct distinct structures."""
        # Templates should have device_configurations (set) and maps
        for template in config_templates:
            assert hasattr(template, 'device_configurations')
            assert isinstance(template.device_configurations, set)
            assert hasattr(template, 'max_concurrency_map')
            assert hasattr(template, 'max_context_map')
            assert hasattr(template, 'weights')
            
        # Final configs should have device_type (single) and single values
        for config in MODEL_CONFIGS.values():
            assert hasattr(config, 'device_type')
            assert isinstance(config.device_type, DeviceTypes)
            assert hasattr(config, 'max_concurrency')
            assert hasattr(config, 'max_context')
            assert not hasattr(config, 'device_configurations')
            assert not hasattr(config, 'max_concurrency_map')
            assert not hasattr(config, 'max_context_map')
            assert not hasattr(config, 'weights')

    def test_specific_model_configurations(self):
        """Test specific model configurations we know should exist."""
        # Find a Mistral config for N150
        mistral_n150_configs = [
            config for config in MODEL_CONFIGS.values()
            if "Mistral" in config.model_name and config.device_type == DeviceTypes.N150
        ]
        assert len(mistral_n150_configs) > 0
        
        mistral_config = mistral_n150_configs[0]
        assert mistral_config.device_type == DeviceTypes.N150
        assert mistral_config.max_concurrency is not None
        assert mistral_config.max_context is not None
        assert mistral_config.impl.impl_name == "tt-transformers"

    def test_backward_compatibility_interface(self):
        """Test that the MODEL_CONFIGS interface is backward compatible."""
        # The MODEL_CONFIGS should still be a dict mapping model_id to ModelConfig
        assert isinstance(MODEL_CONFIGS, dict)
        
        for model_id, config in MODEL_CONFIGS.items():
            assert isinstance(model_id, str)
            assert isinstance(config, ModelConfig)
            assert config.model_id == model_id


class TestModelConfigPerformance:
    """Performance-related tests."""

    def test_config_expansion_performance(self):
        """Test that config expansion doesn't create excessive objects."""
        import time
        
        start_time = time.time()
        config_map = get_model_config_map(config_templates)
        end_time = time.time()
        
        # Should complete quickly (less than 1 second)
        assert end_time - start_time < 1.0
        
        # Should create reasonable number of configs
        assert 50 < len(config_map) < 200


if __name__ == "__main__":
    pytest.main([__file__])

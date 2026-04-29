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
    KnownIssue,
    ModelSpec,
    ModelSpecTemplate,
    VersionRequirement,
    export_model_specs_json,
    get_model_spec_map,
    spec_templates,
    SystemRequirements,
)
from workflows.run_reports import enforce_acceptance_criteria, _is_check_failing
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ReportCheckTypes,
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

    def test_expand_to_specs_propagates_known_issues(self, sample_impl):
        """Test that known_issues on DeviceModelSpec survive template expansion."""
        known_issues = [
            KnownIssue(
                workflow_type="BENCHMARKS",
                reason="GH#2600 - OOM on N150",
                task_name="isl-128_osl-1024_con-32",
            ),
            KnownIssue(
                workflow_type="EVALS",
                reason="GH#2550 - eval harness crash",
            ),
        ]
        template = ModelSpecTemplate(
            impl=sample_impl,
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.N150,
                    max_concurrency=16,
                    max_context=8192,
                    known_issues=known_issues,
                ),
            ],
            weights=["test/model-7B"],
        )
        specs = template.expand_to_specs()
        assert len(specs) == 1
        expanded_issues = specs[0].device_model_spec.known_issues
        assert len(expanded_issues) == 2
        assert expanded_issues[0].workflow_type == "BENCHMARKS"
        assert expanded_issues[0].task_name == "isl-128_osl-1024_con-32"
        assert expanded_issues[1].workflow_type == "EVALS"
        assert expanded_issues[1].task_name is None

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


class TestRequiredTargetTiers:
    """Tests for ModelStatusTypes.required_target_tiers property."""

    def test_experimental_has_no_required_tiers(self):
        assert ModelStatusTypes.EXPERIMENTAL.required_target_tiers == []

    def test_functional_requires_functional(self):
        assert ModelStatusTypes.FUNCTIONAL.required_target_tiers == ["functional"]

    def test_complete_requires_functional_and_complete(self):
        assert ModelStatusTypes.COMPLETE.required_target_tiers == [
            "functional",
            "complete",
        ]

    def test_top_perf_requires_all_tiers(self):
        assert ModelStatusTypes.TOP_PERF.required_target_tiers == [
            "functional",
            "complete",
            "target",
        ]


class TestEnforceAcceptanceCriteria:
    """Tests for the enforce_acceptance_criteria function."""

    @pytest.fixture
    def passing_target_checks(self):
        return {
            "functional": {
                "ttft_check": ReportCheckTypes.PASS,
                "tput_check": ReportCheckTypes.PASS,
            },
            "complete": {
                "ttft_check": ReportCheckTypes.PASS,
                "tput_check": ReportCheckTypes.PASS,
            },
            "target": {
                "ttft_check": ReportCheckTypes.PASS,
                "tput_check": ReportCheckTypes.PASS,
            },
        }

    @pytest.fixture
    def failing_target_checks(self):
        return {
            "functional": {
                "ttft_check": ReportCheckTypes.PASS,
                "tput_check": ReportCheckTypes.PASS,
            },
            "complete": {
                "ttft_check": ReportCheckTypes.FAIL,
                "tput_check": ReportCheckTypes.PASS,
            },
            "target": {
                "ttft_check": ReportCheckTypes.FAIL,
                "tput_check": ReportCheckTypes.FAIL,
            },
        }

    def test_experimental_passes_even_when_all_fail(self, failing_target_checks):
        result = enforce_acceptance_criteria(
            failing_target_checks, ModelStatusTypes.EXPERIMENTAL
        )
        assert result["enforcement_result"] == "PASS"
        assert result["enforced_tiers"] == []
        assert result["failed_enforced_tiers"] == []
        assert set(result["informational_tiers"]) == {
            "functional",
            "complete",
            "target",
        }

    def test_functional_passes_when_functional_passes(self, failing_target_checks):
        result = enforce_acceptance_criteria(
            failing_target_checks, ModelStatusTypes.FUNCTIONAL
        )
        assert result["enforcement_result"] == "PASS"
        assert result["enforced_tiers"] == ["functional"]
        assert result["failed_enforced_tiers"] == []

    def test_complete_fails_when_complete_fails(self, failing_target_checks):
        result = enforce_acceptance_criteria(
            failing_target_checks, ModelStatusTypes.COMPLETE
        )
        assert result["enforcement_result"] == "FAIL"
        assert "complete" in result["failed_enforced_tiers"]

    def test_top_perf_fails_when_any_tier_fails(self, failing_target_checks):
        result = enforce_acceptance_criteria(
            failing_target_checks, ModelStatusTypes.TOP_PERF
        )
        assert result["enforcement_result"] == "FAIL"
        assert "complete" in result["failed_enforced_tiers"]
        assert "target" in result["failed_enforced_tiers"]

    def test_all_pass(self, passing_target_checks):
        result = enforce_acceptance_criteria(
            passing_target_checks, ModelStatusTypes.TOP_PERF
        )
        assert result["enforcement_result"] == "PASS"
        assert result["failed_enforced_tiers"] == []

    def test_handles_integer_check_values(self):
        """Media model reports use raw integers (2=PASS, 3=FAIL)."""
        target_checks = {
            "functional": {"ttft_check": 2, "tput_check": 3},
            "complete": {"ttft_check": 2, "tput_check": 2},
            "target": {"ttft_check": 2, "tput_check": 2},
        }
        result = enforce_acceptance_criteria(target_checks, ModelStatusTypes.FUNCTIONAL)
        assert result["enforcement_result"] == "FAIL"
        assert "functional" in result["failed_enforced_tiers"]

    def test_na_checks_are_not_failures(self):
        target_checks = {
            "functional": {
                "ttft_check": ReportCheckTypes.NA,
                "tput_check": ReportCheckTypes.PASS,
            },
        }
        result = enforce_acceptance_criteria(target_checks, ModelStatusTypes.FUNCTIONAL)
        assert result["enforcement_result"] == "PASS"

    def test_model_status_is_included_in_result(self, passing_target_checks):
        result = enforce_acceptance_criteria(
            passing_target_checks, ModelStatusTypes.COMPLETE
        )
        assert result["model_status"] == "COMPLETE"


class TestKnownIssue:
    """Tests for KnownIssue and DeviceModelSpec skip logic."""

    def test_known_issue_creation(self):
        ki = KnownIssue(
            workflow_type="BENCHMARKS",
            reason="GH#2600 - OOM on T3K",
            task_name="isl-128_osl-1024_con-32",
        )
        assert ki.workflow_type == "BENCHMARKS"
        assert ki.reason == "GH#2600 - OOM on T3K"
        assert ki.task_name == "isl-128_osl-1024_con-32"

    def test_known_issue_defaults(self):
        ki = KnownIssue(workflow_type="EVALS", reason="test reason")
        assert ki.task_name is None

    def test_device_model_spec_with_known_issues(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="BENCHMARKS",
                    reason="whole workflow skip",
                ),
                KnownIssue(
                    workflow_type="EVALS",
                    reason="specific task skip",
                    task_name="mmlu",
                ),
            ],
        )
        assert len(dms.known_issues) == 2

    def test_should_skip_workflow(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="BENCHMARKS",
                    reason="skip all benchmarks",
                ),
            ],
        )
        assert dms.should_skip_workflow("BENCHMARKS") is not None
        assert dms.should_skip_workflow("EVALS") is None

    def test_should_skip_task_whole_workflow(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="BENCHMARKS",
                    reason="skip all benchmarks",
                ),
            ],
        )
        assert dms.should_skip_task("BENCHMARKS", "any_task") is not None

    def test_should_skip_task_specific(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="EVALS",
                    reason="mmlu broken",
                    task_name="mmlu",
                ),
            ],
        )
        assert dms.should_skip_task("EVALS", "mmlu") is not None
        assert dms.should_skip_task("EVALS", "hellaswag") is None
        assert dms.should_skip_workflow("EVALS") is None

    def test_should_skip_case_insensitive_workflow(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="benchmarks",
                    reason="test",
                ),
            ],
        )
        assert dms.should_skip_workflow("BENCHMARKS") is not None

    def test_no_known_issues_skips_nothing(self):
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
        )
        assert dms.should_skip_workflow("BENCHMARKS") is None
        assert dms.should_skip_task("EVALS", "mmlu") is None

    def test_known_issue_json_roundtrip(self, tmp_path):
        """Test KnownIssue survives JSON serialization/deserialization."""
        impl = ImplSpec(
            impl_id="test-impl",
            impl_name="test-impl",
            repo_url="https://github.com/test/repo",
            code_path="models/test",
        )
        dms = DeviceModelSpec(
            device=DeviceTypes.N150,
            max_concurrency=16,
            max_context=8192,
            known_issues=[
                KnownIssue(
                    workflow_type="BENCHMARKS",
                    reason="GH#100 - test issue",
                    task_name="specific_task",
                ),
                KnownIssue(
                    workflow_type="EVALS",
                    reason="GH#200 - another issue",
                ),
            ],
        )
        spec = ModelSpec(
            device_type=DeviceTypes.N150,
            impl=impl,
            hf_model_repo="test/model-7B",
            model_id="test_id",
            model_name="model-7B",
            tt_metal_commit="v1.0.0",
            vllm_commit="abc123",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_spec=dms,
        )

        json_file = spec.to_json(run_id="test_ki", output_dir=str(tmp_path))
        loaded_spec = ModelSpec.from_json(json_file)

        assert len(loaded_spec.device_model_spec.known_issues) == 2
        ki0 = loaded_spec.device_model_spec.known_issues[0]
        ki1 = loaded_spec.device_model_spec.known_issues[1]
        assert ki0.workflow_type == "BENCHMARKS"
        assert ki0.reason == "GH#100 - test issue"
        assert ki0.task_name == "specific_task"
        assert ki1.workflow_type == "EVALS"
        assert ki1.task_name is None


class TestIsCheckFailing:
    """Tests for _is_check_failing helper."""

    def test_report_check_fail(self):
        assert _is_check_failing(ReportCheckTypes.FAIL) is True

    def test_report_check_pass(self):
        assert _is_check_failing(ReportCheckTypes.PASS) is False

    def test_report_check_na(self):
        assert _is_check_failing(ReportCheckTypes.NA) is False

    def test_integer_fail(self):
        assert _is_check_failing(3) is True  # ReportCheckTypes.FAIL == 3

    def test_integer_pass(self):
        assert _is_check_failing(2) is False  # ReportCheckTypes.PASS == 2

    def test_non_check_values(self):
        assert _is_check_failing("FAIL") is False
        assert _is_check_failing(None) is False


if __name__ == "__main__":
    pytest.main([__file__])

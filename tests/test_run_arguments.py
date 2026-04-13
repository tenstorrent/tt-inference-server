#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import argparse
import os
import sys
import subprocess
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the path so we can import from run.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import (
    parse_arguments,
    handle_secrets,
    get_current_commit_sha,
    populate_model_spec_cli_args,
)
from workflows.validate_setup import (
    validate_runtime_args,
    validate_local_setup,
)
from workflows.device_utils import _get_tt_smi_board_type_counts, infer_default_device
from workflows.model_spec import get_runtime_model_spec
from workflows.run_docker_server import (
    generate_docker_run_command,
)
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes, InferenceEngine, ModelType


@pytest.fixture
def base_args():
    """Base valid arguments for testing."""
    return [
        "--model",
        "Mistral-7B-Instruct-v0.3",
        "--workflow",
        "benchmarks",
        "--tt-device",
        "n150",
    ]


@pytest.fixture
def mock_args():
    """Create a mock args object with default values for parse_arguments tests."""
    return argparse.Namespace(
        model="Mistral-7B-Instruct-v0.3",
        workflow="benchmarks",
        device="n150",
        tt_device="n150",
        engine=None,
        impl="tt-transformers",
        docker_server=False,
        local_server=False,
        interactive=False,
        workflow_args=None,
        override_docker_image=None,
        service_port="8000",
        disable_trace_capture=False,
        percentile_report=False,
        dev_mode=False,
        override_tt_config=None,
        vllm_override_args=None,
        device_id=None,
        reset_venvs=False,
        runtime_model_spec_json=None,
        tt_metal_python_venv_dir=None,
        tt_metal_home=None,
        vllm_dir=None,
        no_auth=False,
        print_docker_cmd=False,
        concurrency_sweeps=False,
        streaming=None,
        preprocessing=None,
        limit_samples_mode=None,
        sdxl_num_prompts="100",
        skip_system_sw_validation=False,
        tools="vllm",
        host_volume=None,
        host_hf_cache=None,
        host_weights_dir=None,
        image_user="1000",
    )


@pytest.fixture
def mock_runtime_config():
    return RuntimeConfig(
        model="Mistral-7B-Instruct-v0.3",
        workflow="benchmarks",
        device="n150",
    )


@pytest.fixture
def mock_model_spec():
    """Create a mock model_spec object with default values."""
    mock_spec = MagicMock()
    mock_spec.model_id = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
    mock_spec.model_name = "Mistral-7B-Instruct-v0.3"
    mock_spec.tt_metal_commit = "test-commit"
    mock_spec.vllm_commit = "test-vllm-commit"
    mock_spec.inference_engine = "vLLM"
    mock_spec.to_json.return_value = "/tmp/test-model-spec.json"

    return mock_spec


@pytest.fixture
def mock_setup_config():
    """Mock setup configuration for docker server (default Docker volume mode)."""
    mock_config = MagicMock()
    mock_config.cache_root = Path("/tmp/cache")
    mock_config.container_model_spec_dir = Path("/home/container_app_user/model_spec")
    mock_config.container_tt_metal_cache_dir = Path("/container/cache")
    mock_config.container_model_weights_path = Path(
        "/tmp/cache/weights/Mistral-7B-Instruct-v0.3"
    )
    mock_config.container_model_weights_mount_dir = None
    mock_config.docker_volume_name = (
        "volume_id_tt-transformers-Mistral-7B-Instruct-v0.3"
    )
    mock_config.host_model_volume_root = None
    mock_config.host_model_weights_mount_dir = None
    mock_config.model_source = "huggingface"
    return mock_config


class TestArgumentParsing:
    """Compact tests for argument parsing and validation."""

    def test_required_args_success(self, base_args):
        """Test successful parsing of required arguments."""
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()
        assert args.model == "Mistral-7B-Instruct-v0.3"
        assert args.workflow == "benchmarks"
        assert args.device == "n150"
        assert args.tt_device == "n150"

    @pytest.mark.parametrize(
        "missing_arg,remaining_args",
        [
            ("--model", ["--workflow", "benchmarks"]),
            ("--workflow", ["--model", "Mistral-7B-Instruct-v0.3"]),
        ],
    )
    def test_missing_required_args(self, missing_arg, remaining_args, capsys):
        """Test that missing required arguments show proper error messages."""
        with patch("sys.argv", ["run.py"] + remaining_args):
            with pytest.raises(SystemExit):
                parse_arguments()
        captured = capsys.readouterr()
        assert f"the following arguments are required: {missing_arg}" in captured.err

    @pytest.mark.parametrize(
        "invalid_arg,invalid_value",
        [
            ("--model", "invalid-model"),
            ("--workflow", "invalid-workflow"),
            ("--tt-device", "invalid-device"),
        ],
    )
    def test_invalid_choices(self, base_args, invalid_arg, invalid_value):
        """Test that invalid choices raise SystemExit."""
        args = base_args.copy()
        idx = args.index(invalid_arg) + 1
        args[idx] = invalid_value
        with patch("sys.argv", ["run.py"] + args):
            with patch("sys.stderr") as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    parse_arguments()

                # Verify it's an error exit (code 2 for argparse errors)
                assert exc_info.value.code == 2

                # Verify error message was written to stderr
                stderr_calls = [str(call) for call in mock_stderr.write.call_args_list]
                stderr_output = "".join(stderr_calls)
                assert (
                    "invalid choice" in stderr_output.lower()
                    or "error" in stderr_output.lower()
                )


class TestModelSpecCliArgsCompatibility:
    def test_populate_model_spec_cli_args_uses_runtime_config_values(self):
        runtime_config = RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="benchmarks",
            device="n150",
            docker_server=True,
            override_docker_image="ghcr.io/example/image:latest",
            streaming="true",
            preprocessing="false",
            sdxl_num_prompts="42",
        )
        runtime_config.run_id = "run-123"

        model_spec = MagicMock()
        model_spec.cli_args = {"stale_key": "stale_value"}

        populate_model_spec_cli_args(model_spec, runtime_config)

        assert "stale_key" not in model_spec.cli_args
        assert model_spec.cli_args["model"] == "Mistral-7B-Instruct-v0.3"
        assert model_spec.cli_args["workflow"] == "benchmarks"
        assert model_spec.cli_args["device"] == "n150"
        assert model_spec.cli_args["tt_device"] == "n150"
        assert model_spec.cli_args["docker_server"] is True
        assert (
            model_spec.cli_args["override_docker_image"]
            == "ghcr.io/example/image:latest"
        )
        assert model_spec.cli_args["streaming"] == "true"
        assert model_spec.cli_args["preprocessing"] == "false"
        assert model_spec.cli_args["sdxl_num_prompts"] == "42"
        assert model_spec.cli_args["run_id"] == "run-123"

    @pytest.mark.parametrize(
        "device_ids,expected",
        [("0", [0]), ("0,1", [0, 1]), ("0,1,2,3", [0, 1, 2, 3]), ("0,3", [0, 3])],
    )
    def test_parse_device_ids_valid(self, base_args, device_ids, expected):
        """Test valid device-id values."""
        full_args = base_args.copy()
        full_args += ["--device-id", device_ids]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()
        assert args.device_id == expected

    @pytest.mark.parametrize(
        "device_ids",
        [
            "0 1",  # space instead of comma
            "-1",  # negative value
            "1,-2,3",  # mixed negative
            "abc",  # non-integer
            "1,,2",  # empty entry
            "",  # empty string
        ],
    )
    def test_parse_device_ids_invalid(self, base_args, device_ids):
        """Test invalid device-id values raise the correct error."""
        full_args = base_args.copy()
        full_args += ["--device-id", device_ids]
        with patch("sys.argv", ["run.py"] + full_args):
            with patch("sys.stderr") as mock_stderr:
                with pytest.raises(SystemExit) as exc_info:
                    parse_arguments()

                # Verify it's an error exit (code 2 for argparse errors)
                assert exc_info.value.code == 2

                # Verify error message was written to stderr
                stderr_calls = [str(call) for call in mock_stderr.write.call_args_list]
                stderr_output = "".join(stderr_calls)
                assert (
                    "invalid choice" in stderr_output.lower()
                    or "error" in stderr_output.lower()
                )

    @pytest.mark.parametrize(
        "override_arg,test_value",
        [
            ("--override-tt-config", '{"data_parallel": 16}'),
            (
                "--vllm-override-args",
                '{"max_model_len": 4096, "enable_chunked_prefill": true}',
            ),
        ],
    )
    def test_override_args_parsing(self, base_args, override_arg, test_value):
        """Test parsing of override arguments."""
        args_with_override = base_args + [override_arg, test_value]

        with patch("sys.argv", ["run.py"] + args_with_override):
            args = parse_arguments()

        if override_arg == "--override-tt-config":
            assert args.override_tt_config == test_value
        else:
            assert args.vllm_override_args == test_value

    def test_optional_args_and_defaults(self, base_args):
        """Test optional arguments and default values."""
        # Test with all optional args
        full_args = base_args + [
            "--impl",
            "tt-transformers",
            "--local-server",
            "--docker-server",
            "--interactive",
            "--workflow-args",
            "param=value",
            "--service-port",
            "9000",
            "--disable-trace-capture",
            "--dev-mode",
            "--override-docker-image",
            "custom:latest",
            "--device-id",
            "1",
            "--override-tt-config",
            '{"data_parallel": 16}',
            "--vllm-override-args",
            '{"max_model_len": 4096}',
            "--no-auth",
            "--concurrency-sweeps",
            "--tt-metal-home",
            "/opt/tt-metal",
            "--vllm-dir",
            "/opt/vllm",
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()

        assert args.impl == "tt-transformers"
        assert args.local_server is True
        assert args.docker_server is True
        assert args.interactive is True
        assert args.workflow_args == "param=value"
        assert args.service_port == "9000"
        assert args.disable_trace_capture is True
        assert args.dev_mode is True
        assert args.override_docker_image == "custom:latest"
        assert args.device_id == [1]
        assert args.override_tt_config == '{"data_parallel": 16}'
        assert args.vllm_override_args == '{"max_model_len": 4096}'
        assert args.no_auth is True
        assert args.concurrency_sweeps is True
        assert args.tt_metal_home == "/opt/tt-metal"
        assert args.vllm_dir == "/opt/vllm"

        # Test defaults
        with patch("sys.argv", ["run.py"] + base_args):
            args = parse_arguments()

        assert args.local_server is False
        assert args.docker_server is False
        assert args.interactive is False
        assert args.workflow_args is None
        assert args.service_port == "8000"
        assert args.disable_trace_capture is False
        assert args.dev_mode is False
        assert args.override_docker_image is None
        assert args.override_tt_config is None
        assert args.vllm_override_args is None
        assert args.no_auth is False
        assert args.concurrency_sweeps is False
        assert args.host_volume is None
        assert args.host_hf_cache is None
        assert args.image_user == "1000"
        assert args.engine is None
        assert args.tt_metal_home is None
        assert args.vllm_dir is None

    def test_tt_metal_home_parsing(self, base_args):
        """Test --tt-metal-home parsing."""
        full_args = base_args + ["--local-server", "--tt-metal-home", "/srv/tt-metal"]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()

        assert args.local_server is True
        assert args.tt_metal_home == "/srv/tt-metal"

    def test_tt_metal_home_defaults_from_env(self, base_args):
        """Test --tt-metal-home falls back to TT_METAL_HOME env var."""
        full_args = base_args + ["--local-server"]
        with patch.dict(os.environ, {"TT_METAL_HOME": "/env/tt-metal"}, clear=False):
            with patch("sys.argv", ["run.py"] + full_args):
                args = parse_arguments()

        assert args.tt_metal_home == "/env/tt-metal"

    def test_vllm_dir_defaults_from_env(self, base_args):
        """Test --vllm-dir falls back to vllm_dir env var."""
        full_args = base_args + ["--local-server", "--tt-metal-home", "/srv/tt-metal"]
        with patch.dict(os.environ, {"vllm_dir": "/env/vllm"}, clear=False):
            with patch("sys.argv", ["run.py"] + full_args):
                args = parse_arguments()

        assert args.vllm_dir == "/env/vllm"

    def test_vllm_dir_defaults_from_tt_metal_home(self, base_args):
        """Test --vllm-dir defaults to tt-metal-home/vllm."""
        full_args = base_args + ["--local-server", "--tt-metal-home", "/srv/tt-metal"]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()

        assert args.vllm_dir == "/srv/tt-metal/vllm"

    def test_host_hf_cache_bare_flag_defaults_from_env(self, base_args):
        """Test bare --host-hf-cache resolves via HOST_HF_HOME/HF_HOME defaults."""
        full_args = base_args + ["--host-hf-cache"]
        with patch.dict(
            os.environ,
            {
                "HOST_HF_HOME": "/host/hf-cache",
                "HF_HOME": "/env/hf-home",
            },
            clear=False,
        ):
            with patch("sys.argv", ["run.py"] + full_args):
                args = parse_arguments()

        assert args.host_hf_cache == "/host/hf-cache"

    def test_device_alias_compatibility(self):
        """Test --device alias remains supported."""
        with patch(
            "sys.argv",
            [
                "run.py",
                "--model",
                "Mistral-7B-Instruct-v0.3",
                "--workflow",
                "benchmarks",
                "--device",
                "n150",
            ],
        ):
            args = parse_arguments()

        assert args.device == "n150"
        assert args.tt_device == "n150"

    def test_engine_parsing(self, base_args):
        """Test --engine parsing and normalization."""
        with patch("sys.argv", ["run.py"] + base_args + ["--engine", "vllm"]):
            args = parse_arguments()
        assert args.engine == "vLLM"

    def test_docker_volume_args(self, base_args):
        """Test --host-volume, --host-hf-cache, --image-user parsing."""
        full_args = base_args + [
            "--host-volume",
            "/some/path",
            "--host-hf-cache",
            "/home/user/.cache/huggingface",
            "--image-user",
            "15863",
        ]
        with patch("sys.argv", ["run.py"] + full_args):
            args = parse_arguments()

        assert args.host_volume == "/some/path"
        assert args.host_hf_cache == "/home/user/.cache/huggingface"
        assert args.image_user == "15863"


class TestArgsInference:
    """Tests for argument inference and validation."""

    @staticmethod
    def _build_spec(model_id, impl_name="tt-transformers", default_impl=True):
        spec = MagicMock()
        spec.model_id = model_id
        spec.model_name = "Mistral-7B-Instruct-v0.3"
        spec.device_type = DeviceTypes.N150
        spec.inference_engine = "vLLM"
        spec.impl.impl_name = impl_name
        spec.device_model_spec.default_impl = default_impl
        spec.apply_overrides = MagicMock()
        return spec

    def test_infer_impl_success(self):
        """Test successful impl inference via get_runtime_model_spec."""
        mock_model_spec = self._build_spec(
            "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
        )

        with patch.dict(
            "workflows.model_spec.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            result, resolved_impl, resolved_engine = get_runtime_model_spec(
                model="Mistral-7B-Instruct-v0.3", device="n150"
            )

            assert resolved_impl == "tt-transformers"
            assert resolved_engine == "vLLM"
            assert result == mock_model_spec

    def test_infer_impl_already_set(self):
        """Test that existing impl is preserved."""
        mock_model_spec = self._build_spec(
            "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
        )

        with patch.dict(
            "workflows.model_spec.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            result, resolved_impl, _ = get_runtime_model_spec(
                model="Mistral-7B-Instruct-v0.3",
                device="n150",
                impl="tt-transformers",
            )

            assert resolved_impl == "tt-transformers"
            assert result == mock_model_spec

    def test_infer_impl_no_default(self):
        """Test error when no default impl available."""
        spec = self._build_spec(
            "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150", default_impl=False
        )
        with patch.dict("workflows.model_spec.MODEL_SPECS", {spec.model_id: spec}):
            with pytest.raises(ValueError, match="does not have a default impl"):
                get_runtime_model_spec(model="Mistral-7B-Instruct-v0.3", device="n150")

    def test_engine_filters_selection(self):
        """Test --engine filters runtime model selection."""
        vllm_spec = self._build_spec(
            "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150_vllm"
        )
        vllm_spec.inference_engine = "vLLM"
        media_spec = self._build_spec(
            "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150_media"
        )
        media_spec.inference_engine = "media"

        with patch.dict(
            "workflows.model_spec.MODEL_SPECS",
            {
                vllm_spec.model_id: vllm_spec,
                media_spec.model_id: media_spec,
            },
        ):
            result, _, resolved_engine = get_runtime_model_spec(
                model="Mistral-7B-Instruct-v0.3", device="n150", engine="media"
            )
            assert result == media_spec
            assert resolved_engine == "media"

    def test_infer_default_device_uses_tt_smi_counts_for_n300(self):
        """Infer N300 from tt-smi board counts."""
        with patch(
            "workflows.device_utils._collect_supported_devices_for_model",
            return_value={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        ), patch(
            "workflows.device_utils._get_tt_smi_board_type_counts",
            return_value={"n300": 1},
        ):
            inferred = infer_default_device("Mistral-7B-Instruct-v0.3")
            assert inferred == "n300"

    def test_infer_default_device_uses_tt_smi_counts_for_t3k(self):
        """Infer T3K from tt-smi board counts."""
        with patch(
            "workflows.device_utils._collect_supported_devices_for_model",
            return_value={DeviceTypes.N300, DeviceTypes.T3K},
        ), patch(
            "workflows.device_utils._get_tt_smi_board_type_counts",
            return_value={"n300": 4},
        ):
            inferred = infer_default_device("Mistral-7B-Instruct-v0.3")
            assert inferred == "t3k"

    def test_infer_default_device_fails_for_unmapped_tt_smi_counts(self):
        """Raise when tt-smi board counts are unmapped."""
        with patch(
            "workflows.device_utils._collect_supported_devices_for_model",
            return_value={DeviceTypes.N300, DeviceTypes.T3K},
        ), patch(
            "workflows.device_utils._get_tt_smi_board_type_counts",
            return_value={"n300 l": 3, "n300 r": 3},
        ):
            with pytest.raises(ValueError, match="Unable to map tt-smi board counts"):
                infer_default_device("Mistral-7B-Instruct-v0.3")

    def test_tt_smi_counts_n300_chips_converted_to_boards(self):
        """n300 L/R chip entries are collapsed and halved to board count."""
        tt_smi_output = """{
            "device_info": [
                {"board_info": {"board_type": "n300 L"}},
                {"board_info": {"board_type": "n300 R"}}
            ]
        }"""
        with patch("workflows.device_utils._ensure_tt_smi_venv_setup"), patch(
            "subprocess.check_output", return_value=tt_smi_output
        ):
            counts = _get_tt_smi_board_type_counts()
        assert counts == {"n300": 1}

    def test_tt_smi_counts_n300_odd_chip_count_raises(self):
        """Odd n300 chip counts raise a validation error."""
        tt_smi_output = """{
            "device_info": [
                {"board_info": {"board_type": "n300 L"}},
                {"board_info": {"board_type": "n300 R"}},
                {"board_info": {"board_type": "n300 L"}}
            ]
        }"""
        with patch("workflows.device_utils._ensure_tt_smi_venv_setup"), patch(
            "subprocess.check_output", return_value=tt_smi_output
        ):
            with pytest.raises(ValueError, match="Expected an even number of chips"):
                _get_tt_smi_board_type_counts()

    def test_infer_default_device_tt_galaxy_wh_32_chips(self):
        """Galaxy WH boards map from 32 reported tt-galaxy-wh chips."""
        with patch(
            "workflows.device_utils._collect_supported_devices_for_model",
            return_value={DeviceTypes.T3K, DeviceTypes.GALAXY},
        ), patch(
            "workflows.device_utils._get_tt_smi_board_type_counts",
            return_value={"tt-galaxy-wh": 32},
        ):
            inferred = infer_default_device("Mistral-7B-Instruct-v0.3")
            assert inferred == "galaxy"


class TestRuntimeValidation:
    """Tests for runtime argument validation."""

    @pytest.mark.parametrize(
        "workflow,should_pass",
        [
            ("benchmarks", True),
            ("evals", True),  # Mistral-7B-Instruct-v0.3 is in EVAL_CONFIGS
            ("reports", True),
            ("release", True),  # Mistral-7B-Instruct-v0.3 is in both configs
            ("stress_tests", True),
        ],
    )
    def test_workflow_validation(
        self, mock_model_spec, mock_runtime_config, workflow, should_pass
    ):
        """Test validation for different workflows."""
        mock_runtime_config.workflow = workflow
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            if should_pass:
                validate_runtime_args(mock_model_spec, mock_runtime_config)
            else:
                with pytest.raises(AssertionError):
                    validate_runtime_args(mock_model_spec, mock_runtime_config)

    def test_server_workflow_validation(self, mock_model_spec, mock_runtime_config):
        """Test server workflow specific validation."""
        mock_runtime_config.workflow = "server"

        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            # Should fail without docker or local server
            with pytest.raises(
                ValueError, match="requires --docker-server or --local-server"
            ):
                validate_runtime_args(mock_model_spec, mock_runtime_config)

            # Should pass with docker server
            mock_runtime_config.docker_server = True
            validate_runtime_args(mock_model_spec, mock_runtime_config)

            # Should pass with local server when tt-metal home is provided
            mock_runtime_config.docker_server = False
            mock_runtime_config.local_server = True
            mock_runtime_config.tt_metal_home = "/opt/tt-metal"
            validate_runtime_args(mock_model_spec, mock_runtime_config)

    def test_conflicting_server_options(self, mock_model_spec, mock_runtime_config):
        """Test that both docker and local server raises error."""
        mock_runtime_config.docker_server = True
        mock_runtime_config.local_server = True
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            with pytest.raises(
                AssertionError, match="Cannot run --docker-server and --local-server"
            ):
                validate_runtime_args(mock_model_spec, mock_runtime_config)


class TestOverrideArgsIntegration:
    """Test override arguments integration with model_spec apply_runtime_args."""

    @pytest.mark.parametrize(
        "override_type,cli_arg_name,test_value",
        [
            ("tt_config", "override_tt_config", '{"data_parallel": 16}'),
            ("vllm_args", "vllm_override_args", '{"max_model_len": 4096}'),
        ],
    )
    def test_get_runtime_model_spec_selects_correct_spec(
        self, override_type, cli_arg_name, test_value
    ):
        """Test that get_runtime_model_spec selects the correct spec (caller applies overrides)."""
        mock_model_spec = MagicMock()
        mock_model_spec.model_id = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
        mock_model_spec.model_name = "Mistral-7B-Instruct-v0.3"
        mock_model_spec.device_type = DeviceTypes.N150
        mock_model_spec.inference_engine = "vLLM"
        mock_model_spec.impl.impl_name = "tt-transformers"
        mock_model_spec.device_model_spec.default_impl = True

        with patch.dict(
            "workflows.model_spec.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ):
            result, resolved_impl, resolved_engine = get_runtime_model_spec(
                model="Mistral-7B-Instruct-v0.3",
                device="n150",
                impl="tt-transformers",
            )

            assert result == mock_model_spec
            assert resolved_impl == "tt-transformers"
            assert resolved_engine == "vLLM"

    def _make_mock_model_spec(self):
        """Helper to create a mock model_spec for docker command tests."""
        mock_model_spec = MagicMock()
        mock_model_spec.model_id = "test-model-id"
        mock_model_spec.model_name = "Mistral-7B-Instruct-v0.3"
        mock_model_spec.device_type = "n150"
        mock_model_spec.docker_image = "test:image"
        mock_model_spec.impl.impl_name = "tt-transformers"
        mock_model_spec.impl.impl_id = "tt-transformers"
        mock_model_spec.hf_model_repo = "mistralai/Mistral-7B-Instruct-v0.3"
        mock_model_spec.subdevice_type = None
        mock_model_spec.inference_engine = "vLLM"
        return mock_model_spec

    def _make_mock_runtime_config(self):
        return RuntimeConfig(
            model="Mistral-7B-Instruct-v0.3",
            workflow="server",
            device="n150",
            service_port="8000",
        )

    def test_generate_docker_run_command_mounts_runtime_spec_json(
        self, mock_setup_config
    ):
        """Test that generate_docker_run_command mounts the runtime spec JSON in dev mode."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()
        mock_runtime_config.dev_mode = True

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, container_name = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_setup_config, json_fpath
            )

            assert container_name == "tt-inference-server-test123"

            json_mount_found = False
            env_var_found = False

            for i, arg in enumerate(docker_command):
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    if (
                        "test-model-spec.json" in mount_spec
                        and "readonly" in mount_spec
                    ):
                        json_mount_found = True

                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    if env_setting.startswith("RUNTIME_MODEL_SPEC_JSON_PATH="):
                        env_var_found = True
                        assert "test-model-spec.json" in env_setting

            assert json_mount_found, (
                f"JSON file mount not found in docker command: {docker_command}"
            )
            assert env_var_found, (
                f"RUNTIME_MODEL_SPEC_JSON_PATH not found in docker command: {docker_command}"
            )

    def test_default_mode_uses_docker_volume_and_user(self, mock_setup_config):
        """Test default mode emits type=volume mount and --user 1000."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_setup_config, json_fpath
            )

            assert "--volume" in docker_command, (
                "Default mode should use Docker named volume"
            )
            assert "--user" not in docker_command, (
                "Default image_user=1000 should not emit --user (Dockerfile default)"
            )
            # Should NOT have CACHE_ROOT env var (baked into Dockerfile)
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    assert not docker_command[i + 1].startswith("CACHE_ROOT="), (
                        "CACHE_ROOT should not be set as runtime env var"
                    )

    def test_host_volume_mode_uses_bind_mount(self):
        """Test --host-volume mode emits type=bind mount."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()
        mock_config = MagicMock()
        mock_config.cache_root = Path("/tmp/cache")
        mock_config.container_model_spec_dir = Path(
            "/home/container_app_user/model_spec"
        )
        mock_config.container_tt_metal_cache_dir = Path("/container/cache")
        mock_config.container_model_weights_path = Path("/tmp/cache/weights/model")
        mock_config.container_model_weights_mount_dir = None
        mock_config.host_model_volume_root = Path("/host/volumes/cache")
        mock_config.host_model_weights_mount_dir = None
        mock_config.model_source = "huggingface"

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_config, json_fpath
            )

            cmd_str = " ".join(str(c) for c in docker_command)
            assert "type=bind,src=/host/volumes/cache" in cmd_str, (
                "Host volume mode should use bind mount"
            )

    def test_host_hf_cache_mode_mounts_weights_readonly(self):
        """Test --host-hf-cache mode emits separate readonly weights mount."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()
        mock_config = MagicMock()
        mock_config.cache_root = Path("/tmp/cache")
        mock_config.container_model_spec_dir = Path(
            "/home/container_app_user/model_spec"
        )
        mock_config.container_tt_metal_cache_dir = Path("/container/cache")
        mock_config.container_model_weights_path = Path(
            "/container/readonly_weights_mount/model/snapshots/abc"
        )
        mock_config.container_model_weights_mount_dir = Path(
            "/container/readonly_weights_mount/model"
        )
        mock_config.host_model_volume_root = None
        mock_config.host_model_weights_mount_dir = Path(
            "/host/hf_cache/hub/models--meta-llama"
        )
        mock_config.model_source = "huggingface"

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_config, json_fpath
            )

            # Should have readonly weights mount
            weights_mount_found = False
            for i, arg in enumerate(docker_command):
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    if "models--meta-llama" in mount_spec and "readonly" in mount_spec:
                        weights_mount_found = True
            assert weights_mount_found, (
                "Host HF cache mode should mount weights readonly"
            )

    def test_default_mode_no_separate_weights_mount(self, mock_setup_config):
        """Test default mode (no --host-hf-cache) does NOT emit separate weights mount."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_setup_config, json_fpath
            )

            # Should NOT have readonly weights mount (host_model_weights_mount_dir is None)
            for i, arg in enumerate(docker_command):
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    assert "readonly_weights_mount" not in mount_spec, (
                        "Default mode should not have separate readonly weights mount"
                    )

    def test_dev_mode_does_not_duplicate_model_spec_mount(self, mock_setup_config):
        """Test dev mode mounts model_spec JSON exactly once."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()
        mock_runtime_config.dev_mode = True

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            json_fpath = Path("/tmp/test-model-spec.json")
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_setup_config, json_fpath
            )

            json_mount_count = 0
            for i, arg in enumerate(docker_command):
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    if (
                        "test-model-spec.json" in mount_spec
                        and "readonly" in mount_spec
                    ):
                        json_mount_count += 1

            assert json_mount_count == 1, (
                "Dev mode should not duplicate model spec JSON mount"
            )

    def test_no_runtime_spec_mount_without_json_fpath(self, mock_setup_config):
        """Test that no runtime spec mount or env var is emitted when json_fpath is None."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            docker_command, _ = generate_docker_run_command(
                mock_model_spec, mock_runtime_config, mock_setup_config
            )

            for i, arg in enumerate(docker_command):
                if arg == "--mount" and i + 1 < len(docker_command):
                    mount_spec = docker_command[i + 1]
                    assert (
                        "model_specs" not in mount_spec
                        or "default_model_spec" in mount_spec
                    ), "No runtime spec mount should be present without json_fpath"
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    assert not env_setting.startswith(
                        "RUNTIME_MODEL_SPEC_JSON_PATH="
                    ), (
                        "RUNTIME_MODEL_SPEC_JSON_PATH should not be set without json_fpath"
                    )

    def test_generate_docker_run_command_without_setup_config(self):
        """Test that generate_docker_run_command works without setup_config for --print-docker-cmd."""
        mock_model_spec = self._make_mock_model_spec()
        mock_runtime_config = self._make_mock_runtime_config()

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            docker_command, container_name = generate_docker_run_command(
                mock_model_spec, mock_runtime_config
            )

            assert container_name == "tt-inference-server-test123"
            # Command should contain base elements
            assert "docker" in docker_command
            assert "run" in docker_command
            assert "--rm" in docker_command
            assert "test:image" in docker_command
            assert "--ipc" in docker_command
            assert "--tt-device" in docker_command
            tt_device_index = docker_command.index("--tt-device")
            assert docker_command[tt_device_index + 1] == "n150"
            # Should NOT contain setup_config-dependent mounts or env vars
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    env_setting = docker_command[i + 1]
                    assert not env_setting.startswith("CACHE_ROOT=")
                    assert not env_setting.startswith("TT_MODEL_SPEC_JSON_PATH=")
                    assert not env_setting.startswith("RUNTIME_MODEL_SPEC_JSON_PATH=")

    def test_generate_docker_run_command_media_service_port(self):
        """Test that generate_docker_run_command passes SERVICE_PORT for media engine containers."""
        mock_model_spec = self._make_mock_model_spec()
        mock_model_spec.inference_engine = "media"
        mock_model_spec.model_type = ModelType.AUDIO
        mock_model_spec.device_type = MagicMock()
        mock_model_spec.device_type.name.lower.return_value = "n150"
        mock_runtime_config = self._make_mock_runtime_config()
        mock_runtime_config.service_port = "7001"

        with patch(
            "workflows.run_docker_server.get_repo_root_path", return_value=Path("/tmp")
        ), patch("workflows.run_docker_server.DeviceTypes"), patch(
            "workflows.run_docker_server.short_uuid", return_value="test123"
        ):
            docker_command, container_name = generate_docker_run_command(
                mock_model_spec, mock_runtime_config
            )

            assert container_name == "tt-inference-server-test123"
            # Verify SERVICE_PORT is passed as an environment variable
            service_port_found = False
            for i, arg in enumerate(docker_command):
                if arg == "-e" and i + 1 < len(docker_command):
                    if docker_command[i + 1] == "SERVICE_PORT=7001":
                        service_port_found = True
                        break
            assert service_port_found, (
                "SERVICE_PORT=7001 should be set in docker environment variables"
            )


class TestSecretsHandling:
    """Tests for secrets handling functionality."""

    @pytest.mark.parametrize(
        "workflow,docker_server,interactive,no_auth,jwt_required,hf_required",
        [
            ("benchmarks", False, False, False, False, False),  # Client-side
            ("evals", False, False, False, False, False),  # Client-side
            ("server", True, False, False, True, True),  # Server with docker
            ("server", False, False, False, False, True),  # Server without docker
            ("server", True, True, False, False, False),  # Interactive mode
            ("server", True, False, True, False, True),  # Server with docker + no-auth
            ("release", False, False, False, False, True),  # Non-client workflow
            ("reports", False, False, False, False, True),  # Non-client workflow
        ],
    )
    def test_secrets_requirements(
        self,
        mock_runtime_config,
        workflow,
        docker_server,
        interactive,
        no_auth,
        jwt_required,
        hf_required,
    ):
        """Test secret requirements for different configurations."""
        mock_runtime_config.workflow = workflow
        mock_runtime_config.docker_server = docker_server
        mock_runtime_config.interactive = interactive
        mock_runtime_config.no_auth = no_auth

        env_vars = {}
        if jwt_required:
            env_vars["JWT_SECRET"] = "test-jwt"
        if hf_required:
            env_vars["HF_TOKEN"] = "test-hf"

        with patch("run.load_dotenv", return_value=True):
            with patch.dict(os.environ, env_vars, clear=True):
                if jwt_required or hf_required:
                    if not env_vars:
                        with pytest.raises(
                            AssertionError, match="is not set in .env file"
                        ):
                            handle_secrets(mock_runtime_config)
                    else:
                        handle_secrets(mock_runtime_config)
                else:
                    handle_secrets(mock_runtime_config)

    @patch("run.load_dotenv")
    @patch("run.write_dotenv")
    @patch("getpass.getpass")
    def test_secrets_prompting(
        self, mock_getpass, mock_write_dotenv, mock_load_dotenv, mock_runtime_config
    ):
        """Test prompting for missing secrets."""
        mock_runtime_config.workflow = "server"
        mock_runtime_config.docker_server = True
        mock_runtime_config.interactive = False

        mock_load_dotenv.side_effect = [False, True]
        mock_getpass.side_effect = ["test-jwt", "test-hf"]

        with patch.dict(os.environ, {}, clear=True):
            handle_secrets(mock_runtime_config)

        assert mock_getpass.call_count == 2
        mock_write_dotenv.assert_called_once()


class TestUtilityFunctions:
    """Tests for utility functions."""

    @patch("subprocess.check_output")
    def test_get_commit_sha(self, mock_check_output):
        """Test git commit SHA retrieval."""
        mock_check_output.return_value = b"abc123def456\n"
        result = get_current_commit_sha()
        assert result == "abc123def456"

        # Test error handling
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            get_current_commit_sha()

    @patch("workflows.validate_setup.ensure_readwriteable_dir")
    @patch("workflows.validate_setup.get_default_workflow_root_log_dir")
    def test_validate_local_setup(
        self,
        mock_get_log_dir,
        mock_ensure_dir,
        mock_model_spec,
        mock_runtime_config,
    ):
        """Test local setup validation."""
        mock_log_dir = Path("/tmp/test_logs")
        mock_get_log_dir.return_value = mock_log_dir

        # Create a temporary directory for the model spec JSON
        with patch.dict(
            "workflows.validate_setup.MODEL_SPECS",
            {mock_model_spec.model_id: mock_model_spec},
        ), tempfile.TemporaryDirectory() as tempdir:
            # dump the ModelSpec to a tempdir
            model_spec_path = mock_model_spec.to_json(run_id="temp", output_dir=tempdir)
            validate_local_setup(mock_model_spec, mock_runtime_config, model_spec_path)

        mock_get_log_dir.assert_called_once()
        mock_ensure_dir.assert_called_once_with(mock_log_dir)


if __name__ == "__main__":
    pytest.main([__file__])

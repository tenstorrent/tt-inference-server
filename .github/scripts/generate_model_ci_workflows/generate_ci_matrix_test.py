# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for generate_ci_matrix.py
"""

import pytest
import tempfile
import json
import yaml
import os
from contextlib import contextmanager
import generate_ci_matrix


@contextmanager
def temp_file_from_string(content, suffix='.json'):
    """
    Context manager that creates a temporary file from a raw string.

    Args:
        content: String content to write to file
        suffix: File extension (default: '.json')

    Yields:
        str: Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        yield temp_path
    finally:
        os.unlink(temp_path)


@contextmanager
def temp_config_file(data_dict, format='json'):
    """
    Context manager that creates a temporary config file in JSON or YAML format.

    Args:
        data_dict: Dictionary to write to file
        format: File format - 'json' or 'yaml' (default: 'json')

    Yields:
        str: Path to the temporary config file
    """
    if format == 'json':
        content = json.dumps(data_dict)
        suffix = '.json'
    elif format == 'yaml':
        content = yaml.dump(data_dict)
        suffix = '.yaml'
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

    with temp_file_from_string(content, suffix) as temp_path:
        yield temp_path


class TestArgumentValidation:
    """Tests for CLI argument validation using argparse choices"""

    def test_rejects_invalid_schedule(self, monkeypatch):
        """Test that invalid schedule argument is rejected by argparse."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['generate_ci_matrix.py', '--schedule', 'invalid_schedule'])

        with pytest.raises(SystemExit) as exc_info:
            generate_ci_matrix.main()

        assert exc_info.value.code == 2

    def test_rejects_invalid_device(self, monkeypatch):
        """Test that invalid device argument is rejected by argparse."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['generate_ci_matrix.py', '--schedule', 'nightly', '--device', 'INVALID_DEVICE'])

        with pytest.raises(SystemExit) as exc_info:
            generate_ci_matrix.main()

        assert exc_info.value.code == 2

    def test_rejects_invalid_server_type(self, monkeypatch):
        """Test that invalid server-type argument is rejected by argparse."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['generate_ci_matrix.py', '--schedule', 'nightly', '--server-type', 'invalid-server'])

        with pytest.raises(SystemExit) as exc_info:
            generate_ci_matrix.main()

        # argparse exits with code 2 for invalid arguments
        assert exc_info.value.code == 2

    def test_requires_config_argument(self, monkeypatch):
        """Test that --config is a required argument."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['generate_ci_matrix.py', '--schedule', 'nightly'])

        with pytest.raises(SystemExit) as exc_info:
            generate_ci_matrix.main()

        assert exc_info.value.code == 2

    def test_schema_argument_is_optional(self, monkeypatch):
        """Test that --schema is optional (omitting it does not cause argparse error code 2)."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['generate_ci_matrix.py', '--schedule', 'nightly', '--config', '/tmp/non_existent.json'])

        with pytest.raises(SystemExit) as exc_info:
            generate_ci_matrix.main()

        # Should exit with 1 (file not found, handled by except), not 2 (argparse error)
        assert exc_info.value.code == 1


class TestConfigLoading:
    def test_raises_error_when_no_config_file_exists(self):
        """Test that FileNotFoundError is raised when config file doesn't exist."""
        non_existent_path = "/tmp/non_existent_config_12345.json"

        with pytest.raises(FileNotFoundError, match="Config file not found at"):
            generate_ci_matrix.load_ci_config(config_path=non_existent_path)

    def test_raises_error_when_config_path_param_does_not_exist(self):
        """Test that FileNotFoundError is raised when config_path param points to non-existent file."""
        non_existent_path = "/tmp/non_existent_config_via_param_67890.json"

        with pytest.raises(FileNotFoundError, match="Config file not found at"):
            generate_ci_matrix.load_ci_config(config_path=non_existent_path)

    def test_load_ci_config_without_schema_skips_validation(self):
        """Test that omitting schema_path skips validation and loads config regardless of validity."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM"
                    # Missing required "ci" field - would fail schema validation
                }
            }
        }
        with temp_config_file(config) as config_path:
            # Should not raise even though config is missing required "ci" field
            generate_ci_matrix.load_ci_config(config_path=config_path)


class TestConfigValidation:
    """Tests for JSON schema validation."""

    _SCHEMA_CONTENT = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CI Models Configuration",
        "description": "Schema for models CI configuration",
        "type": "object",
        "$defs": {
            "schedule_config": {
                "type": "object",
                "properties": {
                    "devices": {
                        "oneOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["N150", "N300", "T3K", "GALAXY", "GALAXY_BH", "P100", "P150X4", "P150X8", "P300X2"]
                                },
                                "minItems": 1,
                            },
                            {
                                "type": "string",
                                "enum": ["ALL"],
                            }
                        ]
                    }
                },
                "required": ["devices"],
                "additionalProperties": False
            }
        },
        "properties": {
            "models": {
                "type": "object",
                "patternProperties": {
                    ".*": {
                        "type": "object",
                        "properties": {
                            "inference_engine": {
                                "type": "string",
                                "enum": ["vLLM", "MEDIA", "FORGE"]
                            },
                            "ci": {
                                "type": "object",
                                "properties": {
                                    "nightly": {"$ref": "#/$defs/schedule_config"},
                                    "weekly": {"$ref": "#/$defs/schedule_config"},
                                    "bi_weekly": {"$ref": "#/$defs/schedule_config"},
                                    "release": {"$ref": "#/$defs/schedule_config"}
                                },
                                "additionalProperties": False,
                                "minProperties": 1
                            }
                        },
                        "required": ["inference_engine", "ci"]
                    }
                }
            }
        },
        "required": ["models"],
        "additionalProperties": False
    }

    @pytest.fixture(scope="class")
    def schema_path(self, tmp_path_factory):
        schema_file = tmp_path_factory.mktemp("schema") / "schema.json"
        schema_file.write_text(json.dumps(self._SCHEMA_CONTENT))
        return str(schema_file)

    def test_raises_error_when_duplicate_model_names(self, schema_path):
        """Test that ValueError is raised when duplicate model names are found."""
        # Must use raw JSON string because Python dict literals don't allow duplicates
        json_with_duplicates = """
        {
            "models": {
                "test-model": {"inference_engine": "vLLM"},
                "test-model": {"inference_engine": "MEDIA"}
            }
        }
        """
        with temp_file_from_string(json_with_duplicates) as config_path:
            with pytest.raises(ValueError, match="Duplicate key found in JSON"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    @pytest.mark.parametrize("invalid_config", [
        {
            "models": {
                "test-model": {
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        },
        {
            "models": {
                "test-model": {
                    "inference_engine": "InvalidEngine",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
    ])
    def test_raises_error_when_inference_engine_invalid(self, invalid_config, schema_path):
        """Test that ValueError is raised when inference_engine field is missing or invalid."""
        with temp_config_file(invalid_config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_ci_field_missing(self, schema_path):
        """Test that ValueError is raised when ci field is missing from model config."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM"
                    # Missing required "ci" field
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_ci_is_empty(self, schema_path):
        """Test that ValueError is raised when ci object is empty (no schedules)."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {}  # Empty ci object - must have at least one schedule
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_ci_has_invalid_schedule_name(self, schema_path):
        """Test that ValueError is raised when ci has invalid schedule name."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "invalid_schedule": {  # Invalid schedule name
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_schedule_missing_devices(self, schema_path):
        """Test that ValueError is raised when schedule is missing devices field."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            # Missing required "devices" field
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_devices_is_empty(self, schema_path):
        """Test that ValueError is raised when devices array is empty."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": []  # Empty devices array
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_allows_multiple_schedules(self, schema_path):
        """Test that config with multiple schedules is valid."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        },
                        "weekly": {
                            "devices": ["N300"]
                        },
                        "bi_weekly": {
                            "devices": ["GALAXY"]
                        },
                        "release": {
                            "devices": ["T3K"]
                        }
                    }
                }
            }
        }
        # Should not raise - multiple schedules are allowed
        with temp_config_file(config) as config_path:
            generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_allows_single_schedule(self, schema_path):
        """Test that config with just one schedule is valid."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
        # Should not raise - single schedule is allowed
        with temp_config_file(config) as config_path:
            generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_device_name_is_invalid(self, schema_path):
        """Test that ValueError is raised when device name is not in the allowed list."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["INVALID_DEVICE"]  # Invalid device name
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_allows_all_valid_device_names(self, schema_path):
        """Test that all valid device names are accepted."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": [
                                "N150", "N300", "T3K", "GALAXY", "GALAXY_BH",
                                "P100", "P150X4", "P150X8", "P300X2"
                            ]
                        }
                    }
                }
            }
        }
        # Should not raise - all device names are valid
        with temp_config_file(config) as config_path:
            generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_allows_devices_as_string_all(self, schema_path):
        """Test that devices can be the string 'ALL' instead of an array."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": "ALL"  # String instead of array
                        }
                    }
                }
            }
        }
        # Should not raise - "ALL" as string is valid
        with temp_config_file(config) as config_path:
            generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)

    def test_raises_error_when_devices_is_string_but_not_all(self, schema_path):
        """Test that devices as a string only works for 'ALL', not other device names."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": "N150"  # String but not "ALL"
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_ci_config(config_path=config_path, schema_path=schema_path)


class TestGetRunnerConfig:
    """Tests for get_runner_config() function."""

    def test_returns_correct_config_for_common_device(self):
        """Test that common devices return correct config across all schedules."""
        # N150 has same config across all schedules
        assert generate_ci_matrix.get_runner_config("nightly", "N150") == {"label": "n150", "type": "n150"}
        assert generate_ci_matrix.get_runner_config("weekly", "N150") == {"label": "n150", "type": "n150"}
        assert generate_ci_matrix.get_runner_config("release", "N150") == {"label": "n150", "type": "n150"}
        assert generate_ci_matrix.get_runner_config("bi_weekly", "N150") == {"label": "n150", "type": "n150"}

    def test_returns_correct_config_for_t3k(self):
        """Test that T3K device returns correct config."""
        assert generate_ci_matrix.get_runner_config("nightly", "T3K") == {"label": "llmbox", "type": "t3k"}
        assert generate_ci_matrix.get_runner_config("weekly", "T3K") == {"label": "llmbox", "type": "t3k"}

    def test_returns_correct_config_for_galaxy(self):
        """Test that GALAXY device returns correct config."""
        assert generate_ci_matrix.get_runner_config("nightly", "GALAXY") == {"label": "6u", "type": "galaxy"}
        assert generate_ci_matrix.get_runner_config("weekly", "GALAXY") == {"label": "6u", "type": "galaxy"}
        assert generate_ci_matrix.get_runner_config("release", "GALAXY") == {"label": "6u", "type": "galaxy"}
        assert generate_ci_matrix.get_runner_config("bi_weekly", "GALAXY") == {"label": "6u", "type": "galaxy"}

    def test_returns_correct_config_for_galaxy_bh(self):
        """Test that GALAXY_BH device returns correct config."""
        assert generate_ci_matrix.get_runner_config("nightly", "GALAXY_BH") == {"label": "bh-galaxy", "type": "galaxy"}
        assert generate_ci_matrix.get_runner_config("weekly", "GALAXY_BH") == {"label": "bh-galaxy", "type": "galaxy"}

    def test_raises_error_for_unknown_device(self):
        """Test that unknown devices raise ValueError instead of falling back."""
        with pytest.raises(ValueError, match="Unknown device 'UNKNOWN_DEVICE'"):
            generate_ci_matrix.get_runner_config("nightly", "UNKNOWN_DEVICE")

    def test_returns_default_for_unknown_schedule(self):
        """Test that unknown schedules use default mapping (no override)."""
        # Should use default config for N150 when schedule is unknown
        result = generate_ci_matrix.get_runner_config("unknown_schedule", "N150")
        assert result == {"label": "n150", "type": "n150"}


class TestGetServerType:
    """Tests for get_server_type() function."""

    def test_returns_correct_server_type_for_vllm(self):
        """Test that vLLM maps to tt-inference-server."""
        assert generate_ci_matrix.get_server_type("vLLM") == "tt-inference-server"

    def test_returns_correct_server_type_for_media(self):
        """Test that MEDIA maps to media-inference-server."""
        assert generate_ci_matrix.get_server_type("MEDIA") == "media-inference-server"

    def test_returns_correct_server_type_for_forge(self):
        """Test that FORGE maps to forge-media-inference-server."""
        assert generate_ci_matrix.get_server_type("FORGE") == "forge-media-inference-server"

    def test_raises_error_for_unknown_inference_engine(self):
        """Test that unknown inference engines raise ValueError instead of falling back."""
        with pytest.raises(ValueError, match="Unknown inference engine 'UNKNOWN_ENGINE'"):
            generate_ci_matrix.get_server_type("UNKNOWN_ENGINE")


class TestRunnerMappings:
    """Tests for runner_mappings.yml schema validation."""

    def test_missing_defaults_field_fails_validation(self, monkeypatch):
        """Test that YAML without 'defaults' field fails validation."""
        invalid_config = {
            "overrides": []
        }

        with temp_config_file(invalid_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'RUNNER_MAPPINGS_FILE_PATH', temp_path)
            # Should raise - missing required 'defaults' field
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_runner_mappings()

    def test_override_missing_schedule_field_fails_validation(self, monkeypatch):
        """Test that override without 'schedule' field fails validation."""
        invalid_config = {
            "defaults": {
                "N150": {"label": "n150", "type": "n150"}
            },
            "overrides": [
                {
                    "devices": {
                        "N150": {"label": "custom-n150", "type": "n150"}
                    }
                }
            ]
        }

        with temp_config_file(invalid_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'RUNNER_MAPPINGS_FILE_PATH', temp_path)
            # Should raise - override missing required 'schedule' field
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_runner_mappings()

    def test_override_missing_devices_field_fails_validation(self, monkeypatch):
        """Test that override without 'devices' field or with empty devices fails validation."""
        invalid_config = {
            "defaults": {
                "N150": {"label": "n150", "type": "n150"}
            },
            "overrides": [
                {
                    "schedule": "nightly"
                    # Missing 'devices' field
                }
            ]
        }

        with temp_config_file(invalid_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'RUNNER_MAPPINGS_FILE_PATH', temp_path)
            # Should raise - override missing required 'devices' field
            with pytest.raises(ValueError, match="Config validation failed"):
                generate_ci_matrix.load_runner_mappings()


class TestLoadExclusions:
    """Tests for load_exclusions() function."""

    def test_returns_empty_list_when_file_does_not_exist(self, monkeypatch):
        """Test that empty list is returned when exclusions file doesn't exist."""
        monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', '/tmp/non_existent_exclusions.yaml')
        result = generate_ci_matrix.load_exclusions()
        assert result == []

    def test_loads_exclusions_from_valid_yaml(self, monkeypatch):
        """Test that exclusions are loaded correctly from valid YAML file."""
        exclusions_config = {
            'exclusions': [
                {'model': 'test-model', 'device': 'N150', 'reason': 'Test exclusion'},
                {'model': 'another-model', 'schedule': 'nightly'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            result = generate_ci_matrix.load_exclusions()
            assert len(result) == 2
            assert result[0]['model'] == 'test-model'
            assert result[0]['device'] == 'N150'
            assert result[1]['model'] == 'another-model'

    def test_raises_on_invalid_yaml(self, monkeypatch):
        """Test that invalid YAML raises an exception."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [[[')
            temp_path = f.name
        try:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            with pytest.raises(Exception):  # Should raise on invalid YAML
                generate_ci_matrix.load_exclusions()
        finally:
            os.unlink(temp_path)


class TestIsExcluded:
    """Tests for is_excluded() function."""

    def test_excludes_exact_model_match(self, monkeypatch):
        """Test that exact model name match triggers exclusion."""
        exclusions_config = {
            'exclusions': [
                {'model': 'llama2_7b', 'reason': 'Test'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True

    def test_excludes_model_substring_match(self, monkeypatch):
        """Test that model substring match triggers exclusion."""
        exclusions_config = {
            'exclusions': [
                {'model': 'llama', 'reason': 'Exclude all llama models'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('llama3_8b', 'N300', 'weekly', 'tt-inference-server', exclusions) is True

    def test_does_not_exclude_unrelated_model(self, monkeypatch):
        """Test that models with no name match are not excluded."""
        exclusions_config = {
            'exclusions': [
                {'model': 'llama', 'reason': 'Test'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('qwen_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is False

    def test_excludes_device_match(self, monkeypatch):
        """Test that device match triggers exclusion."""
        exclusions_config = {
            'exclusions': [
                {'device': 'N150', 'reason': 'N150 under maintenance'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'nightly', 'tt-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('any-model', 'n150', 'nightly', 'tt-inference-server', exclusions) is True  # Case insensitive

    def test_does_not_exclude_when_device_does_not_match(self, monkeypatch):
        """Test that non-matching device is not excluded."""
        exclusions_config = {
            'exclusions': [
                {'device': 'N150', 'reason': 'Test'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N300', 'nightly', 'tt-inference-server', exclusions) is False

    def test_excludes_schedule_match(self, monkeypatch):
        """Test that schedule match triggers exclusion."""
        exclusions_config = {
            'exclusions': [
                {'schedule': 'nightly', 'reason': 'Skip nightly runs'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'nightly', 'tt-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'NIGHTLY', 'tt-inference-server', exclusions) is True  # Case insensitive

    def test_does_not_exclude_when_schedule_does_not_match(self, monkeypatch):
        """Test that non-matching schedule is not excluded."""
        exclusions_config = {
            'exclusions': [
                {'schedule': 'nightly', 'reason': 'Test'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'weekly', 'tt-inference-server', exclusions) is False

    def test_excludes_server_type_match(self, monkeypatch):
        """Test that server_type match triggers exclusion."""
        exclusions_config = {
            'exclusions': [
                {'server_type': 'media-inference-server', 'reason': 'Media server down'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'nightly', 'media-inference-server', exclusions) is True

    def test_does_not_exclude_when_server_type_does_not_match(self, monkeypatch):
        """Test that non-matching server_type is not excluded."""
        exclusions_config = {
            'exclusions': [
                {'server_type': 'media-inference-server', 'reason': 'Test'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'nightly', 'tt-inference-server', exclusions) is False

    def test_excludes_when_all_criteria_match(self, monkeypatch):
        """Test that ALL specified criteria must match for exclusion."""
        exclusions_config = {
            'exclusions': [
                {
                    'model': 'llama2_7b',
                    'device': 'N150',
                    'schedule': 'nightly',
                    'server_type': 'tt-inference-server',
                    'reason': 'Specific combination broken'
                }
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            # Exact match - should be excluded
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True

    def test_does_not_exclude_when_one_criterion_mismatches(self, monkeypatch):
        """Test that if ANY criterion doesn't match, no exclusion occurs."""
        exclusions_config = {
            'exclusions': [
                {
                    'model': 'llama2_7b',
                    'device': 'N150',
                    'schedule': 'nightly',
                    'server_type': 'tt-inference-server'
                }
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            # Different device - should NOT be excluded
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N300', 'nightly', 'tt-inference-server', exclusions) is False
            # Different schedule - should NOT be excluded
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'weekly', 'tt-inference-server', exclusions) is False
            # Different server_type - should NOT be excluded
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'media-inference-server', exclusions) is False

    def test_excludes_with_partial_criteria(self, monkeypatch):
        """Test that rules with only some fields specified apply broadly."""
        exclusions_config = {
            'exclusions': [
                {'model': 'llama2_7b', 'device': 'N150'}  # No schedule or server_type specified
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            # Should exclude across all schedules and server types
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'weekly', 'media-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'release', 'forge-media-inference-server', exclusions) is True

    def test_applies_first_matching_rule(self, monkeypatch):
        """Test that first matching rule in order is applied."""
        exclusions_config = {
            'exclusions': [
                {'model': 'llama', 'reason': 'First rule'},
                {'model': 'llama2_7b', 'reason': 'Second rule'}  # More specific but comes second
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True

    def test_handles_empty_exclusions_list(self, monkeypatch):
        """Test that empty exclusions list excludes nothing."""
        exclusions_config = {'exclusions': []}
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            assert generate_ci_matrix.is_excluded('any-model', 'N150', 'nightly', 'tt-inference-server', exclusions) is False

    def test_skips_empty_rules(self, monkeypatch):
        """Test that None rules in the list are skipped (empty objects are rejected by schema)."""
        exclusions_config = {
            'exclusions': [
                None,
                {'model': 'llama2_7b'}
            ]
        }
        with temp_config_file(exclusions_config, format='yaml') as temp_path:
            monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', temp_path)
            exclusions = generate_ci_matrix.load_exclusions()
            # Should only match the second rule
            assert generate_ci_matrix.is_excluded('llama2_7b', 'N150', 'nightly', 'tt-inference-server', exclusions) is True
            assert generate_ci_matrix.is_excluded('other-model', 'N150', 'nightly', 'tt-inference-server', exclusions) is False


###
### "e2e" matrix generation tests
###

class TestGenerateMatrixBasics:
    """Tests for basic generate_matrix() behavior."""

    def test_skips_models_without_schedule_in_ci_config(self):
        """Test that models without the requested schedule are skipped."""
        config = {
            "models": {
                "nightly-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                },
                "weekly-only-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "weekly": {
                            "devices": ["N150"]
                        }
                        # No nightly schedule
                    }
                }
            }
        }

        with temp_config_file(config) as config_path:
            # Request nightly schedule
            result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)

            # Only nightly-model should appear
            assert 'tt-inference-server' in result
            # Find N150 device (key format is now "N150_<label>")
            n150_keys = [k for k in result['tt-inference-server'].keys() if k.startswith('N150_')]
            assert len(n150_keys) == 1
            n150_key = n150_keys[0]
            assert result['tt-inference-server'][n150_key]['models'] == ['nightly-model']
            assert 'weekly-only-model' not in result['tt-inference-server'][n150_key]['models']

    def test_models_are_sorted_by_runner_type_then_model_name_in_output(self):
        """Test that models in the flattened matrix are sorted by runner type then model name."""
        config = {
            "models": {
                "alpha-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N300"]
                        }
                    }
                },
                "zebra-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                },
                "mango-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150", "N300"]
                        }
                    }
                }
            }
        }

        with temp_config_file(config) as config_path:
            result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)
            flat = generate_ci_matrix._flatten_matrix(result)

            sort_keys = [(entry["runner"]["type"], entry["model"]) for entry in flat]
            assert sort_keys == sorted(sort_keys), f"Flat matrix is not sorted by runner type then model name: {sort_keys}"

    def test_expands_all_devices_when_devices_is_all(self):
        """Test that 'ALL' expands to all available devices."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": "ALL"
                        }
                    }
                }
            }
        }

        with temp_config_file(config) as config_path:
            result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)
            assert 'tt-inference-server' in result

            # All devices from runner_mappings.yml defaults should be present
            expected_devices = ["N150", "N300", "T3K", "GALAXY", "GALAXY_BH", "P100", "P150X4", "P150X8", "P300X2"]

            # Keys are now in format "DEVICE_label", so extract device name from each key
            device_keys = list(result['tt-inference-server'].keys())
            found_devices = set()
            for key in device_keys:
                # Extract device name (everything before first underscore + label)
                device_name = key.rsplit('_', 1)[0] if '_' in key else key
                found_devices.add(device_name)
                assert 'test-model' in result['tt-inference-server'][key]['models'], f"test-model should be in {key} models list"
                assert result['tt-inference-server'][key]['runner'] is not None, f"Runner config should be set for {key}"

            # Verify all expected devices are present
            assert found_devices == set(expected_devices), f"Expected devices {set(expected_devices)}, found {found_devices}"


class TestGenerateMatrixWithExclusions:
    """Integration tests for generate_matrix() with exclusions enabled."""

    def test_excludes_model_device_combination(self, monkeypatch):
        """Test that excluded model/device combination doesn't appear in matrix."""
        config = {
            "models": {
                "test-model-1": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150", "N300"]
                        }
                    }
                },
                "test-model-2": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
        exclusions_config = {
            'exclusions': [
                {'model': 'test-model-1', 'device': 'N150', 'reason': 'Test exclusion'}
            ]
        }

        with temp_config_file(config) as config_path:
            with temp_config_file(exclusions_config, format='yaml') as exclusions_path:
                monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', exclusions_path)
                result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)

                # test-model-1 should only appear on N300 (excluded from N150)
                assert 'tt-inference-server' in result

                # Find N150 and N300 device keys (format is "DEVICE_label")
                device_keys = list(result['tt-inference-server'].keys())
                n150_keys = [k for k in device_keys if k.startswith('N150_')]
                n300_keys = [k for k in device_keys if k.startswith('N300_')]

                assert len(n150_keys) == 1, "Should have exactly one N150 device group"
                assert len(n300_keys) == 1, "Should have exactly one N300 device group"

                # N150 should only have test-model-2 (test-model-1 excluded)
                assert result['tt-inference-server'][n150_keys[0]]['models'] == ['test-model-2']

                # N300 should only have test-model-1
                assert result['tt-inference-server'][n300_keys[0]]['models'] == ['test-model-1']

    def test_excludes_entire_device_from_all_models(self, monkeypatch):
        """Test that excluding a device removes it from all models."""
        config = {
            "models": {
                "test-model-1": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                },
                "test-model-2": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
        exclusions_config = {
            'exclusions': [
                {'device': 'N150', 'reason': 'Device maintenance'}
            ]
        }

        with temp_config_file(config) as config_path:
            with temp_config_file(exclusions_config, format='yaml') as exclusions_path:
                monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', exclusions_path)
                result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)

                # N150 should not appear in the matrix at all
                if 'tt-inference-server' in result:
                    n150_keys = [k for k in result['tt-inference-server'].keys() if k.startswith('N150_')]
                    assert len(n150_keys) == 0, "N150 should be excluded"

    def test_excludes_by_schedule(self, monkeypatch):
        """Test that exclusions can be schedule-specific."""
        config = {
            "models": {
                "test-model": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {
                            "devices": ["N150"]
                        },
                        "weekly": {
                            "devices": ["N150"]
                        }
                    }
                }
            }
        }
        exclusions_config = {
            'exclusions': [
                {'model': 'test-model', 'schedule': 'nightly', 'reason': 'Skip nightly only'}
            ]
        }

        with temp_config_file(config) as config_path:
            with temp_config_file(exclusions_config, format='yaml') as exclusions_path:
                monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', exclusions_path)

                # Nightly should be excluded
                nightly_result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)
                # Check that no N150 device keys exist
                if 'tt-inference-server' in nightly_result:
                    n150_keys = [k for k in nightly_result['tt-inference-server'].keys() if k.startswith('N150_')]
                    assert len(n150_keys) == 0, "N150 should be excluded from nightly"

                # Weekly should NOT be excluded
                weekly_result = generate_ci_matrix.generate_matrix(schedule="weekly", config_path=config_path)
                assert 'tt-inference-server' in weekly_result
                # Check that N150 device key exists
                n150_keys = [k for k in weekly_result['tt-inference-server'].keys() if k.startswith('N150_')]
                assert len(n150_keys) == 1, "N150 should be present in weekly"
                assert 'test-model' in weekly_result['tt-inference-server'][n150_keys[0]]['models']

    def test_empty_exclusion_rule_rejected_by_schema(self, monkeypatch):
        """Test that exclusion rules with only 'reason' field are rejected by schema validation."""
        config = {
            "models": {
                "test-model-1": {
                    "inference_engine": "vLLM",
                    "ci": { "nightly": {"devices": ["N150", "N300"]}}
                }
            }
        }
        exclusions_config = {
            'exclusions': [
                {'reason': 'N150 flaky'}  # No filtering fields - invalid per schema
            ]
        }

        with temp_config_file(config) as config_path:
            with temp_config_file(exclusions_config, format='yaml') as exclusions_path:
                monkeypatch.setattr(generate_ci_matrix, 'EXCLUSIONS_FILE_PATH', exclusions_path)
                # Schema validation should reject this configuration
                with pytest.raises(ValueError, match="Config validation failed"):
                    generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)


class TestGenerateMatrixWithRunnerOverrides:
    """End-to-end tests for runner overrides in generate_matrix()."""

    def test_schedule_wide_override_for_one_device(self, monkeypatch):
        """Test that schedule-wide override (no models field) applies to all models on that schedule."""
        # Runner mappings with schedule-wide override
        runner_mappings = {
            "defaults": {
                "N150": {"label": "n150", "type": "n150"},
                "N300": {"label": "n300", "type": "n300"}
            },
            "overrides": [
                {
                    "schedule": "nightly",
                    "devices": {
                        "N150": {"label": "nightly-n150", "type": "n150"}
                    }
                    # No 'models' field - applies to ALL models on nightly
                }
            ]
        }

        # Models config: model-A runs on both N150 and N300, model-B only on N150
        models_config = {
            "models": {
                "model-A": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {"devices": ["N150", "N300"]}
                    }
                },
                "model-B": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {"devices": ["N150"]}
                    }
                }
            }
        }

        with temp_config_file(runner_mappings, format='yaml') as runner_path:
            with temp_config_file(models_config) as config_path:
                monkeypatch.setattr(generate_ci_matrix, 'RUNNER_MAPPINGS_FILE_PATH', runner_path)
                result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)

                # Schedule-wide override applies to ALL models on N150
                assert 'tt-inference-server' in result
                assert 'N150_nightly-n150' in result['tt-inference-server']  # Override for N150
                assert 'N300_n300' in result['tt-inference-server']  # Default for N300 (no override)
                assert 'N150_n150' not in result['tt-inference-server']  # Default N150 label should not be present

                # Both models should be in the N150 overridden device group
                models_n150 = result['tt-inference-server']['N150_nightly-n150']['models']
                assert set(models_n150) == {'model-A', 'model-B'}

                # Only model-A should be in N300 default device group
                models_n300 = result['tt-inference-server']['N300_n300']['models']
                assert set(models_n300) == {'model-A'}

                # Verify runner configs
                runner_n150 = result['tt-inference-server']['N150_nightly-n150']['runner']
                assert runner_n150['label'] == 'nightly-n150'
                assert runner_n150['type'] == 'n150'

                runner_n300 = result['tt-inference-server']['N300_n300']['runner']
                assert runner_n300['label'] == 'n300'
                assert runner_n300['type'] == 'n300'

    def test_model_specific_override_for_two_devices(self, monkeypatch):
        """Test that model-specific override (has models field) applies only to specified models."""
        # Runner mappings with model-specific overrides
        # Note: Two separate override entries to demonstrate device-specific overrides per model
        runner_mappings = {
            "defaults": {
                "N150": {"label": "n150", "type": "n150"},
                "N300": {"label": "n300", "type": "n300"}
            },
            "overrides": [
                {
                    "schedule": "nightly",
                    "devices": {
                        "N150": {"label": "special-n150", "type": "n150"}
                    },
                    "models": ["model-A", "model-B"]  # These models get N150 override
                },
                {
                    "schedule": "nightly",
                    "devices": {
                        "N300": {"label": "special-n300", "type": "n300"}
                    },
                    "models": ["model-A"]  # Only A gets N300 override (B and C use default)
                }
            ]
        }

        models_config = {
            "models": {
                "model-A": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {"devices": ["N150", "N300"]}
                    }
                },
                "model-B": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {"devices": ["N150", "N300"]}
                    }
                },
                "model-C": {
                    "inference_engine": "vLLM",
                    "ci": {
                        "nightly": {"devices": ["N150", "N300"]}
                    }
                }
            }
        }

        with temp_config_file(runner_mappings, format='yaml') as runner_path:
            with temp_config_file(models_config) as config_path:
                monkeypatch.setattr(generate_ci_matrix, 'RUNNER_MAPPINGS_FILE_PATH', runner_path)
                result = generate_ci_matrix.generate_matrix(schedule="nightly", config_path=config_path)

                assert 'tt-inference-server' in result

                # Should have 4 device groups:
                # - N150_special-n150: models A, B (both have N150 override)
                # - N300_special-n300: model A (only A has N300 override)
                # - N150_n150: model C (no N150 override)
                # - N300_n300: models B, C (B and C don't have N300 override)
                assert 'N150_special-n150' in result['tt-inference-server']
                assert 'N300_special-n300' in result['tt-inference-server']
                assert 'N150_n150' in result['tt-inference-server']
                assert 'N300_n300' in result['tt-inference-server']

                # Verify models A and B use special runner on N150
                assert set(result['tt-inference-server']['N150_special-n150']['models']) == {'model-A', 'model-B'}

                # Verify only model A uses special runner on N300
                assert set(result['tt-inference-server']['N300_special-n300']['models']) == {'model-A'}

                # Verify runner configs for overridden devices
                assert result['tt-inference-server']['N150_special-n150']['runner']['label'] == 'special-n150'
                assert result['tt-inference-server']['N300_special-n300']['runner']['label'] == 'special-n300'

                # Verify model C uses default runner on N150
                assert set(result['tt-inference-server']['N150_n150']['models']) == {'model-C'}

                # Verify models B and C use default runner on N300
                assert set(result['tt-inference-server']['N300_n300']['models']) == {'model-B', 'model-C'}

                # Verify runner configs for default devices
                assert result['tt-inference-server']['N150_n150']['runner']['label'] == 'n150'
                assert result['tt-inference-server']['N300_n300']['runner']['label'] == 'n300'


class TestGenerateMatrixDeviceArgs:
    """Tests for device-args flowing through generate_matrix() and _flatten_matrix()."""

    def test_device_args_included_in_flat_matrix_entry(self):
        """Test that device-args for a specific device appear in the flat matrix entry for that model+device."""
        config = {
            "models": {
                "sdxl-model": {
                    "inference_engine": "MEDIA",
                    "ci": {
                        "weekly": {
                            "devices": ["T3K"],
                            "device-args": {
                                "T3K": {
                                    "additional-args": "--sdxl_num_prompts 5000",
                                    "throttle-perf": "5"
                                }
                            }
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            result = generate_ci_matrix.generate_matrix(schedule="weekly", config_path=config_path)
            flat = generate_ci_matrix._flatten_matrix(result)

            assert len(flat) == 1
            entry = flat[0]
            assert entry["model"] == "sdxl-model"
            assert entry["additional-args"] == "--sdxl_num_prompts 5000"
            assert entry["throttle-perf"] == "5"

    def test_device_args_only_apply_to_matching_device(self):
        """Test that device-args for T3K do not appear in the GALAXY flat entry for the same model."""
        config = {
            "models": {
                "sdxl-model": {
                    "inference_engine": "MEDIA",
                    "ci": {
                        "weekly": {
                            "devices": ["GALAXY", "T3K"],
                            "device-args": {
                                "T3K": {
                                    "additional-args": "--sdxl_num_prompts 5000",
                                    "throttle-perf": "5"
                                }
                            }
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            result = generate_ci_matrix.generate_matrix(schedule="weekly", config_path=config_path)
            flat = generate_ci_matrix._flatten_matrix(result)

            assert len(flat) == 2
            t3k_entry = next(e for e in flat if e["runner"]["type"] == "t3k")
            galaxy_entry = next(e for e in flat if e["runner"]["type"] == "galaxy")

            assert t3k_entry["additional-args"] == "--sdxl_num_prompts 5000"
            assert t3k_entry["throttle-perf"] == "5"
            assert "additional-args" not in galaxy_entry
            assert "throttle-perf" not in galaxy_entry

    def test_different_models_same_device_independent_args(self):
        """Test that two models on the same device can have independent device-args."""
        config = {
            "models": {
                "model-with-args": {
                    "inference_engine": "MEDIA",
                    "ci": {
                        "weekly": {
                            "devices": ["GALAXY"],
                            "device-args": {
                                "GALAXY": {
                                    "additional-args": "--extra-flag"
                                }
                            }
                        }
                    }
                },
                "model-without-args": {
                    "inference_engine": "MEDIA",
                    "ci": {
                        "weekly": {
                            "devices": ["GALAXY"]
                        }
                    }
                }
            }
        }
        with temp_config_file(config) as config_path:
            result = generate_ci_matrix.generate_matrix(schedule="weekly", config_path=config_path)
            flat = generate_ci_matrix._flatten_matrix(result)

            assert len(flat) == 2
            with_args = next(e for e in flat if e["model"] == "model-with-args")
            without_args = next(e for e in flat if e["model"] == "model-without-args")

            assert with_args["additional-args"] == "--extra-flag"
            assert "additional-args" not in without_args

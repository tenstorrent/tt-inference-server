# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import json
import logging
from typing import Optional, Union


def get_json_config_from_env(
    env_var_name: str,
    return_type: str = "dict",
    empty_result: Optional[Union[dict, str]] = None,
) -> Union[dict, str, None]:
    """Read and parse JSON configuration from environment variable.

    This is a general function that can be used for parsing JSON configurations
    from environment variables like VLLM_OVERRIDE_ARGS and OVERRIDE_TT_CONFIG.

    Args:
        env_var_name: Name of the environment variable to read from
        return_type: Expected return type - "dict" or "json_string"
        empty_result: What to return when the config is empty (None, {}, etc.)

    Returns:
        Parsed configuration as dict, JSON string, or None depending on return_type
    """
    logger = logging.getLogger(__name__)

    config_str = os.getenv(env_var_name)
    if not config_str:
        return empty_result if empty_result is not None else {}

    try:
        parsed_config = json.loads(config_str)

        # Validate that it's a dict if expected
        if return_type == "dict" and not isinstance(parsed_config, dict):
            logger.error(
                f"{env_var_name} must be a JSON object, got: {type(parsed_config)}"
            )
            return empty_result if empty_result is not None else {}

        # Handle empty configurations
        if not parsed_config:
            logger.info(f"{env_var_name}={config_str}, No overrides provided")
            return empty_result

        logger.info(f"Applying {env_var_name} configuration: {parsed_config}")

        # Return as requested type
        if return_type == "json_string":
            return json.dumps(parsed_config)
        else:
            return parsed_config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {env_var_name}: {e}")
        return empty_result if empty_result is not None else {}


def get_vllm_override_args() -> dict:
    """Read and parse vLLM override arguments from environment variable.

    Returns:
        dict: Parsed override arguments, empty dict if invalid or not set
    """
    return get_json_config_from_env(
        "VLLM_OVERRIDE_ARGS", return_type="dict", empty_result={}
    )


def get_override_tt_config() -> Optional[str]:
    """Read and parse TT config overrides from environment variable.

    Returns:
        str: JSON string of overrides, None if invalid or not set
    """
    return get_json_config_from_env(
        "OVERRIDE_TT_CONFIG", return_type="json_string", empty_result=None
    )

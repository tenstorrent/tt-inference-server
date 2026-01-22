# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
import multiprocessing
import os
import runpy
import sys
from pathlib import Path
from pprint import pprint

from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config
from utils.prompt_client import run_background_trace_capture
from utils.vllm_run_utils import (
    create_model_symlink,
    get_encoded_api_key,
    resolve_commit,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def handle_code_versions(model_spec_json):
    tt_metal_home = os.getenv("TT_METAL_HOME")
    vllm_dir = os.getenv("vllm_dir")

    tt_metal_sha = resolve_commit("HEAD", tt_metal_home)
    logger.info(f"TT_METAL_HOME: {tt_metal_home}")
    logger.info(f"commit SHA: {tt_metal_sha}")

    vllm_sha = resolve_commit("HEAD", vllm_dir)
    logger.info(f"vllm_dir: {vllm_dir}")
    logger.info(f"commit SHA: {vllm_sha}")


def _load_model_spec_json():
    """Load ModelSpec JSON from TT_MODEL_SPEC_JSON_PATH.

    Returns:
        dict: The loaded ModelSpec JSON.

    Raises:
        RuntimeError: If TT_MODEL_SPEC_JSON_PATH environment variable is not set.
        FileNotFoundError: If the specified file does not exist.
        JSONDecodeError: If the file contains invalid JSON.
    """
    model_spec_path = os.getenv("TT_MODEL_SPEC_JSON_PATH")
    if model_spec_path is None:
        raise RuntimeError("TT_MODEL_SPEC_JSON_PATH environment variable is not set")

    with open(model_spec_path, "r") as f:
        return json.load(f)


def register_tt_models(impl_id=None):
    """Register TT models with vLLM ModelRegistry.

    Args:
        impl_id: Implementation ID from ModelSpec JSON (e.g., "tt_transformers",
                 "llama3_70b_galaxy", "qwen3_32b_galaxy"). If None, defaults to
                 "tt_transformers".
    """
    impl_id = impl_id or "tt_transformers"

    # Llama path selection based on impl_id
    if impl_id == "llama3_70b_galaxy":
        path_llama_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:LlamaForCausalLM"
        )
    elif impl_id == "llama2_70b":
        path_llama_text = (
            "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        )
    else:  # default: tt_transformers
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"

    # Llama3.1/3.2 - Text
    ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)

    # Llama3.2 - Vision
    ModelRegistry.register_model(
        "TTMllamaForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration",
    )

    # Qwen2.5 - Text
    ModelRegistry.register_model(
        "TTQwen2ForCausalLM",
        "models.tt_transformers.tt.generator_vllm:QwenForCausalLM",
    )

    # Qwen3 path selection based on impl_id
    if impl_id == "qwen3_32b_galaxy":
        path_qwen3_text = (
            "models.demos.llama3_70b_galaxy.tt.generator_vllm:QwenForCausalLM"
        )
    else:
        path_qwen3_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"

    ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen3_text)

    # Mistral
    ModelRegistry.register_model(
        "TTMistralForCausalLM",
        "models.tt_transformers.tt.generator_vllm:MistralForCausalLM",
    )

    ModelRegistry.register_model(
        "TTGemma3ForConditionalGeneration",
        "models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration",
    )

    # Arcee AFM-4.5B - Text
    ModelRegistry.register_model(
        "TTArceeForCausalLM",
        "models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM",
    )

    # Note: vLLM custom model architecture registry must happen at import time, before runtime    # Qwen2.5 - Vision
    ModelRegistry.register_model(
        "TTQwen2_5_VLForConditionalGeneration",
        "models.demos.qwen25_vl.tt.generator_vllm:Qwen2_5_VLForConditionalGeneration",
    )

    # DeepseekV3
    ModelRegistry.register_model(
        "TTDeepseekV3ForCausalLM",
        "models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM",
    )


# Load model spec at import time for vLLM model registration
_MODEL_SPEC = _load_model_spec_json()
_IMPL_ID = _MODEL_SPEC.get("impl", {}).get("impl_id")
register_tt_models(_IMPL_ID)


def model_setup(model_spec_json, impl_id):
    # step 1: validate env vars passed in
    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_PATH: {os.getenv('MODEL_WEIGHTS_PATH')}")
    assert os.getenv("MODEL_WEIGHTS_PATH") is not None, "MODEL_WEIGHTS_PATH must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_PATH"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_PATH: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    dynamic_env_vars = {
        "VLLM_LOGGING_CONFIG": str(config_path),
    }
    model_env_vars = {}

    # set HF_MODEL environment variable for loading
    logging.info(f"HF model setup for {model_spec_json['hf_model_repo']}")
    model_dir_name = model_spec_json["hf_model_repo"].split("/")[-1]
    hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)
    model_env_vars["HF_MODEL"] = hf_dir
    logging.info(f"HF_MODEL: {os.getenv('HF_MODEL')}")

    if impl_id == "subdevices":
        model_env_vars["LLAMA_VERSION"] = "subdevices"
    elif impl_id == "llama2-t3000":
        model_env_vars.update(
            {
                "meta-llama/Llama-3.1-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
                "meta-llama/Llama-3.3-70B-Instruct": {
                    "LLAMA_VERSION": "llama3",
                    "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
                },
            }.get(model_spec_json["hf_model_repo"], {})
        )

    # merge dynamic and model-specific env vars
    dynamic_env_vars = {**dynamic_env_vars, **model_env_vars}

    # Set dynamic environment variables
    logger.info("setting dynamic runtime environment variables:")
    for key, value in dynamic_env_vars.items():
        if value is not None:
            logger.info(f"setting env var: {key}={value}")
            os.environ[key] = str(value)
        elif key in os.environ:
            logger.warning(
                f"removing env var: {key} from os.environ, previous value={os.environ[key]}"
            )
            del os.environ[key]


def handle_secrets(model_spec_json):
    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

    # Check if --no-auth was passed via CLI args
    no_auth = model_spec_json.get("cli_args", {}).get("no_auth", False)
    if no_auth:
        # Remove VLLM_API_KEY if present to disable authorization
        if "VLLM_API_KEY" in os.environ:
            del os.environ["VLLM_API_KEY"]
        logger.info(
            "--no-auth is set: requests to vLLM API will not require authorization. "
            "HTTP Authorization header will not be checked."
        )
        return

    # Check for VLLM_API_KEY first, then fall back to JWT_SECRET
    vllm_api_key = os.getenv("VLLM_API_KEY")
    if vllm_api_key:
        logger.info("VLLM_API_KEY is already set, using existing value")
        return

    # VLLM_API_KEY is not set, check if JWT_SECRET is available
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warning(
            "Neither VLLM_API_KEY nor JWT_SECRET are set: HTTP requests to vLLM API will not require authorization"
        )
        return

    encoded_api_key = get_encoded_api_key(jwt_secret)
    if encoded_api_key is not None:
        os.environ["VLLM_API_KEY"] = encoded_api_key
        logger.info(
            "JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token."
        )


def runtime_settings(model_spec_json, impl_id):
    logger.info(f"using model: {model_spec_json['model_id']}")
    handle_secrets(model_spec_json)

    # TODO: check HF repo access with HF_TOKEN supplied
    model_setup(model_spec_json, impl_id)


def set_runtime_env_vars(model_spec_json):
    for key, value in model_spec_json["env_vars"].items():
        if not isinstance(key, str):
            key = str(key)
            logger.warning(
                f"env var key:={key} is not a string, converting to string: {key}"
            )
        if not isinstance(value, str):
            logger.warning(
                f"env var value:={value} is not a string, converting to string: {value}"
            )
            value = str(value)

        original_value = os.getenv(key)
        if original_value is not None:
            logger.warning(
                f"env var {key} is already set to {original_value}, overriding with {value}"
            )
        logger.info(f"setting env var: {key}={value}")
        os.environ[key] = value


def start_trace_capture(model_spec_json):
    # Check if trace capture should be disabled
    disable_trace_capture = model_spec_json.get("cli_args", {}).get(
        "disable_trace_capture", False
    )

    if not disable_trace_capture:
        # Start background trace capture process
        service_port = model_spec_json.get("cli_args", {}).get(
            "service_port", int(os.getenv("SERVICE_PORT", "8000"))
        )
        supported_modalities = model_spec_json.get("supported_modalities", ["text"])

        # Get max_context from device_model_spec for trace calculation
        max_context = model_spec_json.get("device_model_spec", {}).get("max_context")
        if max_context is None:
            # Fallback to vllm_args if not in device_model_spec
            max_model_len_str = (
                model_spec_json.get("device_model_spec", {})
                .get("vllm_args", {})
                .get("max_model_len")
            )
            if max_model_len_str:
                max_context = int(max_model_len_str)

        logger.info("Starting background trace capture process...")
        trace_process = multiprocessing.Process(
            target=run_background_trace_capture,
            args=(
                model_spec_json["hf_model_repo"],
                service_port,
                supported_modalities,
                max_context,
            ),
            daemon=True,
            name="trace_capture",
        )
        trace_process.start()
        logger.info(
            f"Background trace capture process started (PID: {trace_process.pid}, "
            f"max_context: {max_context})"
        )
    else:
        logger.info("Trace capture is disabled via cli_args.disable_trace_capture")


def main():
    set_runtime_env_vars(_MODEL_SPEC)
    handle_code_versions(_MODEL_SPEC)

    runtime_settings(_MODEL_SPEC, _IMPL_ID)
    start_trace_capture(_MODEL_SPEC)

    # vLLM CLI arguments
    logger.info("vllm_args:")
    pprint(_MODEL_SPEC["device_model_spec"]["vllm_args"])
    for key, value in _MODEL_SPEC["device_model_spec"]["vllm_args"].items():
        if value is not None:
            # Handle boolean flags
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    sys.argv.append("--" + key)
            else:
                sys.argv.extend(["--" + key, str(value)])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

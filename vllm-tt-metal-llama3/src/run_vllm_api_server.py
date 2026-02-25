# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import json
import logging
import multiprocessing
import os
import runpy
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config
from utils.prompt_client import run_background_trace_capture
from utils.vllm_run_utils import (
    create_model_symlink,
    get_encoded_api_key,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse CLI arguments before any model loading.

    Also removes --model and --device from sys.argv so they don't get passed
    to vLLM's argument parser (which doesn't recognize --device).
    """
    parser = argparse.ArgumentParser(description="TT vLLM API Server")
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model repo (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--tt-device",
        type=str,
        required=True,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device type (e.g., n300, t3k, galaxy)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["vllm", "media", "forge"],
        help="Inference engine override (vllm/media/forge).",
    )
    parser.add_argument(
        "--impl",
        type=str,
        help="Implementation name override (e.g. tt-transformers).",
    )
    # Parse known args to allow vLLM args to pass through
    args, remaining = parser.parse_known_args()

    # Remove --model and --device from sys.argv so vLLM doesn't see them
    # Keep sys.argv[0] (script name) and add remaining args
    sys.argv = [sys.argv[0]] + remaining

    return args


# Device type name to mesh device string mapping
# This maps the canonical device type names (used in model specs JSON) to
# the mesh device strings used for cache paths
DEVICE_TO_MESH_STR = {
    "CPU": "CPU",
    "E150": "E150",
    "N150": "N150",
    "P100": "P100",
    "P150": "P150",
    "P150X4": "P150x4",
    "P150X8": "P150x8",
    "N150X4": "N150x4",
    "N300": "N300",
    "T3K": "T3K",
    "GALAXY": "TG",
    "GALAXY_T3K": "T3K",
    "DUAL_GALAXY": "DUAL",
    "QUAD_GALAXY": "QUAD",
    "GPU": "GPU",
}


def normalize_device_type(device_arg: str) -> str:
    """Convert user-provided device string to canonical device type name.

    Args:
        device_arg: User-provided device type (e.g., "n300", "galaxy", "T3K")

    Returns:
        Canonical device type name (e.g., "N300", "GALAXY", "T3K")
    """
    return device_arg.upper()


def normalize_engine_type(engine_arg: Optional[str]) -> Optional[str]:
    if not engine_arg:
        return None
    engine_map = {
        "vllm": "vLLM",
        "media": "media",
        "forge": "forge",
    }
    return engine_map[engine_arg.lower()]


def device_to_mesh_str(device_type: str) -> str:
    """Convert device type name to mesh device string for cache paths.

    Args:
        device_type: Canonical device type name (e.g., "N300", "GALAXY")

    Returns:
        Mesh device string (e.g., "N300", "TG")
    """
    if device_type not in DEVICE_TO_MESH_STR:
        raise ValueError(f"Unknown device type: {device_type}")
    return DEVICE_TO_MESH_STR[device_type]


def load_all_model_specs() -> dict:
    """Load all model specs from JSON file.

    Supports two modes:
    1. Legacy mode: TT_MODEL_SPEC_JSON_PATH points to a single spec (used by run_docker_server.py)
    2. New multi-spec mode: MODEL_SPECS_JSON_PATH points to all specs (for simplified docker run)

    Returns:
        dict: In legacy mode, a dict keyed by model_id. In multi-spec mode, a nested
              dict: hf_model_repo > device_type > inference_engine > impl_id > spec_dict.
    """
    # Support legacy TT_MODEL_SPEC_JSON_PATH for single-spec mode
    legacy_path = os.getenv("TT_MODEL_SPEC_JSON_PATH")
    if legacy_path and Path(legacy_path).exists():
        logger.info(
            f"Loading single model spec from TT_MODEL_SPEC_JSON_PATH: {legacy_path}"
        )
        with open(legacy_path, "r") as f:
            spec = json.load(f)
        # Return as dict keyed by model_id for compatibility
        return {spec["model_id"]: spec}

    # New multi-spec mode
    specs_path = os.getenv(
        "MODEL_SPECS_JSON_PATH",
        "/home/container_app_user/model_specs/model_spec.json",
    )
    logger.info(f"Loading all model specs from MODEL_SPECS_JSON_PATH: {specs_path}")
    with open(specs_path, "r") as f:
        return json.load(f)


def _resolve_hf_repo(all_specs: dict, model_arg: str) -> str:
    """Resolve model_arg to an hf_model_repo key in all_specs.

    Tries exact match first, then falls back to matching the short model name
    (last path segment) against all hf_model_repo keys.

    Args:
        all_specs: Nested model specs dict keyed by hf_model_repo at top level
        model_arg: The --model argument (HuggingFace repo or model name)

    Returns:
        The matching hf_model_repo key

    Raises:
        ValueError: If no matching hf_model_repo is found
    """
    if model_arg in all_specs:
        return model_arg

    short_name = model_arg.split("/")[-1]
    for hf_repo in all_specs:
        if hf_repo.split("/")[-1] == short_name:
            return hf_repo

    raise ValueError(
        f"No model spec found for model={model_arg}. "
        f"Available models: {list(all_specs.keys())[:10]}..."
    )


def find_default_impl(
    all_specs: dict,
    model_arg: str,
    device_type: str,
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Find the default implementation spec for a given model and device.

    Navigates the nested model spec structure to find the spec with
    default_impl=True for the given hf_model_repo and device_type.

    Args:
        all_specs: Nested dict: hf_model_repo > device_type > engine > impl_id > spec
        model_arg: The --model argument (HuggingFace repo or model name)
        device_type: Canonical device type name (e.g., "N300", "GALAXY")

    Returns:
        dict: The matching model spec with default_impl=True

    Raises:
        ValueError: If no matching spec is found
    """
    hf_repo = _resolve_hf_repo(all_specs, model_arg)
    device_specs = all_specs[hf_repo].get(device_type)
    if not device_specs:
        available_devices = list(all_specs[hf_repo].keys())
        raise ValueError(
            f"No model spec found for model={model_arg}, device={device_type}. "
            f"Available devices for {hf_repo}: {available_devices}"
        )

    if engine_arg:
        device_specs = {engine_arg: device_specs.get(engine_arg, {})}

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            if spec.get("device_model_spec", {}).get("default_impl"):
                return spec

    for engine_specs in device_specs.values():
        for spec in engine_specs.values():
            spec_impl_name = spec.get("impl", {}).get("impl_name")
            if impl_arg and spec_impl_name != impl_arg:
                continue
            return spec

    raise ValueError(
        f"No default_impl found for model={model_arg}, device={device_type}, "
        f"engine={engine_arg}, impl={impl_arg}. "
        f"Check that at least one impl has default_impl=True."
    )


def ensure_weights_available(model_spec: dict) -> Path:
    """Ensure model weights are available, downloading if necessary.

    If MODEL_WEIGHTS_DIR is already set (e.g. from --host-weights-dir bind mount),
    uses that directory directly and skips downloading.

    Args:
        model_spec: The model specification dictionary

    Returns:
        Path: Path to the model weights directory
    """
    # If MODEL_WEIGHTS_DIR is already set, use it directly and skip downloading
    model_weights_dir = os.getenv("MODEL_WEIGHTS_DIR")
    if model_weights_dir:
        weights_path = Path(model_weights_dir)
        if not weights_path.exists():
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} does not exist. "
                "Ensure the host directory is correctly bind-mounted."
            )
        if not any(weights_path.iterdir()):
            raise RuntimeError(
                f"MODEL_WEIGHTS_DIR={model_weights_dir} is empty. "
                "Ensure the host directory contains model weight files."
            )
        logger.info(f"Using pre-mounted weights from MODEL_WEIGHTS_DIR: {weights_path}")
        return weights_path

    # Default: download weights into cache_root
    cache_root = Path(os.getenv("CACHE_ROOT", "/home/container_app_user/cache_root"))
    model_name = model_spec["model_name"]
    weights_path = cache_root / "weights" / model_name

    if not weights_path.exists() or not any(weights_path.iterdir()):
        hf_repo = model_spec.get("hf_weights_repo") or model_spec["hf_model_repo"]
        logger.info(f"Downloading weights from {hf_repo} to {weights_path}")
        weights_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=hf_repo, local_dir=weights_path)
    else:
        logger.info(f"Weights already exist at {weights_path}")

    os.environ["MODEL_WEIGHTS_DIR"] = str(weights_path)
    return weights_path


def set_cache_paths(model_spec: dict, device_type: str):
    """Set TT_CACHE_PATH and MESH_DEVICE for model-specific cache directory.

    Args:
        model_spec: The model specification dictionary
        device_type: Canonical device type name (e.g., "N300", "GALAXY")
    """
    cache_root = Path(os.getenv("CACHE_ROOT", "/home/container_app_user/cache_root"))
    model_name = model_spec["model_name"]
    mesh_device = device_to_mesh_str(device_type)

    # Set MESH_DEVICE env var for other components that need it
    os.environ["MESH_DEVICE"] = mesh_device
    logger.info(f"Set MESH_DEVICE to {mesh_device}")

    # Preserve model/device-specific cache structure
    tt_cache_path = cache_root / "tt_metal_cache" / f"cache_{model_name}" / mesh_device
    tt_cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["TT_CACHE_PATH"] = str(tt_cache_path)
    logger.info(f"Set TT_CACHE_PATH to {tt_cache_path}")


def register_tt_models(impl_id=None):
    """Configure vLLM ModelRegistry according to ModelSpec.impl.impl_id.

    Args:
        impl_id: Implementation ID from ModelSpec JSON (e.g., "tt_transformers",
                 "llama3_70b_galaxy", "qwen3_32b_galaxy"). If None, defaults to
                 "tt_transformers".
    """
    impl_id = impl_id or "tt_transformers"

    # Llama path selection based on impl_id
    if impl_id == "llama3_70b_galaxy":
        os.environ["TT_LLAMA_TEXT_VER"] = "llama3_70b_galaxy"
    else:  # default: tt_transformers
        os.environ["TT_LLAMA_TEXT_VER"] = "tt_transformers"

    # Qwen3 env var setting based on impl_id
    if impl_id == "qwen3_32b_galaxy":
        os.environ["TT_QWEN3_TEXT_VER"] = "qwen3_32b_galaxy"
    else:
        os.environ["TT_QWEN3_TEXT_VER"] = "tt_transformers"

    # Arcee AFM-4.5B - Text
    ModelRegistry.register_model(
        "TTArceeForCausalLM",
        "models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM",
    )


def model_setup(model_spec_json):
    # step 1: validate env vars passed in
    cache_root = Path(os.getenv("CACHE_ROOT"))
    assert cache_root.exists(), f"CACHE_ROOT: {cache_root} does not exist"
    symlinks_dir = cache_root / "model_file_symlinks_map"
    symlinks_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"MODEL_WEIGHTS_DIR: {os.getenv('MODEL_WEIGHTS_DIR')}")
    assert os.getenv("MODEL_WEIGHTS_DIR") is not None, "MODEL_WEIGHTS_DIR must be set"
    weights_dir = Path(os.getenv("MODEL_WEIGHTS_DIR"))
    assert weights_dir.exists(), f"MODEL_WEIGHTS_DIR: {weights_dir} does not exist"

    logging.info(f"TT_CACHE_PATH: {os.getenv('TT_CACHE_PATH')}")
    assert os.getenv("TT_CACHE_PATH") is not None, "TT_CACHE_PATH must be set"

    # step 2: set default runtime env vars
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")

    # set HF_MODEL environment variable for loading
    logging.info(f"HF model setup for {model_spec_json['hf_model_repo']}")
    model_dir_name = model_spec_json["hf_model_repo"].split("/")[-1]
    hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)

    dynamic_env_vars = {
        "VLLM_LOGGING_CONFIG": str(config_path),
        "HF_MODEL": hf_dir,
    }

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


def runtime_settings(model_spec_json):
    logger.info(f"using model: {model_spec_json['model_id']}")
    handle_secrets(model_spec_json)

    # TODO: check HF repo access with HF_TOKEN supplied
    model_setup(model_spec_json)


def set_runtime_env_vars(model_spec_json):
    """Set runtime environment variables from model spec.

    Handles env_vars in two possible locations:
    1. Top level: model_spec_json["env_vars"] (from ModelSpec.__post_init__ merge)
    2. Nested: model_spec_json["device_model_spec"]["env_vars"] (raw JSON)

    Both locations are checked and merged, with top-level taking precedence.
    """
    env_vars = {}

    # Check nested location first (device_model_spec.env_vars)
    device_model_spec = model_spec_json.get("device_model_spec", {})
    if isinstance(device_model_spec, dict):
        nested_env_vars = device_model_spec.get("env_vars", {})
        if nested_env_vars:
            env_vars.update(nested_env_vars)

    # Check top-level location (takes precedence)
    top_level_env_vars = model_spec_json.get("env_vars", {})
    if top_level_env_vars:
        env_vars.update(top_level_env_vars)

    if not env_vars:
        logger.info("No env_vars found in model spec")
        return

    for key, value in env_vars.items():
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

    if disable_trace_capture:
        logger.info("Trace capture is disabled via cli_args.disable_trace_capture")
        return

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


def main():
    # Step 1: Parse --model argument (if provided)
    args = parse_args()
    args.device = args.tt_device or args.device
    args.engine = normalize_engine_type(args.engine)

    # Step 2: Load model specs and determine which spec to use
    all_specs = load_all_model_specs()

    # Check if we're in legacy single-spec mode (TT_MODEL_SPEC_JSON_PATH set)
    legacy_path = os.getenv("TT_MODEL_SPEC_JSON_PATH")

    if legacy_path and Path(legacy_path).exists():
        logger.warning(
            "Deprecation warning: setting TT_MODEL_SPEC_JSON_PATH is deprecated. "
            f"Using model spec from TT_MODEL_SPEC_JSON_PATH: {legacy_path}"
        )
        # Legacy mode: use the single spec from TT_MODEL_SPEC_JSON_PATH
        # (all_specs will have exactly one entry in this case)
        model_spec = list(all_specs.values())[0]
        logger.info("Legacy mode: using model spec from TT_MODEL_SPEC_JSON_PATH")

        # Ensure cache paths and weights are available, setting env vars if
        # they were not passed in (e.g. default Docker volume mode).
        if not os.getenv("TT_CACHE_PATH") and args.device:
            set_cache_paths(model_spec, normalize_device_type(args.device))
        if not os.getenv("MODEL_WEIGHTS_DIR"):
            ensure_weights_available(model_spec)
    elif args.model and args.device:
        # New interface: look up spec by --model and explicit --tt-device/--device
        device_type = normalize_device_type(args.device)
        model_spec = find_default_impl(
            all_specs,
            args.model,
            device_type,
            engine_arg=args.engine,
            impl_arg=args.impl,
        )
        logger.info(
            f"Using default interface: found model spec for --model={args.model}, "
            f"--device={device_type}, --engine={args.engine}, --impl={args.impl}"
        )

        # Set cache paths and ensure weights are available (new interface only)
        set_cache_paths(model_spec, device_type)
        ensure_weights_available(model_spec)
    else:
        raise RuntimeError(
            "Either set TT_MODEL_SPEC_JSON_PATH env var (for 'python run.py --docker-server' "
            "workflow) or provide --model and --tt-device/--device for direct "
            "docker run. Example: docker run <image> --model meta-llama/Llama-3.1-8B --tt-device n300"
        )

    logger.info(f"Using model spec: {model_spec['model_id']}")

    # Step 3: Register TT models (after lookup, with correct impl_id)
    impl_id = model_spec.get("impl", {}).get("impl_id")
    register_tt_models(impl_id)

    # Step 4: Set runtime environment variables and run setup
    set_runtime_env_vars(model_spec)
    runtime_settings(model_spec)
    start_trace_capture(model_spec)

    # Step 5: Add vLLM CLI arguments
    logger.info(
        f"vllm_args: {json.dumps(model_spec['device_model_spec']['vllm_args'], indent=4)}"
    )
    for key, value in model_spec["device_model_spec"]["vllm_args"].items():
        if value is not None:
            # Handle boolean flags
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    sys.argv.append("--" + key)
            else:
                sys.argv.extend(["--" + key, str(value)])

    # Step 6: Launch vLLM server
    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

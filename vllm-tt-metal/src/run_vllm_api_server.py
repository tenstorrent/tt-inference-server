# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import json
import logging
import multiprocessing
import os
import runpy
import shlex
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from vllm import ModelRegistry

from utils.cache_monitor import get_container_cache_dir
from utils.device_utils import get_mesh_device_name
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


DEFAULT_VLLM_SERVER_PORT = "8000"


def parse_args():
    """Parse wrapper CLI args and return remaining vLLM passthrough args."""
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
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable vLLM API key authorization (skips JWT_SECRET requirement)",
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disable automatic trace capture requests on server startup",
    )
    parser.add_argument(
        "--service-port",
        type=int,
        default=None,
        help="Service port for vLLM server and trace capture client",
    )
    # Parse known args to allow vLLM args to pass through
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


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


def unwrap_model_specs_catalog(model_specs: dict) -> dict:
    """Return the nested model specs catalog from wrapped or legacy JSON."""
    if "model_specs" in model_specs and isinstance(model_specs["model_specs"], dict):
        return model_specs["model_specs"]
    return model_specs


def load_model_spec(
    model_arg: Optional[str],
    device_arg: Optional[str],
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Load and resolve a single model spec.

    Resolution order:
    1. Runtime mode: RUNTIME_MODEL_SPEC_JSON_PATH points to a pre-resolved spec
       (produced by run.py --docker-server)
    2. Catalog mode: MODEL_SPECS_JSON_PATH + --model/--tt-device/--device (+ optional
       --engine/--impl) are used to resolve one spec from the built-in catalog.

    Returns:
        dict: The resolved single model spec.

    Raises:
        RuntimeError: If runtime path is not available and required CLI args are missing.
    """
    runtime_path = os.getenv("RUNTIME_MODEL_SPEC_JSON_PATH")
    if runtime_path:
        runtime_path = Path(runtime_path)
        if runtime_path.exists():
            logger.info(
                "Using pre-resolved runtime model spec from "
                f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path}"
            )
            logger.info(f"Loading runtime model spec from: {runtime_path}")
            with open(runtime_path, "r") as f:
                data = json.load(f)
            return data.get("runtime_model_spec", data)
        logger.warning(
            f"RUNTIME_MODEL_SPEC_JSON_PATH={runtime_path} does not exist, "
            "falling back to default model spec catalog."
        )

    if not model_arg or not device_arg:
        raise RuntimeError(
            "Either set RUNTIME_MODEL_SPEC_JSON_PATH env var "
            "(for 'python run.py --docker-server' workflow), or provide --model and "
            "--tt-device/--device for direct docker run. "
            "Example: docker run <image> --model meta-llama/Llama-3.1-8B --tt-device n300"
        )

    # Catalog mode (model_spec.json built into image)
    specs_path = os.getenv(
        "MODEL_SPECS_JSON_PATH",
        "/home/container_app_user/model_specs/model_spec.json",
    )
    logger.info(f"Loading all model specs from MODEL_SPECS_JSON_PATH: {specs_path}")
    with open(specs_path, "r") as f:
        model_specs = unwrap_model_specs_catalog(json.load(f))

    device_type = normalize_device_type(device_arg)
    model_spec = find_default_impl(
        model_specs,
        model_arg,
        device_type,
        engine_arg=engine_arg,
        impl_arg=impl_arg,
    )
    logger.info(
        f"Using default interface: found model spec for --model={model_arg}, "
        f"--device={device_type}, --engine={engine_arg}, --impl={impl_arg}"
    )
    return model_spec


def _resolve_hf_repo(model_specs: dict, model_arg: str) -> str:
    """Resolve model_arg to an hf_model_repo key in model_specs.

    Tries exact match first, then falls back to matching the short model name
    (last path segment) against all hf_model_repo keys.

    Args:
        model_specs: Nested model specs dict keyed by hf_model_repo at top level
        model_arg: The --model argument (HuggingFace repo or model name)

    Returns:
        The matching hf_model_repo key

    Raises:
        ValueError: If no matching hf_model_repo is found
    """
    if model_arg in model_specs:
        return model_arg

    short_name = model_arg.split("/")[-1]
    for hf_repo in model_specs:
        if hf_repo.split("/")[-1] == short_name:
            return hf_repo

    raise ValueError(
        f"No model spec found for model={model_arg}. "
        f"Available models: {list(model_specs.keys())[:10]}..."
    )


def find_default_impl(
    model_specs: dict,
    model_arg: str,
    device_type: str,
    engine_arg: Optional[str] = None,
    impl_arg: Optional[str] = None,
) -> dict:
    """Find the default implementation spec for a given model and device.

    Navigates the nested model spec structure to find the spec with
    default_impl=True for the given hf_model_repo and device_type.

    Args:
        model_specs: Nested dict: hf_model_repo > device_type > engine > impl_id > spec
        model_arg: The --model argument (HuggingFace repo or model name)
        device_type: Canonical device type name (e.g., "N300", "GALAXY")

    Returns:
        dict: The matching model spec with default_impl=True

    Raises:
        ValueError: If no matching spec is found
    """
    hf_repo = _resolve_hf_repo(model_specs, model_arg)
    device_specs = model_specs[hf_repo].get(device_type)
    if not device_specs:
        available_devices = list(model_specs[hf_repo].keys())
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
    mesh_device = get_mesh_device_name(device=device_type)
    tt_cache_path = get_container_cache_dir(model_spec, device=device_type)
    if tt_cache_path is None:
        raise RuntimeError("Could not resolve TT cache path from model spec.")

    # Set MESH_DEVICE env var for other components that need it
    os.environ["MESH_DEVICE"] = mesh_device
    logger.info(f"Set MESH_DEVICE to {mesh_device}")

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


def handle_secrets(no_auth=False):
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN is set")
    else:
        logger.warning(
            "HF_TOKEN is not set - this may cause issues accessing private models or models requiring authorization"
        )

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


def runtime_settings(model_spec_json, no_auth=False):
    logger.info(f"using model: {model_spec_json['model_id']}")
    handle_secrets(no_auth=no_auth)

    # In multihost deployments, model weights are on shared storage and accessed
    # via model-specific environment variables (e.g., DEEPSEEK_V3_HF_MODEL).
    # Skip model_setup() which requires MODEL_WEIGHTS_DIR and creates symlinks.
    # TODO(tt-metal): Update DeepSeek model impl to use standard HF_MODEL env var
    # so we can reuse existing model setup and standard weight/cache mounting.
    if os.getenv("MULTIHOST_ROLE"):
        logger.info(
            "Multihost mode detected, skipping model_setup() - "
            "weights accessed via model-specific env vars on shared storage"
        )
        return

    # TODO: check HF repo access with HF_TOKEN supplied
    model_setup(model_spec_json)


def set_metal_timeout_env_vars():
    """Set tt-metal operation timeout env vars for automatic hang detection.

    When enabled (default), configures TT_METAL_OPERATION_TIMEOUT_SECONDS and
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE so that tt-triage runs
    automatically when an op dispatch hangs.

    Disabled when DISABLE_METAL_OP_TIMEOUT=1 is set (via run.py --disable-metal-timeout).
    """
    if os.getenv("DISABLE_METAL_OP_TIMEOUT") == "1":
        logger.info("Metal op timeout disabled via DISABLE_METAL_OP_TIMEOUT=1")
        return

    tt_metal_home = os.getenv("TT_METAL_HOME", "/home/container_app_user/tt-metal")
    python_env_dir = os.getenv("PYTHON_ENV_DIR", f"{tt_metal_home}/python_env")
    log_dir = os.getenv("TT_METAL_LOGS_PATH", "/home/container_app_user/logs")

    triage_new = Path(tt_metal_home) / "tools" / "triage" / "triage.py"
    triage_old = Path(tt_metal_home) / "scripts" / "debugging_scripts" / "triage.py"
    triage_script = str(triage_new if triage_new.exists() else triage_old)

    timeout_cmd = (
        f"{python_env_dir}/bin/python {triage_script} "
        f"--disable-progress > {log_dir}/tt-triage-$(date +%Y%m%d-%H%M%S).log 2>&1"
    )

    os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "5.0"
    os.environ["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"] = timeout_cmd
    logger.info("Set TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0")
    logger.info(f"Set TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE={timeout_cmd}")


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


def start_trace_capture(
    model_spec_json, service_port: int, disable_trace_capture: bool = False
):
    # Models with builtin warmup handle their own trace capture internally
    if not disable_trace_capture and model_spec_json.get("has_builtin_warmup", False):
        disable_trace_capture = True
        logger.info(
            "Model has builtin warmup (has_builtin_warmup=True), "
            "skipping background trace capture"
        )

    if disable_trace_capture:
        logger.info("Trace capture is disabled via --disable-trace-capture")
        return

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


def _normalize_vllm_arg_name(arg_name: str) -> str:
    return arg_name.lstrip("-").split("=", 1)[0].replace("-", "_")


def _append_vllm_arg(argv: list[str], arg_name: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(arg_name)
        return
    argv.extend([arg_name, str(value)])


def _extract_cli_arg_value(argv: list[str], arg_name: str) -> Optional[str]:
    for index, token in enumerate(argv):
        if token == arg_name:
            if index + 1 < len(argv):
                return argv[index + 1]
            return None
        if token.startswith(f"{arg_name}="):
            return token.split("=", 1)[1]
    return None


def resolve_service_port() -> int:
    port_value = _extract_cli_arg_value(sys.argv[1:], "--port")
    if port_value is not None:
        return int(port_value)
    return int(DEFAULT_VLLM_SERVER_PORT)


def format_vllm_serve_command(argv) -> str:
    """Render the normalized argv as a multi-line bash command."""
    command_lines = ["vllm serve"]
    index = 1
    while index < len(argv):
        token = argv[index]
        rendered_tokens = [shlex.quote(token)]
        has_separate_value = (
            token.startswith("--")
            and "=" not in token
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("--")
        )
        if has_separate_value:
            rendered_tokens.append(shlex.quote(argv[index + 1]))
            index += 1

        command_lines.append(" ".join(rendered_tokens))
        index += 1

    return " \\\n  ".join(command_lines)


def set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args):
    # runpy uses sys.argv, rebuild it with the merged vLLM args.
    vllm_argv = [sys.argv[0]]
    remaining_default_vllm_args = dict(default_vllm_args)
    default_arg_name_by_normalized_name = {
        _normalize_vllm_arg_name(arg_name): arg_name
        for arg_name in remaining_default_vllm_args
    }
    input_vllm_argv = list(remaining_sys_argv)
    if args.service_port is not None:
        already_set_port = _extract_cli_arg_value(input_vllm_argv, "--port")
        if already_set_port is not None:
            logger.warning(
                f"vLLM server --port={already_set_port} already set direcly, ignoring --service-port={args.service_port}"
            )
        else:
            # Remap wrapper --service-port to vLLM's --port.
            input_vllm_argv.extend(["--port", str(args.service_port)])

    index = 0
    while index < len(input_vllm_argv):
        token = input_vllm_argv[index]
        if not token.startswith("--"):
            vllm_argv.append(token)
            index += 1
            continue

        cli_arg_name, separator, inline_value = token.partition("=")
        overridden_default_arg_name = default_arg_name_by_normalized_name.pop(
            _normalize_vllm_arg_name(cli_arg_name), None
        )
        if overridden_default_arg_name is not None:
            remaining_default_vllm_args.pop(overridden_default_arg_name, None)

        if separator:
            vllm_argv.append(f"{cli_arg_name}={inline_value}")
            index += 1
            continue

        vllm_argv.append(cli_arg_name)
        next_token_is_value = index + 1 < len(input_vllm_argv) and not input_vllm_argv[
            index + 1
        ].startswith("--")
        if next_token_is_value:
            value = input_vllm_argv[index + 1]
            vllm_argv.append(value)
            index += 2
            continue

        index += 1

    for key, value in remaining_default_vllm_args.items():
        cli_arg_name = f"--{key}"
        _append_vllm_arg(vllm_argv, cli_arg_name, value)

    # finally set sys.argv to the vllm server args
    sys.argv = vllm_argv
    logger.info(f"vLLM command:\n{format_vllm_serve_command(sys.argv)}")


def main():
    # Step 1: Parse --model argument (if provided)
    args, remaining_sys_argv = parse_args()
    args.device = args.tt_device or args.device
    args.engine = normalize_engine_type(args.engine)

    # Step 2: Load model spec
    model_spec = load_model_spec(
        model_arg=args.model,
        device_arg=args.device,
        engine_arg=args.engine,
        impl_arg=args.impl,
    )
    device_type = model_spec.get("device_type")
    if device_type:
        device_type = normalize_device_type(device_type)
    elif args.device:
        device_type = normalize_device_type(args.device)

    if device_type and not os.getenv("TT_CACHE_PATH"):
        set_cache_paths(model_spec, device_type)
    # NOTE: In multihost deployments, model weights are expected to reside on shared
    # storage (e.g., NFS) and are read directly by each worker via model-specific
    # environment variables (e.g., DEEPSEEK_V3_HF_MODEL). Users are responsible for
    # downloading weights to a location on shared storage beforehand. Therefore,
    # automatic weight download is skipped when MULTIHOST_ROLE is set.
    if not os.getenv("MODEL_WEIGHTS_DIR") and not os.getenv("MULTIHOST_ROLE"):
        ensure_weights_available(model_spec)

    logger.info(f"Using model spec: {model_spec['model_id']}")

    # Step 3: Register TT models (after lookup, with correct impl_id)
    impl_id = model_spec.get("impl", {}).get("impl_id")
    register_tt_models(impl_id)

    # Step 4: Set runtime environment variables and vLLM server args
    set_metal_timeout_env_vars()
    set_runtime_env_vars(model_spec)
    runtime_settings(model_spec, no_auth=args.no_auth)
    default_vllm_args = model_spec["device_model_spec"]["vllm_args"]
    set_vllm_sys_argv(args, remaining_sys_argv, default_vllm_args)

    # Step 5: Start trace capture if needed
    start_trace_capture(
        model_spec,
        service_port=resolve_service_port(),
        disable_trace_capture=args.disable_trace_capture,
    )

    # Step 6: Launch vLLM server
    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

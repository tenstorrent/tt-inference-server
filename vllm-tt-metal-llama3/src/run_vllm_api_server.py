# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import runpy
import json
import subprocess
import logging
from pathlib import Path

import jwt
from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_hf_model_id():
    model = os.getenv("HF_MODEL_REPO_ID")
    if not model:
        logger.Error("Must set environment variable: HF_MODEL_REPO_ID")
        sys.exit()
    return model


def resolve_commit(commit: str, repo_path: Path) -> str:
    repo_path = Path(repo_path)
    assert repo_path.is_dir(), f"The path '{repo_path}' is not a valid directory."
    result = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stdout.strip() not in ["commit", "tag"]:
        logger.warning(f"Commit '{commit}' is not in the repository.")
        return None

    result = subprocess.run(
        ["git", "rev-parse", commit],
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def is_head_eq_or_after_commit(commit: str, repo_path: str = ".") -> bool:
    """
    Checks if the current HEAD of repo_path is the same as or a descendant (i.e., comes after) commit in the git history.

    Args:
        commit (str): The SHA or tag of the commit to compare with HEAD.
        repo_path (str): The path to the git repository (default is current directory).

    Returns:
        bool: True if commit is the same or newer than HEAD, False otherwise.
    """
    repo = Path(repo_path)
    assert repo.is_dir(), f"The path '{repo}' is not a valid directory."

    try:
        # Resolve full commit hashes in case they are tags or shortened SHAs
        commit_full = resolve_commit(commit, repo)
        head_commit = resolve_commit("HEAD", repo)

        if not commit_full:
            logger.warning(
                f"Commit '{commit}' was not resolved. Assuming it is an external or future commit."
            )
            return False  # If the commit is unknown, assume it is newer than HEAD

        # If the commits are the same, return True
        if commit_full == head_commit:
            return True

        # Run the git command to check ancestry
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_full, head_commit],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # A return code of 0 means commit is an ancestor of HEAD
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Unexpected error while checking commit ancestry: {e}")
        return False


def handle_code_versions():
    tt_metal_home = os.getenv("TT_METAL_HOME")
    vllm_dir = os.getenv("vllm_dir")

    tt_metal_sha = resolve_commit("HEAD", tt_metal_home)
    logger.info(f"TT_METAL_HOME: {tt_metal_home} commit SHA: {tt_metal_sha}")

    vllm_sha = resolve_commit("HEAD", vllm_dir)
    logger.info(f"vllm_dir: {vllm_dir} commit SHA: {vllm_sha}")

    metal_tt_transformers_commit = "8815f46aa191d0b769ed1cc1eeb59649e9c77819"
    if os.getenv("MODEL_IMPL") == "tt-transformers":
        assert is_head_eq_or_after_commit(
            commit=metal_tt_transformers_commit, repo_path=tt_metal_home
        ), "tt-transformers model_impl requires tt-metal: v0.57.0-rc1 or later"


# Copied from vllm/examples/offline_inference_tt.py
def register_tt_models():
    model_impl = os.getenv("MODEL_IMPL", "tt-transformers")
    if model_impl == "tt-transformers":
        from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM
        from models.tt_transformers.tt.generator_vllm import (
            MllamaForConditionalGeneration,
        )
        from models.tt_transformers.tt.generator_vllm import Qwen2ForCausalLM

        ModelRegistry.register_model("TTQwen2ForCausalLM", Qwen2ForCausalLM)
        ModelRegistry.register_model(
            "TTMllamaForConditionalGeneration", MllamaForConditionalGeneration
        )
    elif model_impl == "subdevices":
        from models.demos.llama3_subdevices.tt.generator_vllm import LlamaForCausalLM
    elif model_impl == "t3000-llama2-70b":
        from models.demos.t3000.llama2_70b.tt.generator_vllm import (
            TtLlamaForCausalLM as LlamaForCausalLM,
        )
    else:
        raise ValueError(
            f"Unsupported model_impl: {model_impl}, pick one of [tt-transformers, subdevices, llama2-t3000]"
        )

    ModelRegistry.register_model("TTLlamaForCausalLM", LlamaForCausalLM)


register_tt_models()  # Import and register models from tt-metal


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def ensure_mesh_device(hf_model_id):
    # model specific MESH_DEVICE management
    default_mesh_device = {
        "Qwen/QwQ-32B": "T3K",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "T3K",
        "Qwen/Qwen2.5-72B-Instruct": "T3K",
        "Qwen/Qwen2.5-7B-Instruct": "N300",
        "meta-llama/Llama-3.1-70B-Instruct": "T3K",
        "meta-llama/Llama-3.3-70B-Instruct": "T3K",
        "meta-llama/Llama-3.2-1B-Instruct": "N150",
        "meta-llama/Llama-3.2-3B-Instruct": "N150",
        "meta-llama/Llama-3.1-8B-Instruct": "N300",
        "meta-llama/Llama-3.2-11B-Vision-Instruct": "N300",
    }
    valid_mesh_devices = {
        # only T3K_RING available for this 70B model implementation
        # see: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/t3000/llama2_70b/tt/generator_vllm.py#L47
        # TG implementation will be impl in: https://github.com/tenstorrent/tt-metal/blob/main/models/demos/llama3/tt/generator_vllm.py#L136
        "meta-llama/Llama-3.1-70B-Instruct": ["T3K", "TG"],
        "meta-llama/Llama-3.3-70B-Instruct": ["T3K", "TG"],
        "Qwen/QwQ-32B": ["T3K"],
        "Qwen/Qwen2.5-72B-Instruct": ["T3K"],
        "Qwen/Qwen2.5-7B-Instruct": ["N300", "T3K"],
        "meta-llama/Llama-3.2-11B-Vision-Instruct": [
            "N300",
            "T3K",
        ],
    }
    cur_mesh_device = os.getenv("MESH_DEVICE")
    if hf_model_id in default_mesh_device.keys():
        if cur_mesh_device is None:
            # set good default
            os.environ["MESH_DEVICE"] = default_mesh_device[hf_model_id]
            cur_mesh_device = os.getenv("MESH_DEVICE")

    if hf_model_id in valid_mesh_devices.keys():
        assert (
            cur_mesh_device in valid_mesh_devices[hf_model_id]
        ), f"Invalid MESH_DEVICE for {hf_model_id}"

    logger.info(f"using MESH_DEVICE:={os.getenv('MESH_DEVICE')}")


def create_model_symlink(symlinks_dir, model_name, weights_dir, file_symlinks_map={}):
    """Helper function to create and manage model symlinks.

    Args:
        symlinks_dir: Directory to store symlinks
        model_name: Model name to use for the symlink
        weights_dir: Path to the model weights
        file_symlinks_map: Dict of {target_file: source_file} for creating file-specific symlinks

    Returns:
        Path to the created symlink or directory
    """
    symlink_path = symlinks_dir / model_name

    # Handle file-specific symlinks (for vision models)
    if file_symlinks_map:
        # Clean up any existing symlinks
        if symlink_path.exists():
            for _link in symlink_path.iterdir():
                if _link.is_symlink():
                    _link.unlink()
        symlink_path.mkdir(parents=True, exist_ok=True)

        # Create individual file symlinks
        for target_file, source_file in file_symlinks_map.items():
            (symlink_path / target_file).symlink_to(weights_dir / source_file)

        return symlink_path

    # Handle single directory/file symlink (standard case)
    if symlink_path.is_symlink():
        symlink_path.unlink()
    assert (
        not symlink_path.exists()
    ), f"symlink location: {symlink_path} has a non-symlink there."
    symlink_path.symlink_to(weights_dir)
    return symlink_path


def runtime_settings(hf_model_id):
    # step 1: validate env vars passed in
    ensure_mesh_device(hf_model_id)
    model_impl = os.getenv("MODEL_IMPL")
    logger.info(f"MODEL_IMPL:={model_impl}")
    logging.info(f"MODEL_SOURCE: {os.getenv('MODEL_SOURCE')}")

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

    env_vars = {
        # note: the vLLM logging environment variables do not cause the configuration
        # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
        "VLLM_CONFIGURE_LOGGING": "1",
        "VLLM_LOGGING_CONFIG": str(config_path),
        # stop timeout during long sequential prefill batches
        # e.g. 32x 2048 token prefills taking longer than default 30s timeout
        # timeout is 3x VLLM_RPC_TIMEOUT
        "VLLM_RPC_TIMEOUT": "900000",  # 200000ms = 200s
    }
    # note: do not set this post v0.56.0-rc47
    # env_vars["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1",

    if os.getenv("MESH_DEVICE") in ["N150", "N300", "T3K"]:
        env_vars["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    else:
        # remove WH_ARCH_YAML if it was set
        env_vars["WH_ARCH_YAML"] = None

    if hf_model_id.startswith("meta-llama"):
        logging.info(f"Llama setup for {hf_model_id}")

        model_dir_name = hf_model_id.split("/")[-1]
        # the mapping in: models/tt_transformers/tt/model_config.py
        # uses e.g. Llama3.2 instead of Llama-3.2
        model_dir_name = model_dir_name.replace("Llama-", "Llama")
        file_symlinks_map = {}
        if hf_model_id.startswith("meta-llama/Llama-3.2-11B-Vision"):
            # Llama-3.2-11B-Vision requires specific file symlinks with different names
            # The loading code in:
            # https://github.com/tenstorrent/tt-metal/blob/v0.57.0-rc71/models/tt_transformers/demo/simple_vision_demo.py#L55
            # does not handle this difference in naming convention for the weights
            file_symlinks_map = {
                "consolidated.00.pth": "consolidated.pth",
                "params.json": "params.json",
                "tokenizer.model": "tokenizer.model",
            }
        elif model_dir_name.startswith("Llama3.3"):
            # Only Llama 3.1 70B is defined in models/tt_transformers/tt/model_config.py
            env_vars["MAX_PREFILL_CHUNK_SIZE"] = "32"

        llama_dir = create_model_symlink(
            symlinks_dir,
            model_dir_name,
            weights_dir,
            file_symlinks_map=file_symlinks_map,
        )

        env_vars["LLAMA_DIR"] = str(llama_dir)
        env_vars.update({"HF_MODEL": None})
    else:
        logging.info(f"HF model setup for {hf_model_id}")
        model_dir_name = hf_model_id.split("/")[-1]
        hf_dir = create_model_symlink(symlinks_dir, model_dir_name, weights_dir)
        env_vars["HF_MODEL"] = hf_dir
        env_vars.update({"LLAMA_DIR": None})

    if model_impl == "tt-transformers":
        env_var_map = {
            "meta-llama/Llama-3.1-70B-Instruct": {},
            "meta-llama/Llama-3.3-70B-Instruct": {},
            "Qwen/QwQ-32B": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            },
            "Qwen/Qwen2.5-72B-Instruct": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            },
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            },
        }
    elif model_impl == "subdevices":
        env_vars["LLAMA_VERSION"] = "subdevices"
    elif model_impl == "llama2-t3000":
        env_var_map = {
            "meta-llama/Llama-3.1-70B-Instruct": {
                "LLAMA_VERSION": "llama3",
                "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
            },
            "meta-llama/Llama-3.3-70B-Instruct": {
                "LLAMA_VERSION": "llama3",
                "LLAMA_DIR": os.getenv("MODEL_WEIGHTS_PATH"),
            },
        }
    env_vars.update(env_var_map.get(hf_model_id, {}))
    # Set each environment variable
    logger.info("setting runtime environment variables:")
    for key, value in env_vars.items():
        logger.info(f"setting env var: {key}={value}")
        if value is not None:
            os.environ[key] = str(value)
        elif key in os.environ:
            del os.environ[key]


def vllm_override_tt_config(hf_model_id):
    override_tt_config = {}
    # Dispatch core axis is row on wormhole and col on blackhole (by default), but if it's Llama3.x-70B on TG then we force it to col.
    if (
        hf_model_id
        in ["meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]
        and os.getenv("MESH_DEVICE") == "TG"
    ):
        override_tt_config["dispatch_core_axis"] = "col"
        override_tt_config["sample_on_device_mode"] = "all"
        override_tt_config["fabric_config"] = "FABRIC_1D"
        override_tt_config["worker_l1_size"] = 1344544
        override_tt_config["trace_region_size"] = 62000000

    return json.dumps(override_tt_config) if override_tt_config else None


def model_setup(hf_model_id):
    # TODO: check HF repo access with HF_TOKEN supplied
    logger.info(f"using model: {hf_model_id}")
    runtime_settings(hf_model_id)
    args = {
        "model": hf_model_id,
        "block_size": os.getenv("VLLM_BLOCK_SIZE", "64"),
        "max_num_seqs": os.getenv("VLLM_MAX_NUM_SEQS", "32"),
        "max_model_len": os.getenv("VLLM_MAX_MODEL_LEN", "131072"),
        "max_num_batched_tokens": os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "131072"),
        "num_scheduler_steps": os.getenv("VLLM_NUM_SCHEDULER_STEPS", "10"),
        "max-log-len": os.getenv("VLLM_MAX_LOG_LEN", "64"),
        "port": os.getenv("SERVICE_PORT", "7000"),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
    }

    if os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true":
        args["enable-auto-tool-choice"] = None
        args["tool-call-parser"] = os.getenv("TOOL_CALL_PARSER", None)

    override_tt_config = vllm_override_tt_config(hf_model_id)
    if override_tt_config:
        args["override_tt_config"] = override_tt_config

    return args


def main():
    hf_model_id = get_hf_model_id()
    handle_code_versions()
    # vLLM CLI arguments
    args = model_setup(hf_model_id)
    for key, value in args.items():
        if value is not None:
            sys.argv.extend(["--" + key, value])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

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
    model = os.environ.get("HF_MODEL_REPO_ID")
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
    tt_metal_home = os.environ.get("TT_METAL_HOME")
    vllm_dir = os.environ.get("vllm_dir")

    tt_metal_sha = resolve_commit("HEAD", tt_metal_home)
    logger.info(f"TT_METAL_HOME: {tt_metal_home} commit SHA: {tt_metal_sha}")

    vllm_sha = resolve_commit("HEAD", vllm_dir)
    logger.info(f"vllm_dir: {vllm_dir} commit SHA: {vllm_sha}")

    metal_tt_transformers_commit = "8815f46aa191d0b769ed1cc1eeb59649e9c77819"
    metal_ckpt_dir_commit = "ce8bbbadd52d505cd420ed879d9599d8282210ee"
    metal_earliest_supported = "v0.56.0-rc6"

    if is_head_eq_or_after_commit(
        commit=metal_ckpt_dir_commit, repo_path=tt_metal_home
    ):
        # tt-metal model_config.py::TtModelArgs.model_name is defined by CKPT_DIR
        # see https://github.com/tenstorrent/tt-metal/blob/v0.56.0-rc47/models/demos/llama3/tt/model_config.py#L130C13-L130C28
        # needs to match MAX_PREFILL_CHUNK_SIZES_DIV1024 dict format without first dash
        ckpt_dir = os.getenv("CKPT_DIR")
        req_llama_path = Path(ckpt_dir.replace("/Llama-", "/Llama"))
        if not req_llama_path.exists():
            req_llama_path.symlink_to(ckpt_dir, target_is_directory=True)
        os.environ["CKPT_DIR"] = str(req_llama_path)


def register_vllm_models():
    # Import and register models from tt-metal, must run at import time
    # route between different TT model implementations
    legacy_impl_models = [
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    hf_model_id = get_hf_model_id()
    if hf_model_id in legacy_impl_models:
        from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM

        ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
    else:
        from models.demos.llama3.tt.generator_vllm import (
            TtMllamaForConditionalGeneration,
            TtLlamaForCausalLM,
        )

        ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
        # for multimodel vision model
        ModelRegistry.register_model(
            "TTMllamaForConditionalGeneration", TtMllamaForConditionalGeneration
        )

    from models.demos.llama3.tt.generator_vllm import TtQwen2ForCausalLM

    ModelRegistry.register_model("TTQwen2ForCausalLM", TtQwen2ForCausalLM)


# note: register_vllm_models() must run at import time
# otherwise vLLM will exit with:
#   'ValueError: Model architectures ['TTLlamaForCausalLM'] are not supported for now.'
register_vllm_models()


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
        "meta-llama/Llama-3.1-70B-Instruct": ["T3K"],
        "meta-llama/Llama-3.3-70B-Instruct": ["T3K"],
        "Qwen/QwQ-32B": ["T3K"],
        "Qwen/Qwen2.5-72B-Instruct": ["T3K"],
        "Qwen/Qwen2.5-7B-Instruct": ["N300", "T3K"],
        "meta-llama/Llama-3.2-11B-Vision-Instruct": [
            "N300",
            "T3K",
        ],
    }
    cur_mesh_device = os.environ.get("MESH_DEVICE")
    if hf_model_id in default_mesh_device.keys():
        if cur_mesh_device is None:
            # set good default
            os.environ["MESH_DEVICE"] = default_mesh_device[hf_model_id]
            cur_mesh_device = os.environ.get("MESH_DEVICE")

    if hf_model_id in valid_mesh_devices.keys():
        assert (
            cur_mesh_device in valid_mesh_devices[hf_model_id]
        ), f"Invalid MESH_DEVICE for {hf_model_id}"

    logger.info(f"using MESH_DEVICE:={os.environ.get('MESH_DEVICE')}")


def runtime_settings(hf_model_id):
    # default runtime env vars
    env_vars = {
        # note: do note set this post v0.56.0-rc47
        # "TT_METAL_ASYNC_DEVICE_QUEUE": 1,
        "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
    }
    env_var_map = {
        "meta-llama/Llama-3.1-70B-Instruct": {
            "LLAMA_VERSION": "llama3",
        },
        "meta-llama/Llama-3.3-70B-Instruct": {
            "LLAMA_VERSION": "llama3",
        },
        "Qwen/QwQ-32B": {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            "HF_MODEL": hf_model_id.split("/")[-1],
            "LLAMA_CACHE_PATH": os.path.join(
                os.getenv("LLAMA3_CACHE_PATH", ""), os.environ.get("MESH_DEVICE", "")
            ),
        },
        "Qwen/Qwen2.5-72B-Instruct": {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            "HF_MODEL": hf_model_id.split("/")[-1],
            "LLAMA_CACHE_PATH": os.path.join(
                os.getenv("LLAMA3_CACHE_PATH", ""), os.environ.get("MESH_DEVICE", "")
            ),
        },
        "Qwen/Qwen2.5-7B-Instruct": {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            "HF_MODEL": hf_model_id.split("/")[-1],
            "LLAMA_CACHE_PATH": os.path.join(
                os.getenv("LLAMA3_CACHE_PATH", ""), os.environ.get("MESH_DEVICE", "")
            ),
        },
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
            "HF_MODEL": hf_model_id.split("/")[-1],
            "LLAMA_CACHE_PATH": os.path.join(
                os.getenv("LLAMA3_CACHE_PATH", ""), os.environ.get("MESH_DEVICE", "")
            ),
        },
    }
    env_vars.update(env_var_map.get(hf_model_id, {}))
    # Set each environment variable
    logger.info("setting runtime environment variables:")
    for key, value in env_vars.items():
        logger.info(f"{key}={value}")
        os.environ[key] = str(value)


def model_setup(hf_model_id):
    # TODO: check HF repo access with HF_TOKEN supplied
    logger.info(f"using model: {hf_model_id}")
    ensure_mesh_device(hf_model_id)
    runtime_settings(hf_model_id)
    args = {
        "model": hf_model_id,
        "block_size": "64",
        "max_num_seqs": "32",
        "max_model_len": "131072",
        "max_num_batched_tokens": "131072",
        "num_scheduler_steps": "10",
        "max-log-len": "64",
        "port": os.getenv("SERVICE_PORT", "7000"),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
    }

    return args


def main():
    hf_model_id = get_hf_model_id()
    handle_code_versions()
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    logger.info(f"setting vllm logging config at: {config_path}")
    logger.info(f"setting vllm logging file at: {log_path}")
    # note: the vLLM logging environment variables do not cause the configuration
    # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG"] = str(config_path)
    # stop timeout during long sequential prefill batches
    # e.g. 32x 2048 token prefills taking longer than default 30s timeout
    # timeout is 3x VLLM_RPC_TIMEOUT
    os.environ["VLLM_RPC_TIMEOUT"] = "900000"  # 200000ms = 200s
    # vLLM CLI arguments
    args = model_setup(hf_model_id)
    for key, value in args.items():
        if value is not None:
            sys.argv.extend(["--" + key, value])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

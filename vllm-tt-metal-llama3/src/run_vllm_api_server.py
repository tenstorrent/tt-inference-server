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
    metal_llama_dir_commit = "ce8bbbadd52d505cd420ed879d9599d8282210ee"

    if is_head_eq_or_after_commit(
        commit=metal_llama_dir_commit, repo_path=tt_metal_home
    ):
        # tt-metal model_config.py::TtModelArgs.model_name is defined by LLAMA_DIR
        # see https://github.com/tenstorrent/tt-metal/blob/v0.56.0-rc47/models/demos/llama3/tt/model_config.py#L130C13-L130C28
        # needs to match MAX_PREFILL_CHUNK_SIZES_DIV1024 dict format without first dash
        llama_dir = os.getenv("LLAMA_DIR")
        req_llama_path = Path(llama_dir.replace("/Llama-", "/Llama"))
        if not req_llama_path.exists():
            req_llama_path.symlink_to(llama_dir, target_is_directory=True)
        os.environ["LLAMA_DIR"] = str(req_llama_path)
    if os.environ.get("MODEL_IMPL") == "tt-transformers":
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
    elif model_impl == "llama2-t3000":
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
    logger.info(f"using MODEL_IMPL:={os.environ.get('MODEL_IMPL')}")
    # default runtime env vars
    env_vars = {}

    if os.environ.get("MESH_DEVICE") in ["N300", "T3K"]:
        env_vars["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

    # note: do note set this post v0.56.0-rc47
    # env_vars["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1",
    model_impl = os.environ.get("MODEL_IMPL")
    if model_impl == "tt-transformers":
        env_var_map = {
            "meta-llama/Llama-3.1-70B-Instruct": {
                "HF_MODEL": None,
            },
            "meta-llama/Llama-3.3-70B-Instruct": {
                "HF_MODEL": None,
            },
            "Qwen/QwQ-32B": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                "LLAMA_DIR": None,
                "HF_MODEL": os.environ.get(
                    "MODEL_WEIGHTS_PATH", hf_model_id.split("/")[-1]
                ),
                "TT_CACHE_PATH": os.path.join(
                    os.getenv("LLAMA3_CACHE_PATH", ""),
                    os.environ.get("MESH_DEVICE", ""),
                ),
            },
            "Qwen/Qwen2.5-72B-Instruct": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                "LLAMA_DIR": None,
                "HF_MODEL": os.environ.get(
                    "MODEL_WEIGHTS_PATH", hf_model_id.split("/")[-1]
                ),
                "TT_CACHE_PATH": os.path.join(
                    os.getenv("LLAMA3_CACHE_PATH", ""),
                    os.environ.get("MESH_DEVICE", ""),
                ),
            },
            "Qwen/Qwen2.5-7B-Instruct": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                "LLAMA_DIR": None,
                "HF_MODEL": os.environ.get(
                    "MODEL_WEIGHTS_PATH", hf_model_id.split("/")[-1]
                ),
                "TT_CACHE_PATH": os.path.join(
                    os.getenv("LLAMA3_CACHE_PATH", ""),
                    os.environ.get("MESH_DEVICE", ""),
                ),
            },
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1,
                "LLAMA_DIR": None,
                "HF_MODEL": os.environ.get(
                    "MODEL_WEIGHTS_PATH", hf_model_id.split("/")[-1]
                ),
                "TT_CACHE_PATH": os.path.join(
                    os.getenv("LLAMA3_CACHE_PATH", ""),
                    os.environ.get("MESH_DEVICE", ""),
                ),
            },
        }
    elif model_impl == "subdevices":
        env_var_map = {
            "meta-llama/Llama-3.1-70B-Instruct": {
                "LLAMA_VERSION": "llama3",
            },
        }
    elif model_impl == "llama2-t3000":
        env_var_map = {
            "meta-llama/Llama-3.1-70B-Instruct": {
                "LLAMA_VERSION": "llama3",
                "LLAMA_DIR": os.environ["MODEL_WEIGHTS_PATH"],
            },
            "meta-llama/Llama-3.3-70B-Instruct": {
                "LLAMA_VERSION": "llama3",
                "LLAMA_DIR": os.environ["MODEL_WEIGHTS_PATH"],
            },
        }
    env_vars.update(env_var_map.get(hf_model_id, {}))
    # Set each environment variable
    logger.info("setting runtime environment variables:")
    for key, value in env_vars.items():
        logger.info(f"{key}={value}")
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
        and os.environ["MESH_DEVICE"] == "TG"
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

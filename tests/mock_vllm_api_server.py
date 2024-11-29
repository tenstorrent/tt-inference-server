# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import runpy
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import jwt
from huggingface_hub import hf_hub_download

# mock out ttnn fully
sys.modules["ttnn"] = MagicMock()
sys.modules["ttnn.device"] = MagicMock()

from vllm import ModelRegistry

# import classes to mock
from vllm.worker.tt_worker import TTWorker, TTCacheEngine
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.llm_engine import LLMEngine

from utils.logging_utils import set_vllm_logging_config, logging_init_wrapper
from mock_vllm_model import (
    new_init_cache_enginer,
    new_allocate_kv_cache,
    MockModel,
)

# register the mock model
ModelRegistry.register_model("TTLlamaForCausalLM", MockModel)


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def setup_mock_model_weights(cache_root: str, weights_dir: str, hf_token: str):
    if not hf_token:
        raise ValueError(
            "HuggingFace token (HF_TOKEN environment variable) is required"
        )

    metal_ckpt_dir = Path(weights_dir)
    metal_tokenizer_path = metal_ckpt_dir / "tokenizer.model"
    # Create path objects
    # weights_dir = Path(cache_root) / "model_weights" / f"repacked-{model}"
    metal_cache_path = (
        Path(cache_root) / "tt_metal_cache" / f"cache_{metal_ckpt_dir.name}"
    )

    # Create directories
    metal_ckpt_dir.mkdir(parents=True, exist_ok=True)
    metal_cache_path.mkdir(parents=True, exist_ok=True)

    # Define files to download
    files_to_download = ["original/tokenizer.model", "original/params.json"]

    # Download files using huggingface_hub
    for file in files_to_download:
        downloaded_path = hf_hub_download(
            repo_id="meta-llama/Llama-3.1-70B-Instruct",
            filename=file,
            token=hf_token,
            local_dir=metal_ckpt_dir,
        )
        # Move file to weights directory
        target_path = metal_ckpt_dir / Path(file).name
        shutil.move(downloaded_path, target_path)
        print(f"Downloaded {file} to {target_path}")

    # Clean up original directory if it exists and is empty
    original_dir = metal_ckpt_dir / "original"
    if original_dir.exists() and not any(original_dir.iterdir()):
        original_dir.rmdir()
    return str(metal_ckpt_dir), str(metal_tokenizer_path), str(metal_cache_path)


def patched_run_mp_engine(engine_args, usage_context, ipc_path):
    # This function wraps the original `run_mp_engine` function because
    # vLLM engine process is spawned with multiprocessing.get_context("spawn")
    # so we need to apply the patches to this target function
    with patch.object(TTWorker, "init_device", new=lambda x: None), patch.object(
        TTWorker, "_init_cache_engine", new=new_init_cache_enginer
    ), patch.object(
        TTCacheEngine, "_allocate_kv_cache", new=new_allocate_kv_cache
    ), patch.object(LLMEngine, "__init__", new=logging_init_wrapper):
        # Call the original `run_mp_engine` with patches applied
        run_mp_engine(engine_args, usage_context, ipc_path)


@patch("vllm.engine.multiprocessing.engine.run_mp_engine", new=patched_run_mp_engine)
def main():
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    print(f"setting vllm logging config at: {config_path}")
    print(f"setting vllm logging file at: {log_path}")
    # note: the vLLM logging environment variables do not cause the configuration
    # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG"] = str(config_path)
    # automate setup of the mock model tokenizer and params used by llama 3.1 implementation
    metal_ckpt_dir, metal_tokenizer_path, metal_cache_path = setup_mock_model_weights(
        cache_root=os.environ["CACHE_ROOT"],
        weights_dir=os.environ["MODEL_WEIGHTS_PATH"],
        hf_token=os.environ["HF_TOKEN"],
    )
    os.environ["LLAMA3_CKPT_DIR"] = metal_ckpt_dir
    os.environ["LLAMA3_TOKENIZER_PATH"] = metal_tokenizer_path
    os.environ["LLAMA3_CACHE_PATH"] = metal_cache_path
    # stop timeout during long sequential prefill batches
    os.environ["VLLM_RPC_TIMEOUT"] = "200000"  # 200000ms = 200s
    # vLLM CLI arguments
    args = {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "block_size": "64",
        "max_num_seqs": "32",
        "max_model_len": "131072",
        "max_num_batched_tokens": "131072",
        "num_scheduler_steps": "10",
        "port": os.getenv("SERVICE_PORT", "7000"),
        "seed": "4862",
        "download-dir": os.getenv("CACHE_DIR", None),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
    }
    for key, value in args.items():
        if value is not None:
            sys.argv.extend(["--" + key, value])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

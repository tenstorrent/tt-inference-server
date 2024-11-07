# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import runpy
import json
from unittest.mock import patch

import jwt
from vllm import ModelRegistry

# import classes to mock
from vllm.worker.tt_worker import TTWorker, TTCacheEngine
from mock_vllm_model import (
    new_init_cache_enginer,
    new_allocate_kv_cache,
    MockModel,
    new__init__,
)
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.multiprocessing.engine import MQLLMEngine

# register the mock model
ModelRegistry.register_model("TTLlamaForCausalLM", MockModel)


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def patched_run_mp_engine(engine_args, usage_context, ipc_path):
    # This function wraps the original `run_mp_engine` function because
    # vLLM engine process is spawned with multiprocessing.get_context("spawn")
    # so we need to apply the patches to this target function
    with patch.object(TTWorker, "init_device", new=lambda x: None), patch.object(
        TTWorker, "_init_cache_engine", new=new_init_cache_enginer
    ), patch.object(
        TTCacheEngine, "_allocate_kv_cache", new=new_allocate_kv_cache
    ), patch.object(MQLLMEngine, "__init__", new=new__init__):
        # Call the original `run_mp_engine` with patches applied
        run_mp_engine(engine_args, usage_context, ipc_path)


@patch("vllm.engine.multiprocessing.engine.run_mp_engine", new=patched_run_mp_engine)
def main():
    # vLLM CLI arguments
    args = {
        "model": "meta-llama/Meta-Llama-3.1-70B",
        "block_size": "64",
        "max_num_seqs": "32",
        "max_model_len": "131072",
        "max_num_batched_tokens": "131072",
        "num_scheduler_steps": "10",
        "port": os.getenv("SERVICE_PORT", "8000"),
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

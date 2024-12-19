# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import runpy
import json

import jwt
from vllm import ModelRegistry

from utils.logging_utils import set_vllm_logging_config

# importing from tt-metal install path
from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM

# register the model
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def main():
    # set up logging
    config_path, log_path = set_vllm_logging_config(level="DEBUG")
    print(f"setting vllm logging config at: {config_path}")
    print(f"setting vllm logging file at: {log_path}")
    # note: the vLLM logging environment variables do not cause the configuration
    # to be loaded in all cases, so it is loaded manually in set_vllm_logging_config
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG"] = str(config_path)
    # stop timeout during long sequential prefill batches
    # e.g. 32x 2048 token prefills taking longer than default 30s timeout
    # timeout is 3x VLLM_RPC_TIMEOUT
    os.environ["VLLM_RPC_TIMEOUT"] = "200000"  # 200000ms = 200s
    # vLLM CLI arguments
    args = {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "block_size": "64",
        "max_num_seqs": "32",
        "max_model_len": "131072",
        "max_num_batched_tokens": "131072",
        "num_scheduler_steps": "10",
        "max-log-len": "32",
        "port": os.getenv("SERVICE_PORT", "7000"),
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

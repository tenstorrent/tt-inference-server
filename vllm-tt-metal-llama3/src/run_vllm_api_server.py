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

# Import and register models from tt-metal
from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM
from models.demos.llama3.tt.generator_vllm import TtMllamaForConditionalGeneration

ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
ModelRegistry.register_model(
    "TTMllamaForConditionalGeneration", TtMllamaForConditionalGeneration
)


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def get_hf_model_id():
    model = os.environ.get("HF_MODEL_REPO_ID")
    if not model:
        print("Must set environment variable: HF_MODEL_REPO_ID")
        sys.exit()
    return model


def model_setup(hf_model_id):
    # TODO: check HF repo access with HF_TOKEN supplied
    print(f"using model: {hf_model_id}")
    args = {
        "model": hf_model_id,
        "block_size": "64",
        "max_num_seqs": "32",
        "max_model_len": "131072",
        "max_num_batched_tokens": "131072",
        "num_scheduler_steps": "10",
        "max-log-len": "64",
        "port": os.getenv("SERVICE_PORT", "7000"),
        "download-dir": os.getenv("CACHE_DIR", None),
        "api-key": get_encoded_api_key(os.getenv("JWT_SECRET", None)),
    }
    if hf_model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        if os.environ.get("MESH_DEVICE") is None:
            os.environ["MESH_DEVICE"] = "N300"
        else:
            assert os.environ["MESH_DEVICE"] in [
                "N300",
                "T3K_LINE",
            ], "Invalid MESH_DEVICE for multi-modal inference"

    return args


def main():
    hf_model_id = get_hf_model_id()
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
    args = model_setup(hf_model_id)
    for key, value in args.items():
        if value is not None:
            sys.argv.extend(["--" + key, value])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()

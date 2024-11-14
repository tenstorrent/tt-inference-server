# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import runpy
import json

import jwt
from vllm import ModelRegistry

# importing logging utils
from logging_utils import new__init__
from vllm.engine.multiprocessing.engine import MQLLMEngine
from unittest.mock import patch

# importing from tt-metal install path
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration

# register the model
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


@patch.object(MQLLMEngine, "__init__", new=new__init__)
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

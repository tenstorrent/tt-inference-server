# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from unittest.mock import patch

from vllm import ModelRegistry

# import mocking utils + classes to mock
from mock_vllm_model import (
    MockModel,
    new_allocate_kv_cache,
    new_init_cache_enginer,
    setup_mock_model_weights,
)
from vllm.worker.tt_worker import TTCacheEngine, TTWorker
from benchmark_vllm_offline_inference import run_inference, parse_args

ModelRegistry.register_model("TTLlamaForCausalLM", MockModel)


@patch.object(TTWorker, "init_device", new=lambda x: None)
@patch.object(TTWorker, "_init_cache_engine", new=new_init_cache_enginer)
@patch.object(TTCacheEngine, "_allocate_kv_cache", new=new_allocate_kv_cache)
def mock_run_inference(*args, **kwargs):
    run_inference(*args, **kwargs)


if __name__ == "__main__":
    metal_ckpt_dir, metal_tokenizer_path, metal_cache_path = setup_mock_model_weights(
        cache_root=os.environ["CACHE_ROOT"],
        weights_dir=os.environ["MODEL_WEIGHTS_PATH"],
        hf_token=os.environ["HF_TOKEN"],
    )
    os.environ["LLAMA3_CKPT_DIR"] = metal_ckpt_dir
    os.environ["LLAMA3_TOKENIZER_PATH"] = metal_tokenizer_path
    os.environ["LLAMA3_CACHE_PATH"] = metal_cache_path
    args = parse_args()
    mock_run_inference(
        args.prompts_json,
        prompt_len=args.input_seq_len,
        max_tokens=args.output_seq_len,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch,
        async_engine=args.async_engine,
    )

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import argparse
import time
import uvloop
from tqdm import tqdm

from vllm import LLM, ModelRegistry, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.utils import merge_async_iterators

# importing logging utils
from utils.logging_utils import RawStatLogger

# Import and register model from tt-metal
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration

ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_json",
        type=str,
        default="/home/container_app_user/vllm/tt_metal/prompts.json",
        help="Path to JSON file containing prompts",
    )
    parser.add_argument(
        "--input_seq_len",
        type=int,
        default=128,
        help="Length of dummy prompts for performance measurement",
    )
    parser.add_argument(
        "--output_seq_len", type=int, default=128, help="Length of outputs"
    )
    parser.add_argument(
        "--greedy_sampling",
        action="store_true",
        help="Use greedy decoding instead of top-k/p",
    )
    parser.add_argument(
        "--max_seqs_in_batch",
        type=int,
        default=32,
        help="Maximum batch size for inference",
    )
    parser.add_argument("--async_engine", action="store_true", help="Use async engine")
    return parser.parse_args()


def run_inference(
    prompts_json,
    max_tokens,
    max_seqs_in_batch,
    prompt_len,
    greedy_sampling,  # Option to use greedy decoding instead of top-k/p
    async_engine,
):
    # LLM args
    engine_kw_args = {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "block_size": 64,
        "max_num_seqs": max_seqs_in_batch,
        "max_model_len": 131072,
        "disable_log_stats": False,
        "max_num_batched_tokens": 131072,
        "log_global_stats": True,
        "num_scheduler_steps": 10,
        "disable_async_output_proc": True,
    }

    # Generation args
    ignore_eos = True

    print("Generating prompts with output length", max_tokens)
    if greedy_sampling:
        sampling_params = SamplingParams(
            max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
            top_k=10,
            top_p=0.9,
            temperature=1.0,
        )

    # Prepare inputs
    assert prompt_len is not None, "prompt_len is required to generate dummy prompts"
    print("Measuring performance with dummy prompts of length", prompt_len)
    prompt_token_ids = [[0] * prompt_len] * max_seqs_in_batch  # dummy prompts
    sampling_params = (
        sampling_params[:max_seqs_in_batch]
        if isinstance(sampling_params, list)
        else sampling_params
    )

    # check prompt lengths fit in model context
    max_model_len = engine_kw_args["max_model_len"]
    assert_str = f"prompt length ({prompt_len}) + num generated tokens ({sampling_params.max_tokens}) will exceed max_model_len ({max_model_len})"
    assert prompt_len + sampling_params.max_tokens <= max_model_len, assert_str

    # Create and run LLM
    if not async_engine:
        llm = LLM(**engine_kw_args)
        # Add raw stats logging to the llm engine
        llm.llm_engine.stat_loggers["raw_logging"] = RawStatLogger(
            engine_kw_args["num_scheduler_steps"],
            batch_size=engine_kw_args["max_num_seqs"],
        )
        run_inference_perf(llm, prompt_token_ids, sampling_params)
    else:
        print("Using async engine")
        engine_args = AsyncEngineArgs(**engine_kw_args)

        async def _run_inference_async():
            async with build_async_engine_client_from_engine_args(engine_args) as llm:
                await run_inference_perf_async(llm, prompt_token_ids, sampling_params)

        uvloop.run(_run_inference_async())


def run_inference_perf(
    llm: LLM,
    prompt_token_ids,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    """Run llm N_inference times and measure the average time taken per inference run."""
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        generate_tokens(
            llm, None, sampling_params, prompt_token_ids, print_output=False
        )
    avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


async def run_inference_perf_async(
    llm: LLM,
    prompt_token_ids,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        await generate_tokens_async(
            llm, None, sampling_params, prompt_token_ids, print_output=False
        )
    avg_time = (time.perf_counter() - start_time) / (N_inference - N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


def generate_tokens(
    llm: LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True
):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if print_output:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


async def generate_tokens_async(
    llm: MQLLMEngineClient,
    prompts,
    sampling_params,
    prompt_token_ids=None,
    print_output=True,
):
    # Use tokenized prompts if provided
    if prompt_token_ids is not None:
        prompts = []
        for single_prompt_token_ids in prompt_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=single_prompt_token_ids))

    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    generators = []
    for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
        generator = llm.generate(prompt, sp, request_id=f"test{i}")
        generators.append(generator)
    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        prompt = res.prompt
        generated_text = res.outputs[0].text
        if print_output and res.finished:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        args.prompts_json,
        prompt_len=args.input_seq_len,
        max_tokens=args.output_seq_len,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch,
        async_engine=args.async_engine,
    )

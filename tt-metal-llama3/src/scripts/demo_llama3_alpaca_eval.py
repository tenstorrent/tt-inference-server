# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from datetime import datetime

from time import time
from loguru import logger

import ttnn
import tt_lib as ttl
from conftest import get_dispatch_core_type

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from transformers.generation.utils import top_k_top_p_filtering
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from models.demos.t3000.llama2_70b.tt.llama_common import load_llama_state_dict
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import ChatFormat
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_device_mesh,
    string_similarity_score,
)


@dataclass
class ModelArgs:
    implementation: str = None
    llama_version: str = None
    ckpt_dir: str = None
    tokenizer_path: str = None
    skip_model_load: bool = False
    max_batch_size: int = 32
    num_layers: int = None
    max_seq_len: int = 4096
    max_kv_context_len: int = 4096


@dataclass
class TTArgs:
    device_mesh: object = None
    n_devices: int = 8
    emulated: bool = False
    cache_path: str = None
    decode_only: bool = False


@dataclass
class DataArgs:
    max_output_tokens: int = 128
    prompts_file: str = None
    output_at_end: bool = True
    top_p: float = 1
    top_k: int = 1
    temperature: float = 1.0
    chat: bool = False
    sample_len: int = None
    ground_truth: str = None


@dataclass
class DemoArgs:
    model: ModelArgs
    tt: TTArgs
    data: DataArgs


def construct_arg(**kwargs):
    model_args = ModelArgs(**{k: v for k, v in kwargs.items() if hasattr(ModelArgs, k)})
    tt_args = TTArgs(**{k: v for k, v in kwargs.items() if hasattr(TTArgs, k)})
    data_args = DataArgs(**{k: v for k, v in kwargs.items() if hasattr(DataArgs, k)})
    return DemoArgs(model=model_args, tt=tt_args, data=data_args)


def main(args):
    # Set random reproducible seed
    torch.manual_seed(0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = (
        f"/home/user/cache_root/demo_user_output_{timestamp}.txt"
    )
    model_args = args.model
    tt_args = args.tt
    data_args = args.data

    # Load ground truth if available
    if data_args.ground_truth:
        if not os.path.exists(data_args.ground_truth):
            logger.info(f"Ground truth file {data_args.ground_truth} does not exist.")
            data_args.ground_truth = None
        else:
            ground_truth_outputs = json.load(open(data_args.ground_truth, "r"))

            if len(ground_truth_outputs) == 0:
                logger.info("Ground truth outputs are empty")
                data_args.ground_truth = None
            else:
                logger.info(f"Loaded {len(ground_truth_outputs)} ground truth outputs")

    generator = build_generator(model_args, tt_args)

    # Load the model and tokenizer
    model, tokenizer = generator.model, generator.tokenizer

    batch_tokenized, batch_prompts = load_alpaca_eval(tokenizer, batch_size=32, n_batches=25)

    # Run decode
    with torch.no_grad():
        for loop_idx in range(100):
            for batch_idx, (tokenized, prompts) in enumerate(
                zip(batch_tokenized, batch_prompts)
            ):
                logger.info(f"starting: dataset_loop: {loop_idx}, batch: {batch_idx}, n_users:= {len(tokenized)}")
                all_text = run_decode(
                    model_args,
                    tt_args,
                    data_args,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_tokens=tokenized,
                    prompts=prompts,
                )
                logger.info(f"finished batch: {batch_idx}.")
                # write output after each batch
                if data_args.output_at_end:
                    with open(output_filename, "a") as f:
                        for i, (text, prompt) in enumerate(zip(all_text, prompts)):
                            f.write(
                                f"\nbatch: {batch_idx} user: {i}\nprompt: {prompt}\noutput: {text}\n"
                            )


def build_generator(model_args, tt_args):
    generator = Llama.build(
        ckpt_dir=model_args.ckpt_dir,
        tokenizer_path=model_args.tokenizer_path,
        max_seq_len=model_args.max_seq_len,
        max_batch_size=model_args.max_batch_size,
        skip_model_load=model_args.skip_model_load,
        n_layers=1 if model_args.implementation == "tt" else model_args.num_layers,
    )

    state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

    if model_args.implementation == "tt":
        generator.model = TtLlamaModelForGeneration(
            configuration=generator.model.params,
            state_dict=state_dict,
            model_args=model_args,
            tt_args=tt_args,
        )
    return generator


def get_sampling_func(top_k, top_p, temperature):
    if top_k == 1:
        return lambda x: torch.argmax(x, dim=-1).reshape(-1)  # TODO: remove :, -1 since outer code already does that
    else:
        return lambda x: top_pk_logits_efficient(x, p=top_p, k=top_k, temperature=temperature).reshape(-1)


def load_alpaca_eval(tokenizer, batch_size, n_batches):
    bsz = batch_size
    n_samples = bsz * n_batches
    alpaca_ds = load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval", split=f"eval[:{n_samples}]"
    )
    logger.info(f"loaded {len(alpaca_ds)} samples from tatsu-lab/alpaca_eval")
    batch_tokenized = []
    batch_prompts = []
    for batch_idx in range(0, len(alpaca_ds) // bsz):
        batch = alpaca_ds[(batch_idx * bsz) : ((batch_idx * bsz) + bsz)]
        prompts = [batch["instruction"][i] for i in range(0, bsz)]
        tokenized = [tokenizer.encode(p, bos=True, eos=False) for p in prompts]
        batch_prompts.append(prompts)
        batch_tokenized.append(tokenized)
    return batch_tokenized, batch_prompts


def initialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu").clone().detach()
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask, eos_reached


def prepare_next_input(tokenizer, tokens, input_text_mask, finished_mask, prompt_lens, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token
    # llama3 has multiple stop tokens: EOS and EOT
    stop_ids = torch.tensor(list(tokenizer.stop_tokens))
    eos_reached = (input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_ids))
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def run_decode(
    model_args,
    tt_args,
    data_args,
    model,
    tokenizer,
    prompt_tokens,
    prompts,
    return_logits=False,
    return_full_logits=False,
):
    """
    return_logits: return the logits for the last token
    return_full_logits: return the logits for all tokens
    """
    assert not (return_logits and return_full_logits), "return_logits and return_full_logits cannot both be true"

    # decode arguments
    bsz = model_args.max_batch_size
    output_tokens = data_args.max_output_tokens

    sampling_func = get_sampling_func(data_args.top_k, data_args.top_p, data_args.temperature)

    prompt_lens = [len(t) for t in prompt_tokens]
    min_prompt_len = min(prompt_lens) if not tt_args.decode_only else 1
    max_prompt_len = max(prompt_lens)
    assert max_prompt_len <= model_args.max_kv_context_len
    total_len = min(model_args.max_kv_context_len, max_prompt_len + output_tokens)
    assert total_len <= model_args.max_kv_context_len
    if total_len != max_prompt_len + output_tokens:
        logger.warning(
            f"Requested more output tokens than allowed by model. Truncating to {total_len - max_prompt_len} output tokens."
        )

    # prepare inputs
    tokens, input_text_mask, finished_mask = initialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    prev_pos = 0

    # some profiling and logging
    latencies = []
    full_logits = []

    for cur_pos in range(min_prompt_len, total_len):
        logger.info(f"Loop {cur_pos}")
        start = time()
        input_tokens = tokens[:, prev_pos:cur_pos]
        logits = model.forward(input_tokens, prev_pos)

        next_logits = logits[:, -1, :]  # batch, vocab of last token
        next_token = sampling_func(next_logits)

        tokens, cur_finished_mask, prev_pos = prepare_next_input(
            tokenizer, tokens, input_text_mask, finished_mask, prompt_lens, cur_pos, next_token
        )
        latencies.append(time() - start)

        # keep track of if stop token previously generated
        finished_mask = cur_finished_mask | finished_mask
        if all(finished_mask):
            break

        if return_full_logits:
            full_logits.append(logits.clone().detach())

    latency_printout(latencies, model_args, total_len - min_prompt_len)
    output = get_all_text(tokenizer, tokens, prompt_tokens, output_tokens)

    if return_logits:
        output = (output, logits)
    elif return_full_logits:
        full_logits = torch.cat(full_logits, dim=1)
        output = (output, full_logits)
    return output


def latency_printout(latencies, model_args, generated_len):
    latencies = [
        latency for token_pos, latency in enumerate(latencies) if token_pos % 32 != 0
    ]  # We recompute program_cache for multiples of 32
    overall_time = sum(latencies)
    overall_tokens = model_args.max_batch_size * len(latencies)
    # warmup_batch = 2
    # Skip initial warmup batch
    if len(latencies) > warmup_batch:
        overall_time -= sum(latencies[:warmup_batch])
        overall_tokens -= warmup_batch * model_args.max_batch_size
        latencies = latencies[warmup_batch:]

    mean_latency = sum(latencies) / len(latencies) if len(latencies) > 0 else 0

    tokens_per_second = 1 / mean_latency if mean_latency != 0 else 0
    overall_tokens_per_second = overall_tokens / overall_time if overall_time != 0 else 0
    tokens_per_second_per_user = (
        overall_tokens_per_second / model_args.max_batch_size if model_args.max_batch_size != 0 else 0
    )
    throughput = 1000 * overall_time / overall_tokens if overall_tokens != 0 else 0

    logger.info(f"Overall throughput: {throughput:.1f} ms @ {overall_tokens_per_second:.1f} tokens/s")
    logger.info(f"Tokens per second per user: {tokens_per_second_per_user:.1f} tokens/s/u")
    logger.info(f"User latency: {1000 * mean_latency:.1f} ms @ {tokens_per_second:.1f} tokens/s")


def get_all_text(tokenizer, tokens, prompt_tokens, max_gen_len):
    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        try:
            # cut to max gen len
            start = 0
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        except IndexError:
            logger.info(f"Index out of range for sequence {i}, returning entire sequence.")
            pass
        # cut to 1st stop token
        for stop_tok in tokenizer.stop_tokens:
            if stop_tok in toks:
                stop_idx = toks.index(stop_tok)
                toks = toks[:stop_idx]
        out_tokens.append(toks)
    all_text = [tokenizer.decode(toks) for toks in out_tokens]
    return all_text


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def get_t3k_device_mesh(num_devices_requested):
    logger.info("get_t3k_device_mesh ...")
    assert ttnn.get_num_devices() == 8
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    # device_params is empty dict in llama3 70B demo pytest execution
    device_params = {}
    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested], dispatch_core_type=get_dispatch_core_type(), **device_params
    )
    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    return device_mesh


def close_devices(device_mesh):
    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)

    ttnn.close_device_mesh(device_mesh)
    del device_mesh


if __name__ == "__main__":
    # =================================
    # test_LlamaModel_demo arguments
    # =================================
    implementation = "tt"
    skip_model_load = False
    num_layers = 80
    # num_tokens = 2048
    # Generation args
    # max_output_tokens = 128
    max_output_tokens = 4096
    prompts_file = None
    output_at_end = True
    # greedy
    # top_k = 1
    # top_p = 1.0
    # sampling
    top_k = 10
    top_p = 0.9
    temperature = 1.0
    chat = True
    # TT args
    # t3k_device_mesh,
    n_devices = 8
    decode_only = False
    llama_version = "llama3"
    ground_truth = False
    max_batch_size= 32
    max_context_len = 2048
    # use_program_cache
    # =================================
    logger.info("Running LlamaModel demo - first run")
    ## Get model config
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )
    # device setup
    t3k_device_mesh = get_t3k_device_mesh(num_devices_requested=n_devices)
    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_async(True)
        # use_program_cache
        device.enable_program_cache()
    check_device_mesh(t3k_device_mesh, model_config)

    args = construct_arg(
        implementation=implementation,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        max_batch_size=max_batch_size,
        max_kv_context_len=max_context_len,
        num_layers=num_layers,
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=chat,
        device_mesh=t3k_device_mesh,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=decode_only,
        llama_version=llama_version,
        ground_truth=ground_truth,
    )
    main(args)
    close_devices(t3k_device_mesh)

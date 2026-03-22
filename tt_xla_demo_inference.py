# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel

DEFAULT_PROMPTS = [
    "I like taking walks in the",
    "My name is",
    "My favorite color is",
    "Cheese is an excellent",
    "While ham sandwiches are great, I prefer",
    "My bicycle is broken, and",
    "The best way to start my day is with",
    "I love to",
    "I am not a fan of",
    "Toronto has an explosive restaurant scene. There are many different cuisines to",
    "My favorite season is",
    "My least favorite season is",
    "Bison meat is not",
    "If you forget to wear your shoes when you go hiking,",
    "My right elbow is in pain, and",
    "To make a grilled cheese, start by",
    "The internal combustion engine",
    "The first person to walk on the moon was",
    "The ocean is home to many",
    "I",
    "Calculus is a branch of mathematics that",
    "The most common type of cloud is",
    "A matrix dot product is",
    "The 300 Spartans were",
    "Napoleon Bonaparte was born in",
    "The quick brown fox jumps over the lazy dog. And then he",
    "The original title of the book 'The Great Gatsby' was",
    "I forgot to tie my shoelaces",
    "Playing video games",
    "The sun is",
    "My car's transmission is",
    "My keyboard is in a language I don't understand! What",
]

PEFT_MODEL_PATH = "/localdev/mcupac/tt-xla/my_model/b571680f-b3af-4c7c-a7ce-285e0a7ebd12"


# --------------------------------
# Llama Generation Loop Example
# --------------------------------
def llama(interactive: bool = False):

    # Check transformers version
    check_transformers_version()

    # Set up config variables.
    max_cache_len: int = 128
    model_name: str = "google/gemma-1.1-2b-it"

    # Determine if SPMD mode should be enabled, if more than 1 device is available.
    # SPMD must be turned off for llama generate on 1x1 mesh - See https://github.com/tenstorrent/tt-xla/issues/1639
    num_devices = xr.global_runtime_device_count()
    is_spmd: bool = num_devices > 1
    if is_spmd:
        setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    while True:
        if interactive:
            user_prompt = input("Enter your prompt or quit() to exit: ")
            batch_size: int = 1
            if user_prompt.lower() == "quit()":
                break
            user_prompt = [user_prompt]
        else:
            batch_size: int = 32
            user_prompt = DEFAULT_PROMPTS[:batch_size]

        # Construct inputs, including static cache
        input_args = construct_inputs(
            user_prompt, tokenizer, model.config, batch_size, max_cache_len
        )

        # Limit maximum generation count to fit within preallocated static cache
        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        # Transfer model and inputs to device
        model, input_args = transfer_to_device(model, input_args, device)

        # Mark sharding on inputs and model internals if SPMD is enabled
        if is_spmd:
            mark_sharding_on_inputs_and_model(model, input_args, mesh)

        # Compile model
        compiled_model = torch.compile(model, backend="tt")

        # Run generation loop until EOS token generated or max tokens reached
        run_generate(
            compiled_model,
            input_args,
            tokenizer,
            device,
            mesh,
            is_spmd,
            max_tokens_to_generate,
            user_prompt,
            interactive,
        )

        if not interactive:
            break


def setup_spmd():
    """
    Initializes SPMD mode in torch_xla.
    """

    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def setup_model_and_tokenizer(
    model_name: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    if PEFT_MODEL_PATH is not None:
        model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
        model = model.merge_and_unload()
        print(f"Loaded and merged PEFT adapter from: {PEFT_MODEL_PATH}")

    model = model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: str,
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        input_prompt: Input text prompt
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=32,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # Static cache should be initialized on CPU and separately transferred to device
    # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache: StaticCache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    # Attention mask is needed to ignore padding tokens in left-padded batches. The mask should match max_cache_len
    # to prevent recompilation or implicit padding by transformers, which can cause degenerate output.
    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    #   Debug prints
    print("\n=== DEBUG: construct_inputs ===")
    print(f"Input prompt: '{input_prompt}'")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Input attention mask shape: {inputs.attention_mask.shape}")
    print(f"Full attention mask shape (pre-allocated): {full_attention_mask.shape}")
    print(f"Full attention mask: {full_attention_mask}")
    print(f"Cache position shape: {cache_position.shape}")
    print(f"Cache position: {cache_position}")
    print(f"Actual sequence length (non-padding): {inputs.attention_mask.sum().item()}")
    print("=" * 50)

    return input_args


def transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Transfer model and inputs to device.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple of (model, input_args) on device
    """
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    model = model.to(device)

    return model, input_args


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh
):
    """
    Mark sharding on inputs and model internals.
    If mark_sharding is not called on a tensor, it is fully replicated across all devices.
        i.e. on cache_positions, input_ids

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        mesh: Device mesh for SPMD operations
    """

    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Shard model internals
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    is_spmd: bool = True,
    max_tokens_to_generate: int = 128,
    input_prompt: List[str] = [""],
    is_interactive: bool = False,
):
    """
    Run the generation loop.

    Args:
        compiled_model: Compiled model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        mesh: Device mesh for SPMD operations (optional)
        is_spmd: Whether SPMD mode is enabled
        max_tokens_to_generate: Maximum number of tokens to generate
    """
    num_users = input_args["input_ids"].shape[0]
    output_tokens: List[List[str]] = [[] for _ in range(num_users)]
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")
                if is_interactive:
                    print(f"Result: {input_prompt[0]}", end="", flush=True)

            # Run forward pass
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            next_token_id = output_logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[i]) for i in range(num_users)]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])
                if is_interactive:
                    print(output_text[i], end="", flush=True)

            # Check for EOS token and early exit
            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    print()
    if not is_interactive:
        for i in range(num_users):
            print(f"Result for user {i}: {input_prompt[i]}{''.join(output_tokens[i])}")
            print()


def check_transformers_version():
    """
    Check that transformers version is <= 4.52.4.
    Raises RuntimeError if version is incompatible.

    This is because transformers SDPA implementation changed in later versions,
    which causes dynamo trace to fail.

    See https://github.com/tenstorrent/tt-xla/issues/1020
    """
    import packaging.version

    current_version = packaging.version.parse(transformers.__version__)
    max_version = packaging.version.parse("4.57.1")

    if current_version > max_version:
        raise RuntimeError(
            f"Transformers version {transformers.__version__} is not supported. "
            f"Please use version <= 4.57.1"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama generation example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for entering custom prompts",
    )
    args = parser.parse_args()

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama(interactive=args.interactive)

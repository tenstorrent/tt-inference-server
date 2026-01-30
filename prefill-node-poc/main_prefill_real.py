# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 prefill demo using the PrefillEngine infrastructure.

Runs prefill on the model using PrefillEngine with scheduler, block manager,
and model runner. Demonstrates KV cache access after prefill.
Prints the first token (greedy sampled) for each prompt.

Usage:
    python main_prefill_real.py --model-path <HF_MODEL> --cache-dir <CACHE_DIR>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn

# Add tt-metal to path for imports
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "/data/dmadic/tt-metal")
if TT_METAL_HOME not in sys.path:
    sys.path.insert(0, TT_METAL_HOME)

from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

from model_runner import KvCacheConfig, ModelConfig
from prefill_engine import PrefillEngine
from scheduler import SchedulerConfig


@dataclass
class DemoConfig:
    """Configuration for the prefill demo."""
    
    # Scheduler config
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 8192
    block_size: int = 32
    
    # KV cache memory settings
    available_kv_cache_memory_gb: float = 12.0
    num_layers: int = 61
    kvpe_dim: int = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
    kv_cache_dtype_bytes: int = 1  # bfloat8 for DeepSeek
    kv_tensors_per_layer: int = 1  # Combined KV/MLA for DeepSeek
    
    # Model settings
    vocab_size: int = 129280
    
    # Default prompts
    prompts: list[str] = field(default_factory=lambda: [
        "What is the capital of France?",
    ])


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DeepSeek-V3 Prefill Demo")
    p.add_argument(
        "prompts",
        type=str,
        nargs="*",
        default=[],
        help="Prompt text(s) to run prefill on.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to local HF DeepSeek-V3 model (safetensors)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Path to weight cache directory",
    )
    p.add_argument(
        "--prefill-max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to prefill (truncates longer prompts).",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum number of sequences to batch together.",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens in a batch.",
    )
    p.add_argument(
        "--kv-cache-num-blocks",
        type=int,
        default=None,
        help="Explicit number of KV cache blocks (overrides automatic calculation).",
    )
    return p


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Apply greedy sampling (argmax) to logits."""
    return torch.argmax(logits, dim=-1)


def get_last_token_logits(
    logits: torch.Tensor,
    prompt_lens: list[int],
    prefill_max_tokens: int | None = None,
) -> torch.Tensor:
    """
    Extract logits at the last actual token position for each sequence.
    """
    batch_size = logits.shape[0]
    last_logits = []
    
    for i in range(batch_size):
        prompt_len = prompt_lens[i]
        if prefill_max_tokens is not None:
            prompt_len = min(prompt_len, prefill_max_tokens)
        last_logits.append(logits[i, prompt_len - 1, :])
    
    return torch.stack(last_logits, dim=0)


def run_prefill_demo(
    prompts: list[str],
    model_path: str,
    cache_dir: str,
    prefill_max_tokens: int | None = None,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 8192,
    kv_cache_num_blocks: int | None = None,
) -> dict:
    """
    Run the prefill demo using PrefillEngine.
    Demonstrates KV cache access after prefill.
    """
    # Get mesh device configuration
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError(
            "Environment variable $MESH_DEVICE is not set. "
            "Please set it to DUAL, QUAD, or TG."
        )
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    logging.info(f"Selected MESH_DEVICE: '{requested_system_name}' - mesh shape: {mesh_shape}")
    
    # Configure fabric
    fabric_config = ttnn.FabricConfig.FABRIC_1D
    logging.info(f"Setting fabric config to {fabric_config}")
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)
    
    # Open mesh device
    logging.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    
    engine = None
    try:
        # Create configs
        demo_cfg = DemoConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        
        sched_config = SchedulerConfig(
            max_num_seqs=demo_cfg.max_num_seqs,
            max_num_batched_tokens=demo_cfg.max_num_batched_tokens,
            block_size=demo_cfg.block_size,
            available_kv_cache_memory_gb=demo_cfg.available_kv_cache_memory_gb,
            num_layers=demo_cfg.num_layers,
            kvpe_dim=demo_cfg.kvpe_dim,
            kv_cache_dtype_bytes=demo_cfg.kv_cache_dtype_bytes,
            kv_tensors_per_layer=demo_cfg.kv_tensors_per_layer,
        )
        
        model_config = ModelConfig(
            model_path=model_path,
            cache_dir=cache_dir,
            prefill_max_tokens=prefill_max_tokens,
            vocab_size=demo_cfg.vocab_size,
            block_size=demo_cfg.block_size,
        )
        
        # Optional: explicit KV cache configuration
        kv_cache_config: Optional[KvCacheConfig] = None
        if kv_cache_num_blocks is not None:
            kv_cache_config = KvCacheConfig(
                num_blocks=kv_cache_num_blocks,
                block_size=demo_cfg.block_size,
                kvpe_dim=demo_cfg.kvpe_dim,
                num_heads=1,
                dtype="bfloat8_b",
            )
            logging.info("Using explicit KV cache config: %s", kv_cache_config)
        
        # Create engine
        logging.info("Creating PrefillEngine...")
        engine = PrefillEngine(
            scheduler_config=sched_config,
            model_config=model_config,
            mesh_device=mesh_device,
            kv_cache_config=kv_cache_config,
        )
        
        # Get tokenizer from engine
        tokenizer = engine.tokenizer
        
        # Add all prompts to the queue
        logging.info("Adding prompts to engine...")
        for i, prompt in enumerate(prompts):
            # Tokenize using chat template
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            token_ids = list(token_ids)
            
            engine.add_request(token_ids, req_id=f"req_{i}")
            logging.info(f"Added: req_{i} - {len(token_ids)} tokens - '{prompt[:50]}...'")
        
        # Collect results
        results = {
            "prompts": prompts,
            "prompt_lengths": [],
            "first_tokens": [],
            "first_token_texts": [],
            "req_ids": [],
        }
        
        # Run prefills until queue is empty
        logging.info("Running prefills...")
        while not engine.is_finished():
            logits, scheduled = engine.step()
            
            if not scheduled:
                continue
            
            # Get prompt lengths for this batch
            batch_prompt_lens = [len(seq) for seq in scheduled]
            if prefill_max_tokens is not None:
                batch_prompt_lens = [
                    min(pl, prefill_max_tokens) for pl in batch_prompt_lens
                ]
            
            # Get last token logits for each sequence
            last_logits = get_last_token_logits(
                logits, batch_prompt_lens, prefill_max_tokens
            )
            
            # Greedy sample first tokens
            first_token_ids = greedy_sample(last_logits)
            
            # Decode tokens and store results
            for i, seq in enumerate(scheduled):
                token_id = first_token_ids[i].item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                
                results["req_ids"].append(seq.req_id)
                results["prompt_lengths"].append(batch_prompt_lens[i])
                results["first_tokens"].append(token_id)
                results["first_token_texts"].append(token_text)
                
                logging.info(
                    "Prefill complete: req_id=%s prompt_len=%d first_token=%d '%s'",
                    seq.req_id, batch_prompt_lens[i], token_id, token_text
                )
            
            # === DEMONSTRATE KV CACHE ACCESS ===
            print("\n" + "=" * 60)
            print("KV CACHE ACCESS DEMONSTRATION")
            print("=" * 60)
            
            # Get all KV cache tensors
            kv_cache = engine.get_kv_cache()
            print(f"Number of KV cache layers: {len(kv_cache)}")
            
            # Log info about first and last layer KV cache
            for layer_idx in [0, len(kv_cache) - 1]:
                kv_tensor = engine.get_kv_cache_for_layer(layer_idx)
                print(f"\nLayer {layer_idx} KV cache:")
                print(f"  Type: {type(kv_tensor).__name__}")
                if hasattr(kv_tensor, 'shape'):
                    print(f"  Shape: {kv_tensor.shape}")
                if hasattr(kv_tensor, 'dtype'):
                    print(f"  Dtype: {kv_tensor.dtype}")
            
            print("\n" + "=" * 60)
            print("KV cache is now ready for D2D transfer to decode node")
            print("=" * 60 + "\n")
            
            # Release blocks
            engine.release_after_prefill(scheduled)
        
        return results
        
    finally:
        # Cleanup
        logging.info("Cleaning up...")
        if engine is not None:
            try:
                engine.cleanup()
            except Exception as e:
                logging.warning(f"Failed to cleanup engine: {e}")
        
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s"
    )
    
    args = create_parser().parse_args()
    
    # Use provided prompts or defaults
    prompts = args.prompts if args.prompts else [
        "What is the capital of France?",
    ]
    
    print("=" * 60)
    print("DeepSeek V3 Prefill Demo - vLLM-like Interface")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Max sequences per batch: {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    if args.prefill_max_tokens:
        print(f"Max prefill tokens: {args.prefill_max_tokens}")
    if args.kv_cache_num_blocks:
        print(f"Explicit KV cache blocks: {args.kv_cache_num_blocks}")
    print("=" * 60)
    
    results = run_prefill_demo(
        prompts=prompts,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        prefill_max_tokens=args.prefill_max_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_num_blocks=args.kv_cache_num_blocks,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREFILL RESULTS - First Token for Each Prompt")
    print("=" * 60)
    
    for i in range(len(results["prompts"])):
        prompt = results["prompts"][i]
        req_id = results["req_ids"][i] if i < len(results.get("req_ids", [])) else f"req_{i}"
        prompt_len = results["prompt_lengths"][i] if i < len(results.get("prompt_lengths", [])) else "?"
        token_id = results["first_tokens"][i] if i < len(results.get("first_tokens", [])) else "?"
        token_text = results["first_token_texts"][i] if i < len(results.get("first_token_texts", [])) else "?"
        
        print(f"\n[{req_id}] ({prompt_len} tokens)")
        print(f"  Input: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"  First token ID: {token_id}")
        print(f"  First token: '{token_text}'")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

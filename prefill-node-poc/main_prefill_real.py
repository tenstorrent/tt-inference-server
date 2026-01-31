# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field

import torch
import ttnn

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "/data/dmadic/tt-metal")
if TT_METAL_HOME not in sys.path:
    sys.path.insert(0, TT_METAL_HOME)

from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

from model_runner import KvCacheConfig, ModelConfig
from prefill_engine import PrefillEngine
from scheduler import SchedulerConfig


@dataclass
class DemoConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 8192
    block_size: int = 32
    available_kv_cache_memory_gb: float = 12.0
    num_layers: int = 61
    kvpe_dim: int = 576
    kv_cache_dtype_bytes: int = 1
    kv_tensors_per_layer: int = 1
    vocab_size: int = 129280
    prompts: list[str] = field(default_factory=lambda: ["What is the capital of France?"])


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DeepSeek-V3 Prefill Demo")
    p.add_argument("prompts", type=str, nargs="*", default=[])
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--prefill-max-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--kv-cache-num-blocks", type=int, default=2048)
    return p


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def get_last_token_logits(
    logits: torch.Tensor,
    prompt_lens: list[int],
    prefill_max_tokens: int | None = None,
) -> torch.Tensor:
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
    kv_cache_num_blocks: int = 2048,
) -> dict:
    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    logging.info(f"MESH_DEVICE: '{requested_system_name}' - mesh shape: {mesh_shape}")
    
    fabric_config = ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)
    
    logging.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    
    engine = None
    try:
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
        
        kv_cache_config = KvCacheConfig(
            num_blocks=kv_cache_num_blocks,
            block_size=demo_cfg.block_size,
            kvpe_dim=demo_cfg.kvpe_dim,
            num_heads=1,
            dtype="bfloat8_b",
        )
        logging.info("KvCacheConfig: shape=%s dtype=%s", kv_cache_config.shape, kv_cache_config.dtype)
        
        logging.info("Creating PrefillEngine...")
        engine = PrefillEngine(
            scheduler_config=sched_config,
            model_config=model_config,
            mesh_device=mesh_device,
            kv_cache_config=kv_cache_config,
        )
        
        tokenizer = engine.tokenizer
        
        logging.info("Adding prompts...")
        for i, prompt in enumerate(prompts):
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            token_ids = list(token_ids)
            engine.add_request(token_ids, req_id=f"req_{i}")
            logging.info(f"Added: req_{i} - {len(token_ids)} tokens")
        
        results = {
            "prompts": prompts,
            "prompt_lengths": [],
            "first_tokens": [],
            "first_token_texts": [],
            "req_ids": [],
        }
        
        logging.info("Running prefills...")
        while not engine.is_finished():
            logits, scheduled = engine.step()
            
            if not scheduled:
                continue
            
            batch_prompt_lens = [len(seq) for seq in scheduled]
            if prefill_max_tokens is not None:
                batch_prompt_lens = [min(pl, prefill_max_tokens) for pl in batch_prompt_lens]
            
            last_logits = get_last_token_logits(logits, batch_prompt_lens, prefill_max_tokens)
            first_token_ids = greedy_sample(last_logits)
            
            for i, seq in enumerate(scheduled):
                token_id = first_token_ids[i].item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                
                results["req_ids"].append(seq.req_id)
                results["prompt_lengths"].append(batch_prompt_lens[i])
                results["first_tokens"].append(token_id)
                results["first_token_texts"].append(token_text)
                
                logging.info("Prefill: req_id=%s first_token=%d '%s'", seq.req_id, token_id, token_text)
            
            print("\n" + "=" * 60)
            print("KV CACHE ACCESS")
            print("=" * 60)
            kv_cache = engine.get_kv_cache()
            print(f"num_layers: {len(kv_cache)}")
            for layer_idx in [0, len(kv_cache) - 1]:
                kv = engine.get_kv_cache_for_layer(layer_idx)
                print(f"layer[{layer_idx}]: shape={kv.shape if hasattr(kv, 'shape') else 'N/A'}")
            print("=" * 60 + "\n")
            
            engine.release_after_prefill(scheduled)
        
        return results
        
    finally:
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
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = create_parser().parse_args()
    
    prompts = args.prompts if args.prompts else ["What is the capital of France?"]
    
    print("=" * 60)
    print("DeepSeek V3 Prefill Demo")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"KV cache blocks: {args.kv_cache_num_blocks}")
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
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for i in range(len(results["prompts"])):
        prompt = results["prompts"][i]
        req_id = results["req_ids"][i] if i < len(results["req_ids"]) else f"req_{i}"
        token_id = results["first_tokens"][i] if i < len(results["first_tokens"]) else "?"
        token_text = results["first_token_texts"][i] if i < len(results["first_token_texts"]) else "?"
        print(f"\n[{req_id}] {prompt[:80]}...")
        print(f"  First token: {token_id} '{token_text}'")
    print("=" * 60)


if __name__ == "__main__":
    main()

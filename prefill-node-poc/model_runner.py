# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import ttnn

logger = logging.getLogger(__name__)

from sequence import PrefillSequence
from timing import timed


@dataclass
class ModelConfig:
    model_path: str | Path
    cache_dir: str | Path
    prefill_max_tokens: int | None = None
    vocab_size: int = 129280
    block_size: int = 32
    pad_token_id: int = 0


@dataclass
class KvCacheConfig:
    num_blocks: int
    block_size: int
    kvpe_dim: int = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
    num_heads: int = 1
    dtype: str = "bfloat8_b"
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_blocks, self.num_heads, self.block_size, self.kvpe_dim)


class PrefillModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        mesh_device: ttnn.MeshDevice,
    ):
        self.model_config = model_config
        self.mesh_device = mesh_device
        self._current_seqs: List[PrefillSequence] = []
        self._generator = None
        self._tokenizer = None
        self._kv_cache: List[ttnn.Tensor] = None
        self._num_layers: int = 0

    def _initialize_model(self) -> None:
        if self._generator is not None:
            return
            
        TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "/data/dmadic/tt-metal")
        if TT_METAL_HOME not in sys.path:
            sys.path.insert(0, TT_METAL_HOME)
        
        from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
        from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
        
        cfg = self.model_config
        logger.info("=" * 60)
        logger.info("INITIALIZING DEEPSEEK GENERATOR")
        logger.info("=" * 60)
        
        logger.info("ModelConfig:")
        logger.info("  model_path: %s", cfg.model_path)
        logger.info("  cache_dir: %s", cfg.cache_dir)
        logger.info("  prefill_max_tokens: %s", cfg.prefill_max_tokens)
        logger.info("  vocab_size: %d", cfg.vocab_size)
        logger.info("  block_size: %d", cfg.block_size)
        logger.info("  pad_token_id: %d", cfg.pad_token_id)
        
        logger.info("-" * 40)
        logger.info("Loading tokenizer...")
        self._tokenizer = load_tokenizer(cfg.model_path)
        logger.info("Tokenizer loaded:")
        logger.info("  vocab_size: %d", self._tokenizer.vocab_size)
        logger.info("  pad_token_id: %s", self._tokenizer.pad_token_id)
        logger.info("  eos_token_id: %s", self._tokenizer.eos_token_id)
        logger.info("  bos_token_id: %s", self._tokenizer.bos_token_id)
        
        if self._tokenizer.pad_token_id is not None:
            self.model_config.pad_token_id = self._tokenizer.pad_token_id
            logger.info("  Updated pad_token_id from tokenizer: %d", self.model_config.pad_token_id)
        
        logger.info("-" * 40)
        logger.info("Creating DeepseekGenerator with arguments:")
        logger.info("  mesh_device: %s (shape=%s)", type(self.mesh_device).__name__, list(self.mesh_device.shape))
        logger.info("  model_path: %s", cfg.model_path)
        logger.info("  cache_dir: %s", cfg.cache_dir)
        logger.info("  tokenizer: %s", type(self._tokenizer).__name__)
        logger.info("  random_weights: False")
        logger.info("  enable_trace: False")
        logger.info("  prefill_max_tokens: %s", cfg.prefill_max_tokens)
        
        self._generator = DeepseekGenerator(
            mesh_device=self.mesh_device,
            model_path=cfg.model_path,
            cache_dir=cfg.cache_dir,
            tokenizer=self._tokenizer,
            random_weights=False,
            enable_trace=False,
            prefill_max_tokens=cfg.prefill_max_tokens,
        )
        
        self._num_layers = self._generator.hf_config.num_hidden_layers
        
        logger.info("-" * 40)
        logger.info("Generator created:")
        logger.info("  batch_size_per_row: %d", self._generator.batch_size_per_row)
        logger.info("  batch_size (total): %d", self._generator.batch_size)
        logger.info("  dp_factor: %d", self._generator.dp_factor)
        logger.info("  prefill_max_tokens: %s", self._generator.prefill_max_tokens)
        logger.info("  enable_trace: %s", self._generator.enable_trace)
        
        logger.info("-" * 40)
        logger.info("HF Config:")
        logger.info("  num_hidden_layers: %d", self._num_layers)
        logger.info("  hidden_size: %d", self._generator.hf_config.hidden_size)
        logger.info("  num_attention_heads: %d", self._generator.hf_config.num_attention_heads)
        logger.info("  vocab_size: %d", self._generator.hf_config.vocab_size)
        logger.info("  max_seq_len: %d", self._generator.hf_config.max_seq_len)
        if hasattr(self._generator.hf_config, 'kv_lora_rank'):
            logger.info("  kv_lora_rank: %d", self._generator.hf_config.kv_lora_rank)
        if hasattr(self._generator.hf_config, 'qk_rope_head_dim'):
            logger.info("  qk_rope_head_dim: %d", self._generator.hf_config.qk_rope_head_dim)
        
        logger.info("-" * 40)
        logger.info("Paged Config:")
        paged_cfg = self._generator.paged_config
        logger.info("  block_size: %d", paged_cfg.block_size)
        logger.info("  max_num_blocks: %d", paged_cfg.max_num_blocks)
        
        logger.info("=" * 60)
    
    @property
    def tokenizer(self):
        self._initialize_model()
        return self._tokenizer
    
    @property
    def num_layers(self) -> int:
        self._initialize_model()
        return self._num_layers
    
    @property
    def kv_cache(self) -> List[ttnn.Tensor]:
        return self._kv_cache

    @timed()
    def allocate_kv_cache(
        self,
        kv_cache_config: KvCacheConfig,
    ) -> List[ttnn.Tensor]:
        self._initialize_model()
        
        assert kv_cache_config is not None, "kv_cache_config is required (like vLLM interface)"
        assert self._num_layers > 0, "Model must be initialized before allocating KV cache"
        
        logger.info("=" * 60)
        logger.info("ALLOCATING KV CACHE")
        logger.info("  shape: %s dtype: %s", kv_cache_config.shape, kv_cache_config.dtype)
        
        from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig as TtKvCacheConfig
        
        tt_kv_cache_config = TtKvCacheConfig(
            kv_cache_shape=kv_cache_config.shape,
            dtype=getattr(ttnn, kv_cache_config.dtype, ttnn.bfloat8_b),
        )
        
        self._generator._prepare_run_configs("prefill", kv_cache_override=tt_kv_cache_config)
        self._kv_cache = self._generator.get_kv_cache()
        
        logger.info("  num_layers: %d", len(self._kv_cache))
        if self._kv_cache:
            logger.info("  per_layer_shape: %s", self._kv_cache[0].shape if hasattr(self._kv_cache[0], 'shape') else "N/A")
        logger.info("=" * 60)
        
        return self._kv_cache
    
    def get_kv_cache(self) -> List[ttnn.Tensor]:
        if self._kv_cache is None:
            raise RuntimeError("KV cache not allocated. Call allocate_kv_cache() first.")
        return self._kv_cache
    
    def get_kv_cache_for_layer(self, layer_idx: int) -> ttnn.Tensor:
        kv_cache = self.get_kv_cache()
        if layer_idx < 0 or layer_idx >= len(kv_cache):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(kv_cache)})")
        return kv_cache[layer_idx]

    @timed()
    def run_prefill(self, seqs: List[PrefillSequence]) -> torch.Tensor:
        if not seqs:
            return torch.empty(0, 0, self.model_config.vocab_size)

        if self._kv_cache is None:
            raise RuntimeError("KV cache not allocated. Call allocate_kv_cache() first.")

        logger.info("=" * 60)
        logger.info("RUN_PREFILL: num_seqs=%d req_ids=%s", len(seqs), [s.req_id for s in seqs])
        
        self._current_seqs = seqs
        
        try:
            prompt_lens = [len(seq) for seq in seqs]
            max_prompt_len = max(prompt_lens)
            
            cfg = self.model_config
            if cfg.prefill_max_tokens is not None:
                max_prompt_len = min(max_prompt_len, cfg.prefill_max_tokens)
                prompt_lens = [min(pl, cfg.prefill_max_tokens) for pl in prompt_lens]
            
            max_padded_len = ((max_prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            
            all_logits = []
            
            # Build page_table from all sequences' block tables
            # Shape: [batch_size, max_num_blocks_per_req]
            max_blocks_per_req = max(len(seq.block_table) for seq in seqs)
            page_table = torch.zeros((len(seqs), max_blocks_per_req), dtype=torch.int32)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table
                page_table[i, :len(block_table)] = torch.tensor(block_table, dtype=torch.int32)
            
            logger.info("Page table created:")
            logger.info("  shape: %s", tuple(page_table.shape))
            logger.info("  block_tables: %s", [seq.block_table for seq in seqs])
            
            # Sample KV cache block values BEFORE prefill
            block_ids_to_check = list(set(b for seq in seqs for b in seq.block_table))
            
            def get_kv_block_sample(block_id: int, layer_idx: int = 0) -> torch.Tensor:
                """Read a small sample from KV cache block."""
                kv_tensor = self._kv_cache[layer_idx]
                kv_host = ttnn.to_torch(
                    kv_tensor,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        self.mesh_device, dims=(0, -1), mesh_shape=self.mesh_device.shape
                    ),
                )
                # Shape: [num_blocks, num_heads, block_size, kvpe_dim]
                # Return first 8 values from the block
                return kv_host[block_id, 0, 0, :8].float()
            
            logger.info("-" * 40)
            logger.info("KV CACHE BLOCK VALUES BEFORE PREFILL (layer 0, first 8 values):")
            pre_values = {}
            for block_id in sorted(block_ids_to_check):
                vals = get_kv_block_sample(block_id, layer_idx=0)
                pre_values[block_id] = vals.clone()
                logger.info("  block[%d]: %s", block_id, vals.tolist())
            
            for i, seq in enumerate(seqs):
                prompt_len = prompt_lens[i]
                token_ids = seq.token_ids[:prompt_len]
                
                pad_len = ((prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                tokens_padded = token_ids + [cfg.pad_token_id] * (pad_len - prompt_len)
                tokens_tensor = torch.tensor(tokens_padded, dtype=torch.long)
                
                logger.info("-" * 40)
                logger.info("PREFILL user_id=%d", i)
                logger.info("  req_id: %s", seq.req_id)
                logger.info("  prompt_len: %d", prompt_len)
                logger.info("  padded_len: %d", pad_len)
                logger.info("  block_table: %s", seq.block_table)
                logger.info("  first_10_tokens: %s", token_ids[:10])
                logger.info("  last_10_tokens: %s", token_ids[-10:] if len(token_ids) >= 10 else token_ids)
                logger.info("Calling generator._prefill():")
                logger.info("  tokens: shape=%s dtype=%s", tuple(tokens_tensor.shape), tokens_tensor.dtype)
                logger.info("  user_id: %d", i)
                logger.info("  page_table: shape=%s user_row=%s", tuple(page_table.shape), page_table[i].tolist())
                logger.info("  local_user_id: %d", i)
                
                logits = self._generator._prefill(
                    tokens_tensor,
                    user_id=i,
                    page_table=page_table,
                    local_user_id=i,
                )
                
                logger.info("Output: logits.shape=%s", tuple(logits.shape))
                
                user_logits = logits[0, 0, :prompt_len, :]
                
                last_logits = user_logits[-1, :]
                top5_values, top5_indices = torch.topk(last_logits, 5)
                logger.info("Top-5 tokens at last position:")
                for j, (val, idx) in enumerate(zip(top5_values, top5_indices)):
                    token_text = self._tokenizer.decode([idx.item()], skip_special_tokens=False) if self._tokenizer else "?"
                    logger.info("  [%d] token_id=%d logit=%.4f text='%s'", j+1, idx.item(), val.item(), token_text[:50])
                
                if user_logits.shape[0] < max_padded_len:
                    pad_size = max_padded_len - user_logits.shape[0]
                    pad_logits = user_logits[-1:].expand(pad_size, -1)
                    user_logits = torch.cat([user_logits, pad_logits], dim=0)
                
                all_logits.append(user_logits)
                self._generator.ccl.reset_sem_counters()
            
            result = torch.stack(all_logits, dim=0)
            
            # Sample KV cache block values AFTER prefill
            logger.info("-" * 40)
            logger.info("KV CACHE BLOCK VALUES AFTER PREFILL (layer 0, first 8 values):")
            for block_id in sorted(block_ids_to_check):
                vals = get_kv_block_sample(block_id, layer_idx=0)
                pre_vals = pre_values.get(block_id)
                changed = not torch.allclose(vals, pre_vals, atol=1e-6) if pre_vals is not None else "N/A"
                logger.info("  block[%d]: %s", block_id, vals.tolist())
                logger.info("    CHANGED: %s (was: %s)", changed, pre_vals.tolist() if pre_vals is not None else "N/A")
            
            logger.info("RUN_PREFILL COMPLETE: logits_shape=%s", tuple(result.shape))
            logger.info("=" * 60)
            
            return result
            
        finally:
            self._current_seqs = []

    def cleanup(self) -> None:
        if self._generator is not None:
            try:
                self._generator.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup generator: %s", e)
        self._generator = None
        self._tokenizer = None
        self._kv_cache = None
        logger.info("model_runner cleanup")

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
        logger.info("ModelConfig: model_path=%s cache_dir=%s", cfg.model_path, cfg.cache_dir)
        
        self._tokenizer = load_tokenizer(cfg.model_path)
        logger.info("Tokenizer: vocab_size=%d pad_token_id=%s", 
                   self._tokenizer.vocab_size, self._tokenizer.pad_token_id)
        
        if self._tokenizer.pad_token_id is not None:
            self.model_config.pad_token_id = self._tokenizer.pad_token_id
        
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
        
        logger.info("Generator: batch_size=%d num_layers=%d", 
                   self._generator.batch_size, self._num_layers)
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
            
            for i, seq in enumerate(seqs):
                prompt_len = prompt_lens[i]
                token_ids = seq.token_ids[:prompt_len]
                
                pad_len = ((prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                tokens_padded = token_ids + [cfg.pad_token_id] * (pad_len - prompt_len)
                tokens_tensor = torch.tensor(tokens_padded, dtype=torch.long)
                
                logger.info("  user=%d prompt_len=%d padded_len=%d", i, prompt_len, pad_len)
                
                logits = self._generator._prefill(tokens_tensor, user_id=i)
                
                user_logits = logits[0, 0, :prompt_len, :]
                
                if user_logits.shape[0] < max_padded_len:
                    pad_size = max_padded_len - user_logits.shape[0]
                    pad_logits = user_logits[-1:].expand(pad_size, -1)
                    user_logits = torch.cat([user_logits, pad_logits], dim=0)
                
                all_logits.append(user_logits)
                self._generator.ccl.reset_sem_counters()
            
            result = torch.stack(all_logits, dim=0)
            
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

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model runner that bridges the prefill scheduler and DeepseekGenerator.

Uses DeepseekGenerator from tt-metal for actual model inference.
Works like vLLM interface with explicit KV cache allocation and access.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

import ttnn

logger = logging.getLogger(__name__)

from sequence import PrefillSequence
from timing import timed


@dataclass
class ModelConfig:
    """Configuration for the DeepSeek model."""
    
    model_path: str | Path
    cache_dir: str | Path
    prefill_max_tokens: int | None = None
    vocab_size: int = 129280
    block_size: int = 32
    pad_token_id: int = 0


@dataclass
class KvCacheConfig:
    """
    KV cache configuration (mirrors vLLM interface).
    
    Shape: (num_blocks, num_heads=1, block_size, kvpe_dim)
    For DeepSeek V3 with MLA: kvpe_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
    """
    num_blocks: int
    block_size: int
    kvpe_dim: int = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
    num_heads: int = 1   # MLA uses single head for compressed KV
    dtype: str = "bfloat8_b"
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return KV cache shape: (num_blocks, num_heads, block_size, kvpe_dim)."""
        return (self.num_blocks, self.num_heads, self.block_size, self.kvpe_dim)


class PrefillModelRunner:
    """
    Runs prefill for batches of sequences produced by PrefillScheduler.
    Uses DeepseekGenerator from tt-metal for actual model inference.
    
    Works like vLLM interface:
    - Explicit KV cache allocation with configurable shape
    - Access to KV cache tensors after prefill (for D2D transfer)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mesh_device: ttnn.MeshDevice,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ):
        self.model_config = model_config
        self.mesh_device = mesh_device
        self.kv_cache_config = kv_cache_config
        self._current_seqs: List[PrefillSequence] = []
        
        # Model components (initialized lazily)
        self._generator = None
        self._tokenizer = None
        
        # KV cache tensors (populated after allocate_kv_cache)
        self._kv_cache: Optional[List[ttnn.Tensor]] = None
        self._num_layers: int = 0

    def _initialize_model(self) -> None:
        """Initialize the DeepseekGenerator (lazy initialization)."""
        if self._generator is not None:
            return
            
        # Add tt-metal to path for imports
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
        logger.info("Loading tokenizer from %s", cfg.model_path)
        self._tokenizer = load_tokenizer(cfg.model_path)
        logger.info("Tokenizer loaded:")
        logger.info("  vocab_size: %d", self._tokenizer.vocab_size)
        logger.info("  pad_token_id: %s", self._tokenizer.pad_token_id)
        logger.info("  eos_token_id: %s", self._tokenizer.eos_token_id)
        logger.info("  bos_token_id: %s", self._tokenizer.bos_token_id)
        
        # Update pad_token_id from tokenizer if available
        if self._tokenizer.pad_token_id is not None:
            self.model_config.pad_token_id = self._tokenizer.pad_token_id
            logger.info("  Updated pad_token_id from tokenizer: %d", self.model_config.pad_token_id)
        
        logger.info("-" * 40)
        logger.info("Creating DeepseekGenerator...")
        self._generator = DeepseekGenerator(
            mesh_device=self.mesh_device,
            model_path=cfg.model_path,
            cache_dir=cfg.cache_dir,
            tokenizer=self._tokenizer,
            random_weights=False,
            enable_trace=False,
            prefill_max_tokens=cfg.prefill_max_tokens,
        )
        
        # Store num_layers for KV cache
        self._num_layers = self._generator.hf_config.num_hidden_layers
        
        # Log generator configuration
        logger.info("-" * 40)
        logger.info("Generator Configuration:")
        logger.info("  mesh_device.shape: %s", list(self.mesh_device.shape))
        logger.info("  batch_size_per_row: %d", self._generator.batch_size_per_row)
        logger.info("  batch_size (total): %d", self._generator.batch_size)
        logger.info("  dp_factor: %d", self._generator.dp_factor)
        logger.info("  prefill_max_tokens: %s", self._generator.prefill_max_tokens)
        logger.info("  enable_trace: %s", self._generator.enable_trace)
        
        # Log HF config
        logger.info("-" * 40)
        logger.info("HF Config (from model):")
        logger.info("  num_hidden_layers: %d", self._num_layers)
        logger.info("  hidden_size: %d", self._generator.hf_config.hidden_size)
        logger.info("  num_attention_heads: %d", self._generator.hf_config.num_attention_heads)
        logger.info("  vocab_size: %d", self._generator.hf_config.vocab_size)
        logger.info("  max_seq_len: %d", self._generator.hf_config.max_seq_len)
        if hasattr(self._generator.hf_config, 'kv_lora_rank'):
            logger.info("  kv_lora_rank: %d", self._generator.hf_config.kv_lora_rank)
        if hasattr(self._generator.hf_config, 'qk_rope_head_dim'):
            logger.info("  qk_rope_head_dim: %d", self._generator.hf_config.qk_rope_head_dim)
        if hasattr(self._generator.hf_config, 'first_k_dense_replace'):
            logger.info("  first_k_dense_replace (dense layers): %d", self._generator.hf_config.first_k_dense_replace)
        
        # Log paged attention config (internal default)
        logger.info("-" * 40)
        logger.info("Paged Attention Config (internal paged_config):")
        paged_cfg = self._generator.paged_config
        logger.info("  block_size: %d", paged_cfg.block_size)
        logger.info("  max_num_blocks: %d", paged_cfg.max_num_blocks)
        
        logger.info("=" * 60)
        logger.info("GENERATOR CREATED (KV cache not yet allocated)")
        logger.info("=" * 60)
    
    @property
    def tokenizer(self):
        """Access the tokenizer."""
        self._initialize_model()
        return self._tokenizer
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        self._initialize_model()
        return self._num_layers
    
    @property
    def kv_cache(self) -> Optional[List[ttnn.Tensor]]:
        """Access the allocated KV cache tensors (one per layer)."""
        return self._kv_cache

    @timed()
    def allocate_kv_cache(
        self,
        kv_cache_config: Optional[KvCacheConfig] = None,
    ) -> List[ttnn.Tensor]:
        """
        Allocate KV cache with explicit shape control (like vLLM interface).
        
        Args:
            kv_cache_config: Optional override for KV cache configuration.
                            If None, uses self.kv_cache_config or internal defaults.
        
        Returns:
            List of KV cache tensors, one per layer.
        """
        self._initialize_model()
        
        # Use provided config, instance config, or None for internal defaults
        cfg = kv_cache_config or self.kv_cache_config
        
        logger.info("=" * 60)
        logger.info("ALLOCATING KV CACHE")
        logger.info("=" * 60)
        
        if cfg is not None:
            # Use explicit KV cache shape (like vLLM)
            logger.info("Using explicit KV cache configuration:")
            logger.info("  num_blocks: %d", cfg.num_blocks)
            logger.info("  num_heads: %d", cfg.num_heads)
            logger.info("  block_size: %d", cfg.block_size)
            logger.info("  kvpe_dim: %d", cfg.kvpe_dim)
            logger.info("  dtype: %s", cfg.dtype)
            logger.info("  shape: %s", cfg.shape)
            
            # Import KvCacheConfig from tt-metal
            from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig as TtKvCacheConfig
            
            # Create tt-metal KvCacheConfig
            tt_kv_cache_config = TtKvCacheConfig(
                kv_cache_shape=cfg.shape,
                dtype=getattr(ttnn, cfg.dtype, ttnn.bfloat8_b),
            )
            
            logger.info("Calling _prepare_run_configs('prefill') with kv_cache_override...")
            self._generator._prepare_run_configs("prefill", kv_cache_override=tt_kv_cache_config)
        else:
            # Use internal defaults
            logger.info("Using internal KV cache configuration (from paged_config)")
            logger.info("Calling _prepare_run_configs('prefill') without override...")
            self._generator._prepare_run_configs("prefill")
        
        # Get KV cache tensors from generator
        logger.info("-" * 40)
        logger.info("Retrieving KV cache tensors from generator...")
        self._kv_cache = self._generator.get_kv_cache()
        
        logger.info("KV Cache Allocated:")
        logger.info("  num_layers: %d", len(self._kv_cache))
        
        if self._kv_cache:
            first_cache = self._kv_cache[0]
            logger.info("  Per-layer KV cache tensor:")
            logger.info("    type: %s", type(first_cache).__name__)
            if hasattr(first_cache, 'shape'):
                logger.info("    shape: %s", first_cache.shape)
            if hasattr(first_cache, 'dtype'):
                logger.info("    dtype: %s", first_cache.dtype)
        
        logger.info("=" * 60)
        logger.info("KV CACHE ALLOCATION COMPLETE")
        logger.info("=" * 60)
        
        return self._kv_cache
    
    def get_kv_cache(self) -> List[ttnn.Tensor]:
        """
        Get the KV cache tensors (like vLLM interface).
        
        Returns:
            List of KV cache tensors, one per layer.
            
        Raises:
            RuntimeError if KV cache not yet allocated.
        """
        if self._kv_cache is None:
            raise RuntimeError("KV cache not allocated. Call allocate_kv_cache() first.")
        return self._kv_cache
    
    def get_kv_cache_for_layer(self, layer_idx: int) -> ttnn.Tensor:
        """
        Get KV cache tensor for a specific layer.
        
        Args:
            layer_idx: Layer index (0 to num_layers-1)
            
        Returns:
            KV cache tensor for the specified layer.
        """
        kv_cache = self.get_kv_cache()
        if layer_idx < 0 or layer_idx >= len(kv_cache):
            raise IndexError(f"Layer index {layer_idx} out of range [0, {len(kv_cache)})")
        return kv_cache[layer_idx]

    @timed()
    def run_prefill(self, seqs: List[PrefillSequence]) -> torch.Tensor:
        """
        Run prefill for the given scheduled sequences.
        
        After prefill, KV cache is updated and can be accessed via get_kv_cache().
        
        Returns:
            logits: [batch_size, max_padded_len, vocab_size]
        """
        if not seqs:
            return torch.empty(0, 0, self.model_config.vocab_size)

        # Ensure KV cache is allocated
        if self._kv_cache is None:
            logger.info("KV cache not allocated, allocating with defaults...")
            self.allocate_kv_cache()

        logger.info("=" * 60)
        logger.info("RUN_PREFILL START")
        logger.info("=" * 60)
        logger.info("Batch info:")
        logger.info("  num_seqs: %d", len(seqs))
        logger.info("  req_ids: %s", [s.req_id for s in seqs])
        
        self._current_seqs = seqs
        
        try:
            # Get prompt lengths
            prompt_lens = [len(seq) for seq in seqs]
            max_prompt_len = max(prompt_lens)
            
            logger.info("  prompt_lengths: %s", prompt_lens)
            logger.info("  max_prompt_len: %d", max_prompt_len)
            
            # Apply prefill_max_tokens limit if set
            cfg = self.model_config
            if cfg.prefill_max_tokens is not None:
                logger.info("  prefill_max_tokens limit: %d", cfg.prefill_max_tokens)
                max_prompt_len = min(max_prompt_len, cfg.prefill_max_tokens)
                prompt_lens = [min(pl, cfg.prefill_max_tokens) for pl in prompt_lens]
                logger.info("  adjusted prompt_lengths: %s", prompt_lens)
                logger.info("  adjusted max_prompt_len: %d", max_prompt_len)
            
            # Calculate padded length (multiple of TILE_SIZE)
            max_padded_len = ((max_prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            logger.info("  TILE_SIZE: %d", ttnn.TILE_SIZE)
            logger.info("  max_padded_len: %d", max_padded_len)
            
            all_logits = []
            
            for i, seq in enumerate(seqs):
                prompt_len = prompt_lens[i]
                
                logger.info("-" * 40)
                logger.info("PREFILL USER %d", i)
                logger.info("-" * 40)
                logger.info("  req_id: %s", seq.req_id)
                logger.info("  user_id: %d", i)
                logger.info("  original_prompt_len: %d", len(seq.token_ids))
                logger.info("  effective_prompt_len: %d", prompt_len)
                logger.info("  block_table: %s", seq.block_table[:10] if len(seq.block_table) > 10 else seq.block_table)
                logger.info("  num_blocks: %d", seq.num_blocks)
                
                # Get tokens (potentially truncated)
                token_ids = seq.token_ids[:prompt_len]
                
                # Log first and last tokens
                logger.info("  first_10_tokens: %s", token_ids[:10])
                logger.info("  last_10_tokens: %s", token_ids[-10:] if len(token_ids) >= 10 else token_ids)
                
                # Decode first/last tokens for readability
                if self._tokenizer:
                    first_text = self._tokenizer.decode(token_ids[:10], skip_special_tokens=False)
                    last_text = self._tokenizer.decode(token_ids[-10:] if len(token_ids) >= 10 else token_ids, skip_special_tokens=False)
                    logger.info("  first_10_decoded: '%s'", first_text[:100])
                    logger.info("  last_10_decoded: '%s'", last_text[:100])
                
                # Pad to tile size
                pad_len = ((prompt_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                tokens_padded = token_ids + [cfg.pad_token_id] * (pad_len - prompt_len)
                tokens_tensor = torch.tensor(tokens_padded, dtype=torch.long)
                
                logger.info("  padded_len: %d", pad_len)
                logger.info("  num_pad_tokens: %d", pad_len - prompt_len)
                logger.info("  tokens_tensor.shape: %s", tuple(tokens_tensor.shape))
                logger.info("  tokens_tensor.dtype: %s", tokens_tensor.dtype)
                
                # Log what we're passing to generator._prefill()
                logger.info("-" * 20)
                logger.info("Calling generator._prefill():")
                logger.info("  tokens: torch.Tensor shape=%s dtype=%s", 
                           tuple(tokens_tensor.shape), tokens_tensor.dtype)
                logger.info("  user_id: %d", i)
                logger.info("  page_table: None (using internal default)")
                logger.info("  local_user_id: None")
                
                # Run prefill on model
                logits = self._generator._prefill(tokens_tensor, user_id=i)
                # logits shape: [1, 1, seq_len, vocab_size]
                
                logger.info("-" * 20)
                logger.info("Prefill output:")
                logger.info("  logits.shape: %s", tuple(logits.shape))
                logger.info("  logits.dtype: %s", logits.dtype)
                
                # Extract logits up to prompt_len, then pad to max_padded_len
                user_logits = logits[0, 0, :prompt_len, :]  # [prompt_len, vocab_size]
                logger.info("  extracted user_logits.shape: %s", tuple(user_logits.shape))
                
                # Log top-5 predicted tokens at last position
                last_logits = user_logits[-1, :]  # [vocab_size]
                top5_values, top5_indices = torch.topk(last_logits, 5)
                logger.info("  Top-5 tokens at last position:")
                for j, (val, idx) in enumerate(zip(top5_values, top5_indices)):
                    token_text = self._tokenizer.decode([idx.item()], skip_special_tokens=False) if self._tokenizer else "?"
                    logger.info("    [%d] token_id=%d logit=%.4f text='%s'", 
                               j+1, idx.item(), val.item(), token_text[:50])
                
                if user_logits.shape[0] < max_padded_len:
                    # Pad with last logits repeated
                    pad_size = max_padded_len - user_logits.shape[0]
                    pad_logits = user_logits[-1:].expand(pad_size, -1)
                    user_logits = torch.cat([user_logits, pad_logits], dim=0)
                    logger.info("  padded user_logits.shape: %s (added %d rows)", 
                               tuple(user_logits.shape), pad_size)
                
                all_logits.append(user_logits)
                
                # Reset CCL counters for next iteration
                self._generator.ccl.reset_sem_counters()
                logger.info("  CCL counters reset")
            
            # Stack all logits: [batch_size, max_padded_len, vocab_size]
            result = torch.stack(all_logits, dim=0)
            
            # Log KV cache state after prefill
            logger.info("-" * 40)
            logger.info("KV Cache after prefill:")
            logger.info("  num_layers with KV cache: %d", len(self._kv_cache) if self._kv_cache else 0)
            logger.info("  KV cache is accessible via get_kv_cache()")
            
            logger.info("=" * 60)
            logger.info("RUN_PREFILL COMPLETE")
            logger.info("  final_logits.shape: %s", tuple(result.shape))
            logger.info("=" * 60)
            
            return result
            
        finally:
            self._current_seqs = []

    def cleanup(self) -> None:
        """Release all resources."""
        if self._generator is not None:
            try:
                self._generator.cleanup_all()
            except Exception as e:
                logger.warning("Failed to cleanup generator: %s", e)
        self._generator = None
        self._tokenizer = None
        self._kv_cache = None
        logger.info("model_runner cleanup")

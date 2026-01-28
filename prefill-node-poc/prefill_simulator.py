# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union, Protocol
from enum import Enum

import torch


class KVCacheState(Enum):
    """State of a KV cache layer."""

    UNALLOCATED = "unallocated"
    ALLOCATED = "allocated"
    FILLED = "filled"
    STREAMED = "streamed"


@dataclass
class KVCacheReference:
    """
    Reference to a KV cache tensor for a single attention layer.

    This is passed to the callback when KV cache is ready for streaming.
    The consumer can use this reference to initiate D2D transfer.

    In batch-layer mode (layer-by-layer for the whole batch), when the callback
    is invoked the ref's tensor covers the full layer for the entire batch;
    user_id is not used for the "ready" signal.

    Attributes:
        layer_idx: The transformer layer index this cache belongs to
        tensor: The actual KV cache tensor on device (ttnn.Tensor or mock)
        shape: Shape of the KV cache (num_blocks, num_heads, block_size, kvpe_dim)
        dtype: Data type of the cache
        state: Current state of the cache
        seq_len: Sequence length that was filled (batch-layer: not per-user)
        user_id: Deprecated in batch-layer mode; ref = full layer for batch
    """

    layer_idx: int
    tensor: object  # ttnn.Tensor in real implementation
    shape: Tuple[int, ...]
    dtype: str
    state: KVCacheState
    seq_len: int = 0
    user_id: int = 0

    def mark_streamed(self) -> None:
        """Mark this cache as having been streamed to decode node."""
        self.state = KVCacheState.STREAMED


class KVCacheCallback(Protocol):
    """
    Protocol for KV cache ready callback.

    In batch-layer mode the callback is invoked once per layer when that layer
    is filled for the whole batch. kv_cache_ref.tensor covers the full layer;
    user_id on the ref is not used.
    """

    def __call__(self, layer_idx: int, kv_cache_ref: KVCacheReference) -> None:
        """Called when KV cache for a layer is ready for streaming (whole batch)."""
        ...


@dataclass
class PrefillConfig:
    """
    Configuration for the DeepSeek prefill simulator.

    Based on DeepSeek V3 671B architecture defaults.
    """

    num_layers: int = 61
    num_dense_layers: int = 1
    hidden_size: int = 7168
    num_attention_heads: int = 128
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    max_seq_len: int = 4096
    batch_size: int = 32
    block_size: int = 32
    vocab_size: int = 129280
    pad_token_id: int = 0

    @property
    def kvpe_dim(self) -> int:
        """KV + PE dimension for MLA."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def max_num_blocks(self) -> int:
        """Maximum number of blocks for paged attention."""
        return (self.max_seq_len * self.batch_size) // self.block_size

    def get_kv_cache_shape(self, dp_factor: int = 1) -> Tuple[int, int, int, int]:
        """Get the shape of KV cache for a single layer."""
        return (
            self.max_num_blocks * dp_factor,
            1,
            self.block_size,
            self.kvpe_dim,
        )


def _pad_tokens(
    tokens: torch.Tensor,
    pad_value: int = 0,
    block_size: int = 32,
) -> torch.Tensor:
    """Pad tokens to the nearest multiple of block_size. Same as generator_vllm."""
    batch_size, seq_len = tokens.shape
    padded_len = ((seq_len + block_size - 1) // block_size) * block_size
    if padded_len == seq_len:
        return tokens
    padded_tokens = torch.full(
        (batch_size, padded_len), pad_value, dtype=tokens.dtype, device=tokens.device
    )
    padded_tokens[:, :seq_len] = tokens
    return padded_tokens


@dataclass
class PrefillResult:
    """Result of a prefill forward pass."""

    logits: torch.Tensor
    seq_len: int
    user_id: int
    kv_cache_refs: List[KVCacheReference] = field(default_factory=list)


class DeepSeekPrefillSimulator:
    """
    Minimal DeepSeek V3 model simulator focused on prefill phase.

    This simulator models the prefill behavior of DeepSeek V3 with emphasis on:
    1. KV cache allocation on device
    2. Layer-by-layer prefill with callbacks for KV cache streaming
    3. Support for disaggregated P/D (Prefill/Decode) setup
    """

    def __init__(
        self,
        config: PrefillConfig,
        mesh_device: object = None,
        on_kv_cache_ready: Optional[KVCacheCallback] = None,
        dp_factor: int = 1,
    ):
        """
        Initialize the prefill simulator.

        Args:
            config: Model configuration
            mesh_device: TTNN mesh device (or None for simulation)
            on_kv_cache_ready: Callback invoked when each layer's KV cache is filled
            dp_factor: Data parallelism factor
        """
        self.config = config
        self.mesh_device = mesh_device
        self.on_kv_cache_ready = on_kv_cache_ready
        self.dp_factor = dp_factor

        self._kv_caches: List[KVCacheReference] = []
        self._is_initialized = False

    def allocate_kv_cache(
        self,
        dtype: str = "bfloat8_b",
    ) -> List[KVCacheReference]:
        """
        Allocate KV cache on device for all layers.

        Args:
            dtype: Data type for cache tensors

        Returns:
            List of KVCacheReference for all layers
        """
        cache_shape = self.config.get_kv_cache_shape(self.dp_factor)

        self._kv_caches = []
        for layer_idx in range(self.config.num_layers):
            if self.mesh_device is not None:
                tensor = self._allocate_device_tensor(cache_shape, dtype)
            else:
                tensor = torch.zeros(cache_shape)

            cache_ref = KVCacheReference(
                layer_idx=layer_idx,
                tensor=tensor,
                shape=cache_shape,
                dtype=dtype,
                state=KVCacheState.ALLOCATED,
            )
            self._kv_caches.append(cache_ref)

        self._is_initialized = True
        return self._kv_caches

    def _allocate_device_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> object:
        """Allocate a tensor on device."""
        try:
            import ttnn

            cache_tensor = ttnn.as_tensor(
                torch.zeros(shape),
                dtype=getattr(ttnn, dtype, ttnn.bfloat8_b),
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return cache_tensor
        except ImportError:
            return torch.zeros(shape)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        prompt_lens: Union[torch.Tensor, List[int]],
        page_table: torch.Tensor,
        kv_cache: List[object],
        start_pos: torch.Tensor,
        *,
        enable_trace: bool = False,  # Runner always passes; simulator ignores (prefill tracing not supported).
        empty_slots: Optional[
            Union[torch.Tensor, List[int]]
        ] = None,  # Runner passes only when DP > 1; simulator ignores, uses batch index as user_id.
        **kwargs,  # multi_modal (pixel_values, image_grid_thw, etc.); runner may pass, simulator ignores.
    ) -> torch.Tensor:
        """
        Run prefill forward pass. Interface matches vllm/v1/worker/tt_model_runner.py.

        Mandatory (runner always passes):
            tokens: Input token IDs [batch_size, max_prompt_tokens], padded.
            prompt_lens: Length of each prompt in the batch; shape [batch_size]. One int per request:
                the actual number of tokens before padding. E.g. [128, 64, 200] means request 0 has 128
                real tokens, request 1 has 64, request 2 has 200. Used to slice tokens per request and
                to trim/pad logits. Runner passes numpy or list; tensor also accepted.
            page_table: Block tables for paged KV cache [batch_size, max_num_blocks_per_req].
            kv_cache: List of KV cache tensors (one per layer); runner passes allocated caches.
            start_pos: Per-request start positions [batch_size]; must be all zeros (prefix caching not supported).

        Optional (runner may pass; simulator ignores where noted):
            enable_trace, empty_slots, **kwargs (multi_modal).

        Returns:
            last_logits: [num_of_users, max_padded_len, vocab_size], same as vLLM generator.
        """
        if not all(x == 0 for x in start_pos):
            raise AssertionError(
                f"Prefix caching is not supported, got start_pos: {start_pos}"
            )

        # Runner may pass prompt_lens as numpy or list; normalize for iteration/max.
        lengths = prompt_lens
        if hasattr(lengths, "tolist"):
            lengths_list = lengths.tolist()
        else:
            lengths_list = list(lengths)
        if not self._is_initialized:
            raise RuntimeError(
                "KV cache not allocated. Call allocate_kv_cache() first."
            )

        if all(length == 0 for length in lengths_list):
            return torch.zeros(
                tokens.shape[0],
                self.config.vocab_size,
                device=tokens.device,
                dtype=tokens.dtype,
            )

        if not any(entry is None for entry in kv_cache):
            self._set_kv_cache_from_list(kv_cache)

        pad_value = self.config.pad_token_id
        pad_block_size = self.config.block_size
        max_prompt_len = int(max(lengths_list)) if lengths_list else 0
        max_padded_len = (
            ((max_prompt_len + pad_block_size - 1) // pad_block_size) * pad_block_size
            if max_prompt_len > 0
            else 0
        )
        num_of_users = tokens.shape[0]

        tokens_padded = _pad_tokens(
            tokens[:, :max_prompt_len], pad_value=pad_value, block_size=pad_block_size
        )
        if tokens_padded.shape[1] < max_padded_len:
            padding = torch.full(
                (num_of_users, max_padded_len - tokens_padded.shape[1]),
                pad_value,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            tokens_padded = torch.cat([tokens_padded, padding], dim=1)
        else:
            tokens_padded = tokens_padded[:, :max_padded_len]

        hidden_states = self._simulate_embedding_batch(tokens_padded)

        for layer_idx in range(self.config.num_layers):
            hidden_states = self._simulate_transformer_layer_batch(
                hidden_states, layer_idx
            )
            cache_ref = self._kv_caches[layer_idx]
            self._fill_kv_cache_batch(cache_ref, lengths_list)
            if self.on_kv_cache_ready is not None:
                self.on_kv_cache_ready(layer_idx, cache_ref)

        logits_batch = self._simulate_lm_head_batch(hidden_states)

        last_logits = []
        for i in range(num_of_users):
            prompt_len = int(lengths_list[i])
            if prompt_len == 0:
                last_logits.append(
                    torch.zeros(
                        max_padded_len,
                        self.config.vocab_size,
                        device=tokens.device,
                        dtype=tokens.dtype,
                    )
                )
                continue
            user_logits = logits_batch[i, :prompt_len, :].clone()
            if user_logits.shape[0] < max_padded_len:
                pad_len = max_padded_len - user_logits.shape[0]
                pad_logits = user_logits[-1:].expand(pad_len, -1)
                user_logits = torch.cat([user_logits, pad_logits], dim=0)
            last_logits.append(user_logits)

        return torch.stack(last_logits)

    def _set_kv_cache_from_list(self, kv_cache_list: List[object]) -> None:
        """Set internal KV cache tensors from external list (e.g. from decode node)."""
        if len(kv_cache_list) != len(self._kv_caches):
            raise ValueError(
                f"kv_cache_list length {len(kv_cache_list)} != num layers {len(self._kv_caches)}"
            )
        for ref, ext_tensor in zip(self._kv_caches, kv_cache_list):
            ref.tensor = ext_tensor

    def _simulate_embedding_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """Simulate embedding lookup for batch. tokens [B, S] -> [B, S, H]."""
        batch_size, seq_len = tokens.shape
        return torch.randn(
            batch_size, seq_len, self.config.hidden_size, device=tokens.device
        )

    def _simulate_transformer_layer_batch(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Simulate a single transformer layer for the whole batch. [B, S, H] -> [B, S, H]."""
        return (
            hidden_states
            + torch.randn_like(hidden_states, device=hidden_states.device) * 0.01
        )

    def _fill_kv_cache_batch(
        self,
        cache_ref: KVCacheReference,
        prompt_lens: List[int],
    ) -> None:
        """Mark KV cache for a layer as filled for the whole batch."""
        cache_ref.state = KVCacheState.FILLED

    def _simulate_lm_head_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Simulate final norm + LM head. [B, S, H] -> [B, S, vocab_size]."""
        return torch.randn(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.config.vocab_size,
            device=hidden_states.device,
        )

    def get_kv_cache(self, layer_idx: int) -> Optional[KVCacheReference]:
        """Get KV cache reference for a specific layer."""
        if layer_idx < len(self._kv_caches):
            return self._kv_caches[layer_idx]
        return None

    def get_all_kv_caches(self) -> List[KVCacheReference]:
        """Get all KV cache references."""
        return self._kv_caches.copy()

    def cleanup(self) -> None:
        """Release all allocated resources."""
        for cache_ref in self._kv_caches:
            if self.mesh_device is not None:
                try:
                    import ttnn

                    ttnn.deallocate(cache_ref.tensor)
                except (ImportError, Exception):
                    pass
        self._kv_caches = []
        self._is_initialized = False


class DisaggregatedPrefillServer:
    """
    Example server wrapper for disaggregated prefill node.

    Demonstrates how to use DeepSeekPrefillSimulator in a
    disaggregated setup where prefill and decode run on separate nodes.
    """

    def __init__(
        self,
        config: PrefillConfig,
        mesh_device: object = None,
        d2d_transfer_fn: Optional[Callable[[int, KVCacheReference], None]] = None,
    ):
        """
        Initialize the disaggregated prefill server.

        Args:
            config: Model configuration
            mesh_device: TTNN mesh device
            d2d_transfer_fn: Function to transfer KV cache to decode node
        """
        self.config = config
        self.d2d_transfer_fn = d2d_transfer_fn
        self._pending_transfers: List[KVCacheReference] = []

        self.simulator = DeepSeekPrefillSimulator(
            config=config,
            mesh_device=mesh_device,
            on_kv_cache_ready=self._on_kv_cache_ready,
        )

    def _on_kv_cache_ready(
        self,
        layer_idx: int,
        kv_cache_ref: KVCacheReference,
    ) -> None:
        """Internal callback when KV cache is ready."""
        self._pending_transfers.append(kv_cache_ref)

        if self.d2d_transfer_fn is not None:
            self.d2d_transfer_fn(layer_idx, kv_cache_ref)
            kv_cache_ref.mark_streamed()

    def initialize(self) -> None:
        """Initialize and allocate resources."""
        self.simulator.allocate_kv_cache()

    def process_prefill_request(
        self,
        tokens: torch.Tensor,
        prompt_lens: Union[torch.Tensor, List[int]],
        page_table: torch.Tensor,
        kv_cache: List[object],
        start_pos: torch.Tensor,
        *,
        enable_trace: bool = False,
        empty_slots: Optional[Union[torch.Tensor, List[int]]] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """
        Process a prefill request. Same interface as tt_model_runner (mandatory + optional).

        Returns:
            last_logits: [num_of_users, max_padded_len, vocab_size]
        """
        self._pending_transfers.clear()

        return self.simulator.prefill_forward(
            tokens=tokens,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=start_pos,
            enable_trace=enable_trace,
            empty_slots=empty_slots,
            **kwargs,
        )

    def get_pending_transfers(self) -> List[KVCacheReference]:
        """Get list of KV caches pending D2D transfer."""
        return self._pending_transfers.copy()

    def shutdown(self) -> None:
        """Clean up resources."""
        self.simulator.cleanup()


if __name__ == "__main__":

    def example_callback(layer_idx: int, kv_cache_ref: KVCacheReference):
        print(
            f"[Callback] Layer {layer_idx:2d} KV cache ready for whole batch, shape: {kv_cache_ref.shape}"
        )

    config = PrefillConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        kv_lora_rank=64,
        qk_rope_head_dim=32,
        max_seq_len=512,
        batch_size=4,
        vocab_size=1000,
    )

    print("=" * 60)
    print("DeepSeek V3 Prefill Simulator Demo")
    print("=" * 60)
    print(f"\nConfig: num_layers={config.num_layers}, kvpe_dim={config.kvpe_dim}")
    print("empty_slots: maps batch index i -> slot/user_id for KV cache")

    simulator = DeepSeekPrefillSimulator(
        config=config,
        on_kv_cache_ready=example_callback,
    )

    print("\n--- Allocating KV Cache ---")
    kv_caches = simulator.allocate_kv_cache()
    print(f"Allocated {len(kv_caches)} KV caches")

    print("\n--- Running Prefill Forward (same interface as tt_model_runner) ---")
    batch_size = 1
    tokens = torch.randint(0, config.vocab_size, (batch_size, 128))
    prompt_lens = [128]  # one int per request: actual token count before padding
    max_blocks = (config.max_seq_len * batch_size) // config.block_size
    page_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    kv_cache = [ref.tensor for ref in kv_caches]
    start_pos = torch.zeros(batch_size, dtype=torch.int32)

    last_logits = simulator.prefill_forward(
        tokens=tokens,
        prompt_lens=prompt_lens,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=start_pos,
    )

    print("\n--- Prefill Complete ---")
    print(
        f"last_logits shape: {last_logits.shape}  (num_of_users, max_padded_len, vocab_size)"
    )

    simulator.cleanup()
    print("\n--- Demo Complete ---")

import torch
from torch import nn
import ttnn
from contextlib import suppress
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from models.tt_transformers.tt.generator_sglang import LlamaForCausalLM as TT_Llama
from models.tt_transformers.tt.model_config import DecodersPrecision
from ..utils.tt_utils import open_mesh_device, close_mesh_device  # Use plugin's tt_utils
import logging
from sglang.srt.server_args import get_global_server_args
import os

logger = logging.getLogger(__name__)

class TTModels(nn.Module):
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__()
        self.config = config
        self.kv_caches = None
        self.block_size = 32 # TT standard

        if tt_model is not None:
            self.tt_model = tt_model
        else:
            # Initialize TT model if not provided
            # Default parameters
            self.max_batch_size = 32
            # Determine max_seq_len from config
            if hasattr(config, "max_sequence_length"):
                self.max_seq_len = config.max_sequence_length
            elif hasattr(config, "max_position_embeddings"):
                self.max_seq_len = config.max_position_embeddings
            else:
                self.max_seq_len = 4096 # Fallback
                
            self.n_layers = None
            self.tt_data_parallel = 1
            self.optimizations = "performance"
            self.override_tt_config = None

            # Initialize TT device
            try:
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                else:
                    rank = 0
            except Exception:
                rank = 0

            os.environ["HF_MODEL"] = get_global_server_args().model_path

            self.mesh_device =open_mesh_device(self.override_tt_config, trace_mode=False, dp_rank=rank)


    def load_weights(self, weights):
        # TT model loads weights during initialization
        pass

    def allocate_kv_cache(self, kv_cache_shape, dtype=None, layer_num=None):
        if self.kv_caches is None:
            # We need to extract block size from the shape if possible, or use default
            # Shape is usually (num_blocks, num_kv_heads, block_size, head_dim)
            if len(kv_cache_shape) >= 3:
                self.block_size = kv_cache_shape[2]
            
            # Use layer_num if provided (SGLang passes total layers), otherwise get from model args
            num_layers = layer_num if layer_num is not None else self.tt_model.model_args[0].n_layers
            
            # Delegate to TT model's allocate_kv_cache (same as vLLM plugin)
            self.kv_caches = self.tt_model.allocate_kv_cache(
                kv_cache_shape=kv_cache_shape,
                dtype=dtype,
                num_layers=num_layers
            )
        return self.kv_caches

    def _build_page_table(self, forward_batch):
        """Build page table for TT-Metal paged attention.
        
        Uses SGLang's token pool indices to derive block indices.
        This ensures we use DIFFERENT blocks than warmup (which always uses block 0).
        After warmup uses indices 0-127 (block 0), real requests get indices 128+,
        so they use block 1+ which contains no warmup garbage.
        """
        req_to_token_pool = forward_batch.req_to_token_pool
        req_pool_indices = forward_batch.req_pool_indices
        
        # (batch_size, max_len) - contains SGLang's global token indices
        batch_req_tokens = req_to_token_pool.req_to_token[req_pool_indices]
        
        # Subsample to get block indices: take first token of each block
        # (batch_size, max_blocks)
        page_table = batch_req_tokens[:, ::self.block_size] // self.block_size
        
        # Ensure int32 dtype for TT-Metal
        page_table = page_table.to(torch.int32)
        
        logger.info(f"[DEBUG] _build_page_table: block_size={self.block_size}, seq_lens={forward_batch.seq_lens.tolist()}")
        logger.info(f"[DEBUG] batch_req_tokens[0,:10]={batch_req_tokens[0,:min(10, batch_req_tokens.shape[1])].tolist()}")
        logger.info(f"[DEBUG] page_table shape={page_table.shape}, page_table[0,:5]={page_table[0,:min(5, page_table.shape[1])].tolist()}")
        
        return page_table

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        
        page_table = self._build_page_table(forward_batch)
        
        if forward_batch.forward_mode.is_extend():
            # Prefill
            # sglang input_ids is flattened (total_tokens,)
            # We need to reconstruct batch structure (batch, max_len)
            
            batch_size = forward_batch.batch_size
            seq_lens = forward_batch.seq_lens
            max_len = torch.max(seq_lens).item()
            
            logger.info(f"[DEBUG] PREFILL: batch_size={batch_size}, seq_lens={seq_lens.tolist()}, positions={positions.tolist()[:10]}...")
            
            # Create padded tokens tensor
            padded_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
            
            start = 0
            for i in range(batch_size):
                length = seq_lens[i].item()
                padded_tokens[i, :length] = input_ids[start:start+length]
                start += length
            
            # TT expects tokens on host usually, but can handle device tensors if mapped?
            # generator.py usually moves to device.
            # We pass what we have.
            
            logits = self.tt_model.prefill_forward(
                tokens=padded_tokens,
                page_table=page_table,
                kv_cache=self.kv_caches,
                prompt_lens=seq_lens,
                enable_trace=False 
            )

            logger.info(
            f"tt_model.prefill_forward executed"
            )
            
            # logits is (batch, 1, vocab_size)
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        elif forward_batch.forward_mode.is_decode():
            # Decode
            # input_ids is (batch,)
            tokens = input_ids.unsqueeze(1)
            start_pos = positions  # (batch,)
            
            logger.info(f"[DEBUG] DECODE: start_pos={start_pos.tolist()}, token={input_ids.tolist()}")

            # TT decode expects per-device batch == max_batch_size; pad then slice back
            dp = getattr(self.tt_model, "data_parallel", len(self.tt_model.model))
            max_bsz = self.tt_model.model_args[0].max_batch_size
            required_bsz = dp * max_bsz
            actual_bsz = tokens.shape[0]
            if actual_bsz > required_bsz:
                raise ValueError(f"Decode batch {actual_bsz} exceeds TT capacity {required_bsz}")

            if actual_bsz < required_bsz:
                pad_n = required_bsz - actual_bsz
                # Pad tokens (long), positions (long), and page_table (int32) with zeros
                pad_tokens = torch.zeros((pad_n, 1), dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, pad_tokens], dim=0)

                pad_pos = torch.zeros((pad_n,), dtype=start_pos.dtype, device=start_pos.device)
                start_pos = torch.cat([start_pos, pad_pos], dim=0)

                if page_table is not None:
                    pt_width = page_table.shape[1]
                    pad_pt = torch.zeros((pad_n, pt_width), dtype=page_table.dtype, device=page_table.device)
                    page_table = torch.cat([page_table, pad_pt], dim=0)

            decode_output = self.tt_model.decode_forward(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=self.kv_caches,
                enable_trace=False,
                read_from_device=True,
            )

            logger.info(
            f"tt_model.decode_forward executed"
            )

            # decode_forward returns (logits, log_probs) tuple
            if isinstance(decode_output, tuple):
                logits = decode_output[0]
            else:
                logits = decode_output

            # Slice to actual batch, then squeeze sequence dim
            logits = logits[:actual_bsz]
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def allocate_on_device(self):
        """
        Allocate the actual KV cache on the TT-Metal device.
        This method should be called from the ModelRunner after the pool is initialized.
        """

        try:
            import os
            import math
            import ast
            import ttnn

            # Compute num_devices per model (single-process TT in SGLang)
            mesh_device_env = os.environ.get("MESH_DEVICE")
            if mesh_device_env:
                try:
                    parsed = ast.literal_eval(mesh_device_env)
                    if isinstance(parsed, tuple) and len(parsed) == 2:
                        num_devices_per_model = parsed[0] * parsed[1]
                    else:
                        # fallback to dictionary from vLLM plugin
                        grid_map = {
                            "N150": (1, 1),
                            "P100": (1, 1),
                            "P150": (1, 1),
                            "P150x2": (1, 2),
                            "N300": (1, 2),
                            "P300": (1, 2),
                            "N150x4": (1, 4),
                            "P150x4": (1, 4),
                            "T3K": (1, 8),
                            "P150x8": (1, 8),
                            "TG": (8, 4),
                        }
                        grid = grid_map.get(mesh_device_env, (1, 1))
                        num_devices_per_model = grid[0] * grid[1]
                except Exception:
                    num_devices_per_model = len(ttnn.get_device_ids())
            else:
                num_devices_per_model = len(ttnn.get_device_ids())

            # Heuristic blocks: align to vLLM defaults
            is_wormhole = "wormhole_b0" in ttnn.get_arch_name()

            if ( num_devices_per_model == 1 and is_wormhole):
                max_tokens_all_users = 65536
            else:
                max_tokens_all_users = 131072
            # Account for worst-case batch touching new blocks
            max_tokens_all_users += get_global_server_args().page_size * ( get_global_server_args().max_running_requests or 32)
            num_tt_blocks = math.ceil(max_tokens_all_users / get_global_server_args().page_size)
            # Override tokens capacity with heuristic blocks
            max_total_num_tokens = int(num_tt_blocks * get_global_server_args().page_size)

        except Exception as e:
            logger.warning(f"[TT-SGLANG] KV sizing heuristic failed: {e}")

        # 1. Build the KV Cache Shape
        # We use the attributes initialized in __init__ which come from sglang's configuration
        kv_cache_shape = (
            max_total_num_tokens // get_global_server_args().page_size,  # num_blocks
            1,                # num_kv_heads (already adjusted for TP)
            get_global_server_args().page_size,               # block_size
            self.config.head_dim,                # head_size
        )

        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", 32)
        num_effective_layers = end_layer - start_layer

        dtype = torch.bfloat16

        logger.info(f"Allocating KV Cache on device with shape: {kv_cache_shape}, dtype: {dtype}, layers: {num_effective_layers}")

        self.allocate_kv_cache(kv_cache_shape, dtype, num_effective_layers)

        logger.info(
            f"tt_metal.allocate_kv_cache executed"
        )

    def __del__(self):
        """Destructor to clean up TT resources"""
        with suppress(AttributeError):
            # Delete TT model first in case there are model artifacts
            if hasattr(self, 'tt_model'):
                del self.tt_model
            
            # Close mesh device
            if hasattr(self, 'mesh_device') and self.mesh_device is not None:
                close_mesh_device(self.mesh_device, self.override_tt_config)
                del self.mesh_device
                logger.info("Mesh device closed in destructor")

class TTLlamaForCausalLM(TTModels):
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__(config, quant_config, tt_model, **kwargs)

        # Cap max_seq_len for N150 (single device Wormhole B0)
        # TT-Metal raises error if max_seq_len > 65536 on N150 for 8B/11B models
        num_devices = self.mesh_device.get_num_devices()
        is_wormhole = "wormhole_b0" in ttnn.get_arch_name()
        if num_devices == 1 and is_wormhole and self.max_seq_len > 65536:
            logger.info(f"[TT-Plugin] Capping max_seq_len from {self.max_seq_len} to 65536 for N150")
            self.max_seq_len = 65536

        self.tt_model = TT_Llama.initialize_vllm_model(
            config,
            self.mesh_device,
            self.max_batch_size,
            self.max_seq_len,
            self.n_layers,
            self.tt_data_parallel,
            self.optimizations
        )

        logger.info(
            f"TT_Llama.initialize_vllm_model executed"
        )

        self.allocate_on_device()


EntryClass = [TTLlamaForCausalLM]

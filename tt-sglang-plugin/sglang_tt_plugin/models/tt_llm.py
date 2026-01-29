import torch
from torch import nn
import ttnn
from contextlib import suppress
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from models.tt_transformers.tt.generator_sglang import (
    LlamaForCausalLM as TT_Llama,
    QwenForCausalLM as TT_Qwen,
    MistralForCausalLM as TT_Mistral,
    GptOssForCausalLM as TT_GptOss,
)
from models.tt_transformers.tt.model_config import DecodersPrecision
from ..utils.tt_utils import open_mesh_device, close_mesh_device, get_mesh_grid
import logging
from sglang.srt.server_args import get_global_server_args
import os
import math

logger = logging.getLogger(__name__)

class TTModels(nn.Module):
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__()
        self.config = config # hf model config
        self.kv_caches = None # will be allocated on device in allocate_on_device()
        self.block_size = get_global_server_args().page_size or 64  # Block size comes from server's --page-size arg Fall back to 64 if server args not yet available

        if tt_model is not None: #model initialized only once 
            self.tt_model = tt_model
        else:
            server_args = get_global_server_args() # Initialize TT model - get params from server args 
            self.max_batch_size = server_args.max_running_requests or 32 #fallback to 32 if not set
            self.max_seq_len = server_args.context_length 
            self.tt_data_parallel = server_args.dp_size  
            self.optimizations = os.environ.get("TT_METAL_OPTIMIZATIONS", "performance")  # From --optimizations CLI arg
            self.override_tt_config = None
            
            logger.info(f"[TT-SGLANG] Model init: max_batch_size={self.max_batch_size}, "f"max_seq_len={self.max_seq_len}, page_size={self.block_size}")
            # Initialize TT device (rank 0 for single-device, or from distributed if multi-device)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            os.environ["HF_MODEL"] = get_global_server_args().model_path

            self.mesh_device =open_mesh_device(self.override_tt_config, trace_mode=False, dp_rank=rank) #open communication with device

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:

        page_table = self._build_page_table(forward_batch) # returns block IDs for every user in current batch

        if forward_batch.forward_mode.is_extend():  # prefill mode
        
            padded_tokens = self._flatten_to_padded(input_ids, forward_batch)
            
            logits = self.tt_model.prefill_forward(
                tokens=padded_tokens,
                page_table=page_table,
                kv_cache=self.kv_caches,
                prompt_lens=forward_batch.seq_lens,
                enable_trace=False 
            )
            logger.info( f"tt_model.prefill_forward executed")
            # returns scores for every possible next word and sglang picks the most likely one ( it will become the next token )
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        elif forward_batch.forward_mode.is_decode(): # decode mode
           
            tokens = input_ids.unsqueeze(1) # make it batch_size x seq_len dimensions (in decode mode seq_len = 1 )
            start_pos = positions  # at which position is each request starting
            actual_bsz = tokens.shape[0] # number of requests in current batch (needed later to slice output)
            tokens, start_pos, page_table = self._pad_decode_batch(tokens, start_pos, page_table) # pad batch to required size for TT-Metal

            decode_output = self.tt_model.decode_forward( # call TT-Metal decode forward
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                kv_cache=self.kv_caches,
                enable_trace=False,
                read_from_device=True,
            )
            logger.info(f"tt_model.decode_forward executed")
            # returns scores for every possible next word and sglang picks the most likely one ( it will become the next token )
            logits = decode_output[0]
            logits = logits[:actual_bsz] # ignore output of padded requests
            return LogitsProcessorOutput(next_token_logits=logits.squeeze(1))

        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def allocate_on_device(self):
        """
        Allocate the actual KV cache on the TT-Metal device.
        This method should be called from the SGlang's ModelRunner after the pool is initialized.
        """
        # Get mesh grid and hardware info
        mesh_grid = get_mesh_grid(dp_rank=0)
        num_devices_per_model = mesh_grid[0] * mesh_grid[1]
        is_wormhole = "wormhole_b0" in ttnn.get_arch_name()
        model_path = get_global_server_args().model_path or ""
        # Get max tokens based on hardware + model configuration
        max_tokens_all_users = self._get_max_tokens_for_hardware(model_path, num_devices_per_model, is_wormhole)
        
        logger.info(f"[TT-SGLANG] Token limit: {max_tokens_all_users} "f"(model={model_path}, devices={num_devices_per_model}, wormhole={is_wormhole})")
        
        # Build KV Cache Shape
        # 1. Account for worst-case batch allocation, Each user in batch might touch a new block
        page_size = get_global_server_args().page_size # get page size from server args ( how many tokens fit in one block )
        max_batch = get_global_server_args().max_running_requests or 32 # max concurrent requests
        max_tokens_all_users += page_size * max_batch # if all users need new block at the same time
        num_tt_blocks = math.ceil(max_tokens_all_users / page_size) # total number of blocks after worst case scenario
        # 2. Calculate num_kv_heads with tensor parallelism adjustment
        num_devices = self.mesh_device.get_num_devices() // self.tt_data_parallel  # Calculate num_kv_heads with tensor parallelism adjustment
        total_kv_heads = getattr(self.config, 'num_key_value_heads', getattr(self.config, 'num_attention_heads', 8))
        num_kv_heads = total_kv_heads // min(num_devices, total_kv_heads) # kv heads per device
        head_size = getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads)
        
        kv_cache_shape = (
            num_tt_blocks,    # num_blocks
            num_kv_heads,     # num_kv_heads (TP-adjusted)
            page_size,        # block_size
            head_size,        # head_size
        )
        # Get num_layers from config (like vLLM's model_config.get_num_layers_by_block_type())
        num_layers = getattr(self.config, 'num_hidden_layers', getattr(self.config, 'n_layers', getattr(self.config, 'n_layer', 32)))  # Llama/Mistral, GPT-Neo, GPT-2
        dtype = torch.bfloat16

        logger.info(f"Allocating KV Cache on device with shape: {kv_cache_shape}, dtype: {dtype}, layers: {num_layers}")
        # Allocate KV cache directly on TT-Metal
        self.kv_caches = self.tt_model.allocate_kv_cache(
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers
        )
        logger.info(f"tt_model.allocate_kv_cache executed")

    def load_weights(self, weights): # TT model loads weights during initialization
        pass
    #======================================== helper functions ============================================

    def _build_page_table(self, forward_batch):
        """Converts SGLang's token indices (memory positions) per user to block IDs per user.
            helper function for forward function """
        req_to_token_pool = forward_batch.req_to_token_pool # get token pool
        req_pool_indices = forward_batch.req_pool_indices # which rows from token pool are used in current batch
        batch_req_tokens = req_to_token_pool.req_to_token[req_pool_indices] # get tokens used in current batch
        block_size = get_global_server_args().page_size # get block size
        page_table = batch_req_tokens[:, ::block_size] // block_size # convert token indices to block IDs
        return page_table.to(torch.int32)

    def _flatten_to_padded(self, input_ids: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        """Converts SGLang's flattened input_ids to padded batch structure (batch_size, max_len).
            helper function for prefill mode (in forward function)
            
            IMPORTANT: In extend mode, input_ids only contains NEW tokens being extended,
            not the full sequence. Use extend_seq_lens (new token counts) NOT seq_lens (total length).
        """
        batch_size = forward_batch.batch_size # number of requests
        # Use extend_seq_lens (new tokens only) if available, otherwise fall back to seq_lens
        # extend_seq_lens = length of NEW tokens in input_ids
        # seq_lens = TOTAL sequence length (including already-cached prefix)
        if forward_batch.extend_seq_lens is not None:
            extend_lens = forward_batch.extend_seq_lens  # new tokens per request
        else:
            extend_lens = forward_batch.seq_lens  # fallback for non-chunked prefill
        
        max_len = int(torch.max(extend_lens).item()) # length of the longest NEW token chunk
        
        padded_tokens = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
        # sglang gives us flattened input_ids (new tokens only) and their lengths
        # we need to reconstruct batch structure to (batch, max_len)
        start = 0
        for i in range(batch_size):
            length = int(extend_lens[i].item()) # length of NEW tokens for this request
            padded_tokens[i, :length] = input_ids[start:start + length]
            start += length

        return padded_tokens

    def _pad_decode_batch(self, tokens: torch.Tensor, start_pos: torch.Tensor, page_table: torch.Tensor):
        """Pads decode batch to required TT-Metal fixed batch size. We pad smaller batches with dummy requests.
            Helper function for decode mode (in forward function) """
        # calculate required batch size
        dp = getattr(self.tt_model, "data_parallel", len(self.tt_model.model)) # number of model replicas
        max_bsz = self.tt_model.model_args[0].max_batch_size # max batch size for this model
        required_bsz = dp * max_bsz # total slots across all devices
        actual_bsz = tokens.shape[0] # number of requests in current batch
        
        # check if we have too many requests
        if actual_bsz > required_bsz:
            raise ValueError(f"Decode batch {actual_bsz} exceeds TT capacity {required_bsz}")
      
        if actual_bsz < required_bsz: # pad if we have less requests than required
            pad_n = required_bsz - actual_bsz # number of requests to add
            pad_tokens = torch.zeros((pad_n, 1), dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad_tokens], dim=0) # add padding tokens to the end of the batch
            # CRITICAL: pad positions with -1 (tells TT-Metal these are invalid slots to skip)
            pad_pos = torch.full((pad_n,), -1, dtype=start_pos.dtype, device=start_pos.device)
            start_pos = torch.cat([start_pos, pad_pos], dim=0)
            # pad page table with 0s (dummy block IDs)
            if page_table is not None:
                pt_width = page_table.shape[1] # max number of blocks per request
                pad_pt = torch.zeros((pad_n, pt_width), dtype=page_table.dtype, device=page_table.device) 
                page_table = torch.cat([page_table, pad_pt], dim=0)
        
        return tokens, start_pos, page_table

    def _get_max_tokens_for_hardware(self, model_path: str, num_devices: int, is_wormhole: bool) -> int:
        """Calculates max token limit based on model and hardware configuration.
            Helper function for allocate_on_device """
        # Default token limit (generous for Blackhole, multi-device, etc.)
        max_tokens = 131072
        
        # Override for memory-constrained cases (same as vLLM plugin)
        if "gpt-oss" in model_path.lower():
            max_tokens = 1024  # Very limited context
        elif "DeepSeek-R1-0528" in model_path and is_wormhole:
            max_tokens = 32768
        elif is_wormhole:
            # Wormhole-specific memory constraints based on model + device count
            wormhole_limits = [
                (["Llama-3.1-8B", "Mistral-7B", "gemma-3-4b"], 1, 65536),  # N150
                (["DeepSeek-R1-Distill-Qwen-14B", "Qwen2.5-14B"], 2, 65536),  # N300
                (["Llama-3.2-90B", "Qwen2.5-VL-72B"], 8, 65536),  # T3K
            ]
            for models, devices, limit in wormhole_limits:
                if num_devices == devices and any(m in model_path for m in models):
                    max_tokens = limit
                    break
        
        return max_tokens

    #======================================== destructor ============================================
    def __del__(self): # Destructor to clean up TT resources
        with suppress(AttributeError):
            if hasattr(self, 'tt_model'): # Delete TT model first in case there are model artifacts
                del self.tt_model
            # Close mesh device
            if hasattr(self, 'mesh_device') and self.mesh_device is not None: 
                close_mesh_device(self.mesh_device, self.override_tt_config)
                del self.mesh_device
                logger.info("Mesh device closed in destructor")



class TTLlamaForCausalLM(TTModels):
    """SGLang wrapper for Llama models (Llama-3.1-8B, Llama-3.1-70B, Llama-3.2-11B, etc.)"""
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__(config, quant_config, tt_model, **kwargs)

        self.tt_model = TT_Llama.initialize_vllm_model(
            config,
            self.mesh_device,
            self.max_batch_size,
            self.max_seq_len,
            tt_data_parallel=self.tt_data_parallel,
            optimizations=self.optimizations
        )
        logger.info(f"TT_Llama.initialize_vllm_model executed")
        self.allocate_on_device()


class TTQwenForCausalLM(TTModels):
    """SGLang wrapper for Qwen models (Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-72B, etc.)"""
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__(config, quant_config, tt_model, **kwargs)

        self.tt_model = TT_Qwen.initialize_vllm_model(
            config,
            self.mesh_device,
            self.max_batch_size,
            self.max_seq_len,
            tt_data_parallel=self.tt_data_parallel,
            optimizations=self.optimizations
        )
        logger.info(f"TT_Qwen.initialize_vllm_model executed")
        self.allocate_on_device()


class TTMistralForCausalLM(TTModels):
    """SGLang wrapper for Mistral models (Mistral-7B, etc.)"""
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__(config, quant_config, tt_model, **kwargs)

        self.tt_model = TT_Mistral.initialize_vllm_model(
            config,
            self.mesh_device,
            self.max_batch_size,
            self.max_seq_len,
            tt_data_parallel=self.tt_data_parallel,
            optimizations=self.optimizations
        )
        logger.info(f"TT_Mistral.initialize_vllm_model executed")
        self.allocate_on_device()


class TTGptOssForCausalLM(TTModels):
    """SGLang wrapper for GPT-OSS models"""
    def __init__(self, config, quant_config=None, tt_model=None, **kwargs):
        super().__init__(config, quant_config, tt_model, **kwargs)

        self.tt_model = TT_GptOss.initialize_vllm_model(
            config,
            self.mesh_device,
            self.max_batch_size,
            self.max_seq_len,
            tt_data_parallel=self.tt_data_parallel,
            optimizations=self.optimizations
        )
        logger.info(f"TT_GptOss.initialize_vllm_model executed")
        self.allocate_on_device()


# All available TT model classes for SGLang
EntryClass = [TTLlamaForCausalLM, TTQwenForCausalLM, TTMistralForCausalLM, TTGptOssForCausalLM]

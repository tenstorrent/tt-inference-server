import torch
import time
import copy
import os
import sys
import json
from dataclasses import dataclass
from typing import List
from datetime import datetime 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from tt_metal.models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo
from vllm.logger import init_logger
logger = init_logger(__name__)


def new_init_cache_enginer(self):
    assert self.cache_config.num_gpu_blocks is not None
    
    # Get cache path from TT model for caching kv blocks
    self.cache_config.tt_cache_path = None

    from vllm.worker.tt_worker import TTCacheEngine
    
    self.cache_engine = TTCacheEngine(
        self.cache_config, 
        self.model_config, 
        self.parallel_config, 
        self.device_config)
    self.tt_cache = self.cache_engine.tt_cache

def new_allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.
        The assumption is that KV cache for a layer is packed into one tensor. 
        We will have a separate tensor for K and V.

        In the mock implementation, device is always cpu
        """
        # K and V each have the following shape: (num_blocks, num_kv_heads, block_size, head_size)
        kv_cache_shape = (num_blocks, self.num_kv_heads, self.block_size, self.head_size)
        kv_cache: List[torch.Tensor] = []
        num_layers = self.num_attention_layers
        if device == "cpu":
            for _ in range(num_layers):
                # null block in CpuGpuBlockAllocator requires at least that
                # block to be zeroed-out.
                # Zero-initialize CPU cache
                cache_k = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                cache_v = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                kv_cache.append([cache_k, cache_v])
        self.tt_cache = kv_cache # set tt_cache to just be cpu 
        return kv_cache


class MockModel(TtLlamaModelForGeneration):
    # mock implementation in TtLlamaModelForGeneration
    # see: tt-metal/models/demos/t3000/llama2_70b/tt/llama_generation.py
    # inherits from llama at the moment since only this model is currently used with vllm 
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None, vllm=False):

        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        # required to setup model config 
        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device

        # Initial model_config is set in decode mode
        # model conifg is required for vllm 
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            vllm=vllm,
        )
        self.model_config = model_config
        del state_dict
    
    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device, max_batch_size):
        # TODO: pass in model args and tt args as parameters from vllm
        @dataclass
        class ModelArgs:
            llama_version: str = None
            ckpt_dir: str = None
            max_batch_size: int = 32  # overwritten by max_num_seqs from vllm
            num_layers: int = 80
            max_kv_context_len: int = 131072

        @dataclass
        class TTArgs:
            mesh_device: object = None
            cache_path: str = None

        # setup configs
        llama_version = "llama3"
        model_config, ckpt_dir, _, cache_path = setup_llama_env(
            llama_version=llama_version,
        )
        # do not look for mesh device 

        # initialize arg classes
        model_args = ModelArgs(llama_version=llama_version, ckpt_dir=ckpt_dir, max_batch_size=max_batch_size)
        tt_args = TTArgs(mesh_device=t3k_mesh_device, cache_path=cache_path)

        # do not load state dict

        # TODO: delete this configuration setup once llama can directly accept hf_config
        from models.demos.t3000.llama2_70b.reference.llama.llama.model import ModelArgs as ReferenceModelArgs
        from pathlib import Path
        import json

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        configuration = ReferenceModelArgs(
            max_seq_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
            **params,
        )

        return cls(
            configuration=configuration, state_dict=None, model_args=model_args, tt_args=tt_args, vllm=True
        )
    
    def capture_trace(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None):
        # mock out computing trace since TT/GPU device is not being used, only return logits from decode pass 
        tt_logits = self.decode_forward(tokens, start_pos, page_table, kv_cache) # mock out self.tt_model() call
        return None, None, None, None, tt_logits, None
    
    def decode_forward_trace(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        trace_id,
        tt_inp,
        rot_mat,
        cache_idxs_tt,
        tt_logits,
        page_table=None,
        tt_page_table=None,
    ):
        # mock out excuting the trace and only return logits directly 
        batch, seqlen = tokens.shape
        logits = tt_logits
        logits = logits[:batch]  # Remove padded users

        return logits

    def delete_trace(self, trace_id):
        # ttnn.release_trace(self.mesh_device, trace_id)
        return 

    def prefill_forward_single_user(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        user_id: int,
        last_token_idx=None,
        page_table=None,
        kv_cache=None,
    ):
        return self.decode_forward(tokens=tokens, start_pos=start_pos)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
    ):
        assert len(tokens.shape) == 2
        batch, seqlen = tokens.shape
        forward_start = time.time()
        simulated_tps = 10000.0
        simulated_duration = 1.0 / simulated_tps
        # update the new tokens generated to the input id
        # vocab_size = tokenizer.nwords
        # logits: [batch, seqlen, vocab_size]
        logits = torch.randn((batch, seqlen, 128256))
        # send a token every period loops
        EOT_ID = 128009
        # EOS_ID = 128001
        send_index = 200
        send_token = EOT_ID
        if start_pos is not None:
            if isinstance(start_pos, int):
                # if single input per batch 
                cache_idxs = torch.tensor([start_pos for _ in range(batch)], dtype=torch.int64)
            else: # if start_pos is a tensor 
                # if start position is greater than index to send EOT
                cache_idxs = start_pos.to(dtype=torch.int64)
                send_token_mask = cache_idxs > send_index 
                # find positions where start pos passes send_index (ie we are done decording) + make 1D
                batch_indices = torch.nonzero(send_token_mask).squeeze() 
                # assign a high logit at at the send _token index so model will select it and generate the EOT so that generation stops 
                logits[batch_indices, 0, send_token] = 100.0 


        actual_duration = time.time() - forward_start
        # simulate forward latency
        time.sleep(max(simulated_duration - actual_duration, 0))
        return logits
    
    def forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=kv_cache)
        else:
            return self.prefill_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens
            )


class RawStatLogger(StatLoggerBase):

    def __init__(self, num_scheduler_steps) -> None:
        self.time_to_first_token = []
        self.time_per_output_token = []
        self.num_scheduler_steps = num_scheduler_steps
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filepath = f"/home/user/tests/statistics_{timestamp}.json"

    def log(self, stats: Stats, log_to_stdout=True) -> None:
        if len(stats.time_to_first_tokens_iter) > 0:
            self.time_to_first_token.append(stats.time_to_first_tokens_iter)  # Add all values to the list

            if log_to_stdout:
                for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
                    logger.info(f"User {user_idx}: Time to first token {ttft:.2f} s\n")
        if len(stats.time_per_output_tokens_iter) > 0:
            
            tpot = [time / self.num_scheduler_steps for time in stats.time_per_output_tokens_iter]
            self.time_per_output_token.append(tpot)  # Add all values to the list

        self._write_to_json(stats)

    def _write_to_json(self, stats):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {} # if empty or something wrong 
        else:
            data = {}

        if "time to first token" in data:
            # Get the current inference number to use as the key
            num_inference = len(data["time to first token"])
        else:
            # Initialize the "time to first token" dictionary if it doesn't exist
            num_inference = 0
            data["time to first token"] = {}

        data["time to first token"][f"Inference num:{num_inference}"] = {}  # return dict if it exists, otherwise new dict
        for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
            data["time to first token"][f"Inference num:{num_inference}"][f"user {user_idx}"] = ttft

        with open(self.filepath, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info(f"Statistics written to {self.filepath}")

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import copy
import time
from dataclasses import dataclass
from typing import List

import torch

from vllm.engine.metrics import logger

from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
)
from models.demos.t3000.llama2_70b.tt.llama_generation import (
    TtLlamaModelForGeneration,
    get_padded_prefill_len,
)
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)


def new_init_cache_enginer(self):
    assert self.cache_config.num_gpu_blocks is not None

    # Get cache path from TT model for caching kv blocks
    self.cache_config.tt_cache_path = None

    from vllm.worker.tt_worker import TTCacheEngine

    self.cache_engine = TTCacheEngine(
        self.cache_config, self.model_config, self.parallel_config, self.device_config
    )
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
            cache_k = torch.zeros(kv_cache_shape, dtype=self.dtype, device=device)
            cache_v = torch.zeros(kv_cache_shape, dtype=self.dtype, device=device)
            kv_cache.append([cache_k, cache_v])
    self.tt_cache = kv_cache  # set tt_cache to just be cpu
    return kv_cache


class MockModel(TtLlamaModelForGeneration):
    # mock implementation in TtLlamaModelForGeneration
    # see: tt-metal/models/demos/t3000/llama2_70b/tt/llama_generation.py
    # inherits from llama at the moment since only this model is currently used with vllm
    def __init__(
        self,
        configuration,
        state_dict,
        model_args,
        tt_args,
        paged_attention_config=None,
        vllm=False,
    ):
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

    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device, max_batch_size):
        # TODO: pass in model args and tt args as parameters from vllm
        # Note: since mock, do not load state dict and do not look for mesh device
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
        # initialize arg classes
        model_args = ModelArgs(
            llama_version=llama_version,
            ckpt_dir=ckpt_dir,
            max_batch_size=max_batch_size,
        )
        tt_args = TTArgs(mesh_device=t3k_mesh_device, cache_path=cache_path)

        # TODO: delete this configuration setup once llama can directly accept hf_config
        import json
        from pathlib import Path

        from models.demos.t3000.llama2_70b.reference.llama.llama.model import (
            ModelArgs as ReferenceModelArgs,
        )

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        configuration = ReferenceModelArgs(
            max_seq_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
            **params,
        )

        return cls(
            configuration=configuration,
            state_dict=None,
            model_args=model_args,
            tt_args=tt_args,
            vllm=True,
        )

    def capture_trace(
        self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None
    ):
        """
        Called in TTModelRunner to capture trace for the first decode execution
        """
        # mock out computing trace since TT/GPU device is not being used, only return logits from decode pass
        tt_logits = self.decode_forward(
            tokens, start_pos, page_table, kv_cache
        )  # mock out self.tt_model() call
        return None, None, None, None, tt_logits, None

    def decode_forward_trace(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        trace_id,
        tt_inp,
        rot_idxs_tt,
        cache_idxs_tt,
        tt_logits,
        page_table=None,
        tt_page_table=None,
        read_from_device=True,
    ):
        """
        Runs model in TTModelRunner by executing trace
        """
        # mock out excuting the trace and only return logits directly
        batch, seqlen = tokens.shape
        logits = tt_logits
        logits = logits[:batch]  # Remove padded users

        return logits

    def read_forward_trace(self, tt_logits, unpadded_batch=None):
        return tt_logits

    def delete_trace(self, trace_id):
        """
        Called to delete trace in TTModelRunner
        """
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

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
    ):
        """
        Called in forward when seq_len != 1.
        Finds correct padding and calls prefill forward for each user in batch.
        """

        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.params.vocab_size)
        prompt_lens = (
            prompt_lens
            if prompt_lens is not None
            else torch.tensor([batch_seq_len] * batch)
        )
        for user_id in range(batch):
            seq_len = prompt_lens[user_id]
            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [
                    tokens[user_id : user_id + 1, :seq_len],
                    torch.zeros(1, prefill_seq_len - seq_len).long(),
                ],
                dim=-1,
            )
            logger.info(f"Filling kv cache for user {user_id + 1}")
            last_token_idx = seq_len - 1
            logits = self.prefill_forward_single_user(
                prefill_ids,
                start_pos,
                user_id,
                last_token_idx=last_token_idx,
                page_table=page_table,
                kv_cache=kv_cache,
            )
            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits[
                :, last_token_idx % 32 : last_token_idx % 32 + 1, :
            ]

        return output_logits

    def decode_mock_send_token(self, logits, start_pos, batch, send_eot=False):
        # tooling for sending EOT token or other specific token at specific output position
        EOT_ID = 128009
        send_index = 200
        send_token = EOT_ID
        if send_eot:
            if start_pos is not None:
                if isinstance(start_pos, int):
                    # if start pos is same across batch, ie. now in prefill
                    cache_idxs = torch.tensor(
                        [start_pos for _ in range(batch)], dtype=torch.int64
                    )
                else:  # if start_pos is a tensor ie. is different across batch, now in decode mode
                    # if start position is greater than index to send EOT
                    cache_idxs = start_pos.to(dtype=torch.int64)
                    send_token_mask = cache_idxs > send_index
                    # find positions where start pos passes send_index (ie. done decoding) + make 1D
                    batch_indices = torch.nonzero(send_token_mask).squeeze()
                    # assign a high logit at at the send _token index so model will select it and generate the EOT so that generation stops
                    logits[batch_indices, 0, send_token] = 100.0
        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
    ):
        """
        Does forward pass. consdiring if in prefill stage or decode stage.
        """
        assert len(tokens.shape) == 2
        batch, seqlen = tokens.shape
        forward_start = time.time()
        simulated_tps = 10000.0
        simulated_duration = 1.0 / simulated_tps
        # update the new tokens generated to the input id
        # vocab_size = tokenizer.nwords
        # logits: [batch, seqlen, vocab_size]
        logits = torch.randn((batch, seqlen, 128256))
        logits = self.decode_mock_send_token(logits, start_pos, batch, send_eot=True)
        actual_duration = time.time() - forward_start
        # simulate forward latency
        time.sleep(max(simulated_duration - actual_duration, 0))
        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
    ):
        """
        Called in TTModelRunner if trace mode is not on
        """
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache
            )
        else:
            return self.prefill_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
            )

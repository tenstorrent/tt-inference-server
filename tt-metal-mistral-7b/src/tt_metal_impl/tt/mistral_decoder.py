# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.wormhole.mistral7b.tt.mistral_attention import TtMistralAttention
from models.demos.wormhole.mistral7b.tt.mistral_mlp import TtMistralMLP
from models.demos.wormhole.mistral7b.tt.mistral_rms_norm import TtRMSNorm


class TtTransformerBlock(torch.nn.Module):
    def __init__(self, args, device, dtype, state_dict, layer_num, weight_cache_path, rot_mat, start_pos):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.num_devices = 1
        self.start_pos = start_pos

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.sliding_window = args.sliding_window
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.attention = TtMistralAttention(
            devices=[device],
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
            rot_mat=rot_mat,
            start_pos=start_pos,
        )
        self.feed_forward = TtMistralMLP(
            device=device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        self.attention_norm = TtRMSNorm(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            layer_num=layer_num,
            weight_key="attention_norm",
            model_config=self.args.get_model_config(),
        )
        self.ffn_norm = TtRMSNorm(
            device=device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            layer_num=layer_num,
            weight_key="ffn_norm",
            model_config=self.args.get_model_config(),
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        attn_norm = self.attention_norm(x)
        # Attention module expects a list of inputs, attn masks (multi-device support)
        r = self.attention.forward(
            [attn_norm],
            current_pos,
            [attn_masks],
        )
        # Attention also returns multiple outputs (multi-device support)
        assert len(r) == 1, "Multiple devices not yet supported"
        r = r[0]
        r = ttnn.reshape(r, (1, 1, 32, 4096))
        h = ttnn.add(x, r, memory_config=self.model_config["DEC_SKIP_OUTPUT_MEMCFG"])
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = ttnn.add(h, r, memory_config=self.model_config["DEC_SKIP_OUTPUT_MEMCFG"])
        return out

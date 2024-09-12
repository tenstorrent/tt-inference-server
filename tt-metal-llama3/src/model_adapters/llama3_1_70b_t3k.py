import os
import time
import traceback
import threading
from multiprocessing import Queue
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F
from transformers.generation.utils import top_k_top_p_filtering

import ttnn
import tt_lib as ttl


from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
    ChatFormat,
    Message,
)
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_mesh_device,
    setup_llama_env,
)

from models.demos.t3000.llama2_70b.demo.demo import (
    ModelArgs,
    TTArgs,
    DataArgs,
    DemoArgs,
    construct_arg,
    build_generator,
)

from utils import batch_top_pk_logits_efficient
from model_adapters.model_adapter import ModelAdapterABC


class MockModel:

    def __init__(self):
        self.forward_counter = 0

    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        assert len(tokens.shape) == 2
        batch, seqlen = tokens.shape
        forward_start = time.time()
        simulated_tps = 10.0
        simulated_duration = 1.0 / simulated_tps
        # update the new tokens generated to the input id
        # vocab_size = tokenizer.nwords
        # logits: [batch, seqlen, vocab_size]
        logits = torch.randn([batch, seqlen, 128256])
        # send a token every period loops
        EOT_ID = 128009
        EOS_ID = 128001
        period = 100
        send_token = EOT_ID
        if self.forward_counter % period == 0:
            print(f"sending {send_token}")
            logits[:, :, send_token] = 100.0
        self.forward_counter += 1
        actual_duration = time.time() - forward_start
        # simulate forward latency
        time.sleep(max(simulated_duration - actual_duration, 0))
        return logits


def mock_init_model(self):
    weights_path, tt_cache_path = get_model_weights_and_tt_cache_paths()
    tokenizer_path = weights_path.joinpath("tokenizer.model")
    # vocab_size = 32000
    self.tokenizer = Tokenizer3(model_path=tokenizer_path.as_posix())
    self.formatter = ChatFormat(self.tokenizer)
    self.model = MockModel()


class Llama3_70B_T3K(ModelAdapterABC):
    def __init__(device, inference_config, tt_cache_path):
        # set weights using:
        # MODEL_WEIGHTS_ID
        # MODEL_WEIGHTS_PATH
        weights_path, tt_cache_path = get_model_weights_and_tt_cache_paths()
        tokenizer_path = weights_path.joinpath("tokenizer.model")
        logger.info(f"tokenizer_path=:{tokenizer_path}")
        logger.info("init_model ...")
        model_config, _, _, _ = setup_llama_env(
            llama_version=inference_config.model_config.llama_version,
        )
        self.model_config = model_config
        # override for tt-studio
        ckpt_dir = weights_path.as_posix()
        tokenizer_path = tokenizer_path.as_posix()
        cache_path = tt_cache_path
        self.init_tt_metal_device()

        # set unused vars to None to obviously break any code using them
        args = construct_arg(
            implementation="tt",
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            skip_model_load=False,
            num_layers=self.num_layers,
            max_batch_size=self.batch_size,
            max_kv_context_len=self.max_seq_len,
            max_output_tokens=self.max_seq_len,
            prompts_file=None,
            output_at_end=None,
            top_p=None,
            top_k=None,
            temperature=None,
            chat=inference_config.model_config.chat,
            mesh_device=self.device,
            n_devices=inference_config.n_devices,
            cache_path=cache_path,
            decode_only=self.decode_only,
            ground_truth=False,
        )
        model_args = args.model
        tt_args = args.tt

        generator = build_generator(model_args, tt_args)
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.formatter = ChatFormat(self.tokenizer)

    def check_device():
        pass

    def initialize_inputs(self, tokenizer, prompt_tokens, bsz, total_len):
        # pad the model to maximum length
        pad_id = tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = (
                torch.tensor(t[:total_len], dtype=torch.long, device="cpu")
                .clone()
                .detach()
            )
        eos_reached = torch.tensor([False] * bsz, device="cpu")
        input_text_mask = (
            tokens != pad_id
        )  # use prefill token if that token is not masked
        return tokens, input_text_mask, eos_reached

    def tokenize_prompt(
        self,
        prompt: str,
        rag_context: str = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        if self.chat and add_special_tokens:
            if rag_context:
                messages = [
                    Message(
                        role="system",
                        content=f"Please use the following context to answer the question:\n{rag_context}",
                    ),
                    Message(role="user", content=prompt),
                ]
                return self.formatter.encode_dialog_prompt(messages)
            else:
                # encode as a single turn of dialog
                messages = [Message(role="user", content=prompt)]
                return self.formatter.encode_dialog_prompt(messages)
        else:
            return self.tokenizer.encode(prompt, bos=add_special_tokens, eos=False)

    def prepare_batch_inputs(self):
        self.num_users = len(self.get_users())
        assert self.num_users <= self.max_users
        input_prompts = [user.prompt_tokens for user in self.get_users()]
        self.max_prompt_len = max(
            [user.num_prefill_tokens for user in self.get_users()]
        )
        self.min_prompt_len = min(
            [user.num_prefill_tokens for user in self.get_users()]
        )
        # pad inputs, empty users get pad id
        prefill_tokens, input_text_mask, _ = initialize_inputs(
            tokenizer=self.tokenizer,
            prompt_tokens=input_prompts,
            bsz=len(input_prompts),
            total_len=self.min_prompt_len,
        )
        # where does intput_text_mask get used?
        self.input_text_mask = input_text_mask
        self.prefill_ids = prefill_tokens
        # decode_ids are padded to batch_size
        decode_ids = torch.full(
            (self.batch_size, 1), self.tokenizer.pad_id, dtype=torch.long, device="cpu"
        )
        decode_ids[: self.num_users, :1] = prefill_tokens[:, :1].clone()
        self.decode_ids = decode_ids

    def prefill(self):
        self.timer_start("prefill")
        for user in self.get_users():
            user.start_prefill_timer()
        if self.prefill_ids is None:
            return
        batch_size, seq_len = self.prefill_ids.shape
        # runs prefill for full batch
        if seq_len > 1:
            # prefill is defined in TtLlamaModelForGeneration by sending seq_len > 1
            # seq_len is tokens.shape[1]
            prefill_logits = self.model.forward(self.prefill_ids, self.prev_pos)
            self.prefill_seq_len = seq_len
            self.prefill_batch_size = batch_size
            self.prev_pos = seq_len
            self.cur_pos = self.prev_pos + 1

        for user in self.get_users():
            user.num_tokens_prefilled = self.prefill_seq_len
            user.stop_prefill_timer()
            if user.num_prefill_tokens <= user.num_tokens_prefilled:
                user.prefill_complete = True
            else:
                user.start_prefill_via_decode_timer()

        self.prefill_ids = None
        self.timer_stop("prefill")

    def decode(self, decode_ids, pos):
        logits = self.model.forward(decode_ids, pos)
        next_tokens = batch_top_pk_logits_efficient(
            logits,
            top_ps=self.get_user_param("top_p"),
            top_ks=self.get_user_param("top_k"),
            temperatures=self.get_user_param("temperature"),
        ).reshape(self.batch_size, 1)
        return logits

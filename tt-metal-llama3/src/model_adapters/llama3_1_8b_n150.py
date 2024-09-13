import torch
import json

# from time import time
import time
from datetime import datetime
import os
import ttnn
import pytest
from typing import List


from models.demos.wormhole.llama31_8b.tt.llama_common import (
    prepare_inputs_ttnn,
    sample,
    get_single_rot_mat,
    cache_attention,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    encode_prompt_llama_instruct,
    HostEmbedding,
)
from models.demos.wormhole.llama31_8b.tt.llama_model import TtTransformer
from models.demos.wormhole.llama31_8b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
    ChatFormat,
    Message,
)

from models.demos.wormhole.llama31_8b.demo.demo_with_prefill import (
    preprocess_inputs_prefill,
)
from model_adapters.model_adapter import ModelAdapterABC
from model_weights_handler import get_model_weights_and_tt_cache_paths

from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


class MockTtTransformer:
    def __init__(self, *args, **kwargs):
        self.layers = []
        self.device = "cpu"
        self.forward_counter = 1

    def __call__(
        self,
        x,
        current_pos: int,
        attn_masks=None,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
    ):
        if mode == "prefill":
            return
        elif mode == "decode":
            assert len(x.shape) == 4
            _, _, batch, embd_dim = x.shape
            forward_start = time.time()
            simulated_tps = 1000.0
            simulated_duration = 1.0 / simulated_tps
            # update the new tokens generated to the input id
            # vocab_size = tokenizer.nwords
            # logits: [batch, seqlen, vocab_size]
            logits = torch.randn([1, 1, batch, 128256])
            # send a token every period loops
            EOT_ID = 128009
            EOS_ID = 128001
            period = 100
            send_token = EOT_ID
            if self.forward_counter % period == 0:
                logger.info(f"Mock model: sending {send_token}")
                logits[:, :, :, send_token] = 100.0
            self.forward_counter += 1
            actual_duration = time.time() - forward_start
            # simulate forward latency
            time.sleep(max(simulated_duration - actual_duration, 0))
            return logits
        else:
            raise ValueError("mode must be 'prefill' or 'decode'")


class Llama3_1_8B_N150(ModelAdapterABC):
    embed_on_device = True

    def __init__(self, device, inference_config):
        # set weights using:
        # MODEL_WEIGHTS_ID
        # MODEL_WEIGHTS_PATH
        weights_path, tt_cache_path = get_model_weights_and_tt_cache_paths()
        tokenizer_path = weights_path
        logger.info(f"tokenizer_path=:{tokenizer_path}")
        #
        self.device = device
        self.instruct = inference_config.model_config.chat
        self.max_generated_tokens = inference_config.model_config.max_seq_len
        self.batch_size = inference_config.model_config.batch_size
        # need to set for TtModelArgs
        os.environ["LLAMA_CKPT_DIR"] = str(weights_path)
        os.environ["LLAMA_TOKENIZER_PATH"] = str(weights_path)
        os.environ["LLAMA_CACHE_PATH"] = str(tt_cache_path)
        # need to import here because the class on import defines defaults from env vars
        from models.demos.wormhole.llama31_8b.tt.model_config import TtModelArgs

        model_args = TtModelArgs(device, instruct=self.instruct)
        self.model_args = model_args
        dtype = ttnn.bfloat8_b
        self.dtype = dtype
        # Load model args, weights, and tokenizer
        self.tokenizer = Tokenizer(model_args.tokenizer_path)
        self.formatter = ChatFormat(self.tokenizer)
        #
        logger.info("Loading weights...")
        # # profiler.start("weight_loading")
        if inference_config.model_config.device_type == "cpu":
            # for mocking
            state_dict = {
                "tok_embeddings.weight": torch.empty(
                    (model_args.vocab_size, model_args.dim)
                )
            }
        else:
            state_dict = torch.load(
                model_args.consolidated_weights_path, map_location=torch.device("cpu")
            )
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if (
                    any([f"layers.{i}." in k for i in range(model_args.n_layers)])
                    or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                )
            }
        # # profiler.end("weight_loading")
        logger.info("Loading weights finished!")
        # TODO Should we keep initial embedding on host?
        self.embd = HostEmbedding(model_args)
        self.embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
        logger.info("embeddings initialized")
        input_prompts = ["test"]
        (
            _,
            _,
            _,
            _,
            rot_emb_matrix_list,
            self.prefill_seq_len,
            _,
        ) = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.dtype,
            self.embd,
            self.instruct,
            device,
        )
        # # profiler.end("preprocess_prefill_inputs")
        generation_start_pos = self.prefill_seq_len
        max_generated_tokens = inference_config.model_config.max_seq_len
        # users_decoding = True

        # pre-compute the rotational embedding matrix and send to device
        current_rot_mat, rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            device,
            start_pos=0,
        )
        logger.info(
            f"caching attention for {self.prefill_seq_len} prefill tokens + {max_generated_tokens} generated tokens"
        )
        # # profiler.start("cache_attention")
        cache_attention(
            device,
            state_dict,
            model_args,
            current_rot_mat,
            self.dtype,
            self.prefill_seq_len + max_generated_tokens,
        )
        # # profiler.end("cache_attention")

        # Load TTNN Llama3.1 model
        logger.info("Loading weights to device...")
        # # profiler.start("loading_weights_to_device")
        self.tt_model = TtTransformer(
            args=model_args,
            device=device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layers=list(range(model_args.n_layers)),
            rot_mat=rot_emb_matrix_list,
            start_pos=generation_start_pos,
        )
        if self.embed_on_device:
            self.tt_embd = TtLlamaEmbedding(
                device=device,
                args=model_args,
                weight_cache_path=model_args.weight_cache_path(dtype),
                state_dict=state_dict,
                dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
            )
        else:
            self.tt_embd = None
        # # profiler.end("loading_weights_to_device")
        logger.info("Finished loading weights to device. Starting inference...")

    def tokenize_prompt(
        self,
        prompt: str,
        rag_context: str = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        if self.instruct and add_special_tokens:
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

    def initialize_inputs(self, input_prompts):
        (
            self.pt_encoded_input,
            self.tt_decode_input,
            self.pt_prefill_input,
            self.input_mask,
            self.rot_emb_matrix_list,
            self.prefill_seq_len,
            self.encoded_prompts,
        ) = preprocess_inputs_prefill(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.dtype,
            self.embd,
            self.instruct,
            self.device,
        )
        self.generation_start_pos = self.prefill_seq_len
        self.iteration = 0
        # set kv cache to zeros if not first batch, to avoid context leaking
        for layer in self.tt_model.layers:
            k_cache, v_cache = layer.attention.layer_past_list[0]
            k_cache = k_cache * 0
            v_cache = v_cache * 0
            # Deallocation is necessary to avoid memory leaks and running out of L1 in later batches
            layer.attention.layer_past_list[0][0].deallocate(True)
            layer.attention.layer_past_list[0][1].deallocate(True)
            layer.attention.layer_past_list[0] = [k_cache, v_cache]

        # for compatability output
        prefill_tokens = torch.tensor(
            [enc_prompt[: self.prefill_seq_len] for enc_prompt in self.encoded_prompts]
        )
        return prefill_tokens

    def prefill(self):
        if self.prefill_seq_len > 0:
            logger.info(f"Starting prefill [{self.prefill_seq_len} tokens]...")
            # profiler.start(f"prepare_rot_mat_for_prefill", iteration=batch_idx)
            rot_mats_prefill = get_prefill_rot_mat(
                self.model_args.head_dim,
                self.model_args.max_seq_len,
                self.device,
                seq_len=self.prefill_seq_len,
            )
            head_dim = self.model_args.dim // self.model_args.n_heads
            transformation_mat_torch = get_rot_transformation_mat(head_dim)
            transformation_mats = ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # profiler.end(f"prepare_rot_mat_for_prefill", iteration=batch_idx)

            # First user is used for compile time
            num_users_generated_prefill = (
                self.batch_size - 1 if self.batch_size > 1 else 1
            )  # First user is used for compile time

            # profiler.start(f"inference_prefill", iteration=batch_idx)
            for batch_id in range(self.batch_size):
                # if batch_id == 0:  # First user prefill also accounts for compile time
                #     # profiler.start(f"compile_prefill", iteration=batch_idx)
                prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                    self.pt_prefill_input[batch_id],
                    self.device,
                )
                tt_out = self.tt_model(
                    prefill_input,
                    0,  # Current position
                    attn_mask,
                    rot_mats_prefill,
                    transformation_mats,
                    user_id=batch_id,
                    mode="prefill",
                )
                # if batch_id == 0:  # First user prefill also accounts for compile time
                #     # profiler.end(f"compile_prefill", iteration=batch_idx)

            # Device synchrozization ensures profiler is accurate in end-to-end timing
            ttnn.synchronize_device(self.device)
            # profiler.end(f"inference_prefill", iteration=batch_idx)
            logger.info(f"Prefill finished [{self.prefill_seq_len} tokens]!")

        logger.info("Starting decode...")

        # profiler.start(f"get_single_rot_mat_decode_{batch_idx}")
        self.current_rot_mat, self.rot_matrix = get_single_rot_mat(
            self.model_args.head_dim,
            self.device,
            start_pos=self.prefill_seq_len,
        )
        # profiler.end(f"get_single_rot_mat_decode_{batch_idx}")
        # profiler.start(f"inference_decode", iteration=batch_idx)

    def decode(self, tokens, cur_pos):
        # iteration_time_start = time()
        self.curr_pos = self.generation_start_pos + self.iteration

        # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
        # profiler.start(f"prepare_input_decode", iteration=batch_idx)
        if self.embed_on_device and self.iteration > 0:
            current_pos = self.curr_pos
            decode_input = self.pt_encoded_input
        else:
            decode_input, current_pos = prepare_inputs_ttnn(
                self.pt_encoded_input,
                self.curr_pos,
                self.model_args.dim,
                self.model_args.sliding_window,
                self.tt_model.device,
            )
        # profiler.end(f"prepare_input_decode", iteration=batch_idx)

        # profiler.start(f"decode_and_argmax", iteration=batch_idx)
        # Run ttnn llama3.1 model
        tt_out = self.tt_model(decode_input, current_pos, rot_mat=self.current_rot_mat)
        tt_out = ttnn.untilize(
            tt_out, use_multicore=False
        )  # multi-core OOMs (https://github.com/tenstorrent/tt-metal/issues/9022)
        tt_output_torch = (
            ttnn.to_torch(tt_out)
            .permute(2, 1, 0, 3)
            .squeeze(1)[: self.batch_size, :, :]
        )  # [batch, seq, hidden_dim]
        # Update rotation matrix for next iteration
        self.current_rot_mat = ttnn.linear(self.rot_matrix, self.current_rot_mat)
        # If temperature is 0, does greedy decoding (top-1)
        tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)
        # profiler.end(f"decode_and_argmax", iteration=batch_idx)
        out_tok = tt_out_tok.clone()
        if self.embed_on_device:
            # Pad tt_out_tok to batch size of 32
            padded_tt_out_tok = torch.zeros(
                1, 32, dtype=tt_out_tok.dtype, device=tt_out_tok.device
            )
            padded_tt_out_tok[: tt_out_tok.shape[1]] = tt_out_tok
            tt_out_tok = ttnn.from_torch(
                padded_tt_out_tok,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.pt_encoded_input = self.tt_embd(tt_out_tok)
        else:
            self.pt_encoded_input = self.embd(tt_out_tok)
        # profiler.end(f"decode_embedding", iteration=batch_idx)

        self.iteration += 1

        return out_tok

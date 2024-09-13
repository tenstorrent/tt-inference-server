import torch
import json

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
# from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
#     ChatFormat,
#     Message,
# )
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf

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
    batch_idx = -1

    def __init__(self, device, inference_config):
        self.log_cache = inference_config.log_cache
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

        # Start profiler
        logger.info(f"Start profiler")
        self.profiler = BenchmarkProfiler()
        self.profiler.start("run")

        # Load model args, weights, and tokenizer
        self.tokenizer = Tokenizer(model_args.tokenizer_path)
        # self.formatter = ChatFormat(self.tokenizer)
        #
        self.profiler.start("weight_loading")
        if inference_config.model_config.device_type == "cpu":
            # for mocking
            logger.info("Loading mocking tok_embeddings.weights ...")
            state_dict = {
                "tok_embeddings.weight": torch.empty(
                    (model_args.vocab_size, model_args.dim)
                )
            }
        else:
            logger.info("Loading weights ...")
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
        self.profiler.end("weight_loading")
        logger.info("Loading weights finished!")
        # TODO Should we keep initial embedding on host?
        self.embd = HostEmbedding(model_args)
        self.embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
        logger.info("embeddings initialized")
        input_prompts = ["test"]
        self.profiler.start("preprocess_prefill_inputs")
        (
            _,
            _,
            _,
            _,
            self.rot_emb_matrix_list,
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
        self.profiler.end("preprocess_prefill_inputs")
        generation_start_pos = self.prefill_seq_len
        max_generated_tokens = inference_config.model_config.max_seq_len
        # max_generated_tokens = 120
        # users_decoding = True

        # TODO: should this always be for max_seq_len?
        # pre-compute the rotational embedding matrix and send to device
        self.current_rot_mat, self.rot_matrix = get_single_rot_mat(
            model_args.head_dim,
            device,
            start_pos=0,
        )
        logger.info(
            f"caching attention for {self.prefill_seq_len} prefill tokens + {max_generated_tokens} generated tokens"
        )
        self.profiler.start("cache_attention")
        cache_attention(
            device,
            state_dict,
            model_args,
            self.current_rot_mat,
            self.dtype,
            self.prefill_seq_len + max_generated_tokens,
        )
        self.profiler.end("cache_attention")

        # Load TTNN Llama3.1 model
        logger.info("Loading weights to device...")
        self.profiler.start("loading_weights_to_device")
        self.tt_model = TtTransformer(
            args=model_args,
            device=device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layers=list(range(model_args.n_layers)),
            rot_mat=self.rot_emb_matrix_list,
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
        self.profiler.end("loading_weights_to_device")
        logger.info("Finished loading weights to device. Starting inference...")

    def tokenize_prompt(
        self,
        prompt: str,
        rag_context: str = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> List[int]:
        if rag_context:
            logger.info(f"rag_context: {rag_context}")
            prompt = f"{rag_context}\n{prompt}" 
        if self.instruct:
            encoded_prompts = encode_prompt_llama_instruct(self.tokenizer, prompt)
        else:
            encoded_prompts = tokenizer.encode(prompt, bos=True, eos=False)
        return encoded_prompts

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
        self.batch_idx += 1
        # set kv cache to zeros if not first batch, to avoid context leaking
        if self.batch_idx > 0:
            logging.info("Resetting kv cache to zeros")
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
            self.profiler.start(f"prepare_rot_mat_for_prefill", iteration=self.batch_idx)
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
            self.profiler.end(f"prepare_rot_mat_for_prefill", iteration=self.batch_idx)

            # First user is used for compile time
            self.num_users_generated_prefill = (
                self.batch_size - 1 if self.batch_size > 1 else 1
            )  # First user is used for compile time
            self.profiler.start(f"inference_prefill", iteration=self.batch_idx)
            for batch_id in range(self.batch_size):
                if self.batch_idx == 0:  # First user prefill also accounts for compile time
                    self.profiler.start(f"compile_prefill", iteration=self.batch_idx)
                if self.prefill_seq_len > 0:
                    prefill_input, attn_mask, _ = prepare_inputs_ttnn_prefill(
                        self.pt_prefill_input[batch_id],
                        self.device,
                    )
                    tt_out = self.tt_model(
                        x=prefill_input,
                        current_pos=0,  # Current position
                        attn_masks=attn_mask,
                        rot_mat=rot_mats_prefill,
                        transformation_mats=transformation_mats,
                        user_id=batch_id,
                        mode="prefill",
                    )
                if self.batch_idx == 0:  # First user prefill also accounts for compile time
                    self.profiler.end(f"compile_prefill", iteration=self.batch_idx)

            # Device synchrozization ensures profiler is accurate in end-to-end timing
            ttnn.synchronize_device(self.device)
            self.profiler.end(f"inference_prefill", iteration=self.batch_idx)
            logger.info(f"Prefill finished [{self.prefill_seq_len} tokens]!")

        logger.info("Starting decode...")

        self.profiler.start(f"get_single_rot_mat_decode_{self.batch_idx}")
        self.current_rot_mat, self.rot_matrix = get_single_rot_mat(
            self.model_args.head_dim,
            self.device,
            start_pos=self.prefill_seq_len,
        )
        self.profiler.end(f"get_single_rot_mat_decode_{self.batch_idx}")
        self.profiler.start(f"inference_decode", iteration=self.batch_idx)

    def decode(self, tokens, cur_pos):
        # iteration_time_start = time()
        self.curr_pos = self.generation_start_pos + self.iteration

        # Prepare inputs for decode mode (rotary embeddings, attention mask, padding)
        self.profiler.start(f"prepare_input_decode", iteration=self.batch_idx)
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
        self.profiler.end(f"prepare_input_decode", iteration=self.batch_idx)

        self.profiler.start(f"decode_and_argmax", iteration=self.batch_idx)
        # Run ttnn llama3.1 model
        if self.iteration == 0:  # First iteration also accounts for compile time
            self.profiler.start(f"compile_decode", iteration=self.batch_idx)
        tt_out = self.tt_model(
            x=decode_input,
            current_pos=current_pos, 
            attn_masks=None,
            rot_mat=self.current_rot_mat,
            transformation_mats=None,
            user_id=0,
            mode="decode",
        )
        if self.iteration == 0:  # First iteration also accounts for compile time
            self.profiler.end(f"compile_decode", iteration=self.batch_idx)
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
        self.profiler.end(f"decode_and_argmax", iteration=self.batch_idx)
        if self.iteration < self.input_mask.shape[1]:  # If prefill
            # If token is pad token, start generating new token, otherwise, push the next prompt token to the model
            tt_out_tok = torch.where(
                self.input_mask[:, self.iteration], self.tt_decode_input[:, self.iteration], tt_out_tok[:, 0]
            ).unsqueeze(1)
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
        self.profiler.end(f"decode_embedding", iteration=self.batch_idx)

        self.iteration += 1
        self.profiler.end(f"inference_decode", iteration=self.batch_idx)
        return out_tok

    def profiler_batch_output(self, output=True):
        self.profiler.end("run")
        if not output:
            return
        N_warmup_iter = {"inference_prefill": 0, "inference_decode": 0}

        # Benchmark metrics for batch 0
        compile_prefill_time = self.profiler.get_duration("compile_prefill")
        compile_decode_time = self.profiler.get_duration("compile_decode")
        inference_prefill_time = self.profiler.get_duration("inference_prefill")
        inference_decode_time = self.profiler.get_duration("inference_decode")
        # log_printing_time = sum(profiler.get_duration(f"log_printing_iter_{i}") for i in range(max_generated_tokens))
        # log_saving_file_time = self.profiler.get_duration(f"log_saving_file")

        # Correct the inference decode time to remove the time spent on compile (1st iteration) and log_printing (at the end of every iteration)
        inference_decode_time = inference_decode_time - compile_decode_time
        # Correct the inference prefill time to remove the time spent on compile (1st iteration)
        inference_prefill_time = inference_prefill_time - compile_prefill_time

        num_tokens_generated_decode = self.iteration
        # FIXME: Currently our prefill pass does not generate the first token, so we correct the time_to_first to include 1 prefill step + 1 decode step
        prefill_time_to_first = (inference_prefill_time / self.num_users_generated_prefill) + (
            inference_decode_time / num_tokens_generated_decode
        )

        measurements = {
            # Required measurements
            "compile_prefill": compile_prefill_time,
            "compile_decode": compile_decode_time,
            "inference_prefill": inference_prefill_time,
            "inference_decode": inference_decode_time,
            "prefill_time_to_token": prefill_time_to_first,
            "prefill_t/s": (self.num_users_generated_prefill * self.prefill_seq_len) / inference_prefill_time,  # tokens/s
            "decode_t/s/u": num_tokens_generated_decode / inference_decode_time,  # tokens/s
            "decode_t/s": num_tokens_generated_decode / inference_decode_time * self.batch_size,  # tokens/s/user
            # Optional measurements
            # "loading_inputs": self.profiler.get_duration("loading_inputs"),
            "weight_loading": self.profiler.get_duration("weight_loading"),
            "preprocess_prefill_inputs": self.profiler.get_duration("preprocess_prefill_inputs"),
            "loading_weights_to_device": self.profiler.get_duration("loading_weights_to_device"),
            "cache_attention": self.profiler.get_duration("cache_attention"),
            "prepare_rot_mat_for_prefill": self.profiler.get_duration("prepare_rot_mat_for_prefill"),
            "prepare_input_decode": self.profiler.get_duration("prepare_input_decode"),
            "decode_and_argmax": self.profiler.get_duration("decode_and_argmax"),
            "Total compile time": compile_prefill_time + compile_decode_time,
            "Full demo runtime": self.profiler.get_duration("run"),
        }

        # Print some of the perf metrics as well
        logger.info("---")
        logger.info(f"Performance metrics for batch 0")
        logger.info(f"Prefill compile time: {round(measurements['compile_prefill'], 4)}s")
        logger.info(f"Decode compile time: {round(measurements['compile_decode'], 4)}s")
        logger.info(f"Prefill inference time per user: {round(inference_prefill_time/self.num_users_generated_prefill, 4)}s")
        # logger.info(
        #     f"Total Decode inference time ({max_generated_tokens-1} iterations): {round(measurements['inference_decode'], 4)}s"
        # )
        logger.info(
            f"Average Decode inference time per user: {round(inference_decode_time / num_tokens_generated_decode, 4)}s"
        )
        logger.info("---")
        logger.info(f"Time to first token: {round(measurements['prefill_time_to_token']* 1000, 4)}ms")
        logger.info(f"Average tokens/sec/user: {round(measurements['decode_t/s/u'], 2)}")

        target_prefill_ts = 5000  # TODO update target
        target_decode_ts = 1056
        decode_tsu = 33
        targets = {"prefill_t/s": target_prefill_ts, "decode_t/s": target_decode_ts, "decode_t/s/u": decode_tsu}

        benchmark_data = create_benchmark_data(self.profiler, measurements, N_warmup_iter, targets)
        benchmark_data.output_folder = self.log_cache
        benchmark_data.prep_csvs(
            self.profiler,
            run_type=f"demo_with_prefill",
            ml_model_name="Llama3.1-8B",
            ml_model_type="llm",
            num_layers=self.model_args.n_layers,
            batch_size=self.batch_size,
            input_sequence_length=self.prefill_seq_len,
            output_sequence_length=1,
            # config_params=,
            # precision=,
        )

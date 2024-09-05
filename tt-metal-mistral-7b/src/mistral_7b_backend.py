# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import time
import traceback
from multiprocessing import Queue

# from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.utils import top_k_top_p_filtering

if not os.environ.get("MOCK_MODEL"):
    import ttnn
    from tt_metal_impl.tt.mistral_common import (
        prepare_inputs_ttnn,
        sample,
        precompute_freqs,
        freqs_to_rotation_matrix,
    )
    from tt_metal_impl.tt.mistral_model import TtTransformer
    from tt_metal_impl.tt.mistral_embedding import TtMistralEmbedding
    from tt_metal_impl.tt.model_config import TtModelArgs
    from tt_metal_impl.reference.tokenizer import Tokenizer


from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")



def preprocess_inputs(input_prompts, tokenizer, model_args, dtype, embd, instruct, device):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    if instruct:
        # Pre append [INST] and post append [/INST] to the encoded prompts if instruct mode
        encoded_prompts = [tokenizer.encode("[INST] " + prompt + " [/INST]") for prompt in input_prompts]
    else:
        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]

    prompt_lens = [len(x) for x in encoded_prompts]

    # Pad the inputs to the max length prompt
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long)

    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)
    input_mask = input_tokens != tokenizer.pad_id

    num_users = len(encoded_prompts)
    logger.info(f"# of users: {num_users}")

    seqlen = 1  # Generating one token per user at a time
    # Select the first token from the prompts for initial decoding
    pt_tokenized_inputs = torch.tensor(input_tokens)
    emb_inputs = embd(pt_tokenized_inputs[:, 0]).view(model_args.max_batch_size, seqlen, -1)

    # Return the rotational embedding matrix on device
    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)

    rot_emb_matrix_list = []
    for i in range(rot_emb_matrix.shape[0]):
        rot_emb_matrix_list.append(
            ttnn.from_torch(
                rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
            )
        )  # ttnn.bfloat16

    return emb_inputs, pt_tokenized_inputs, input_mask, rot_emb_matrix_list


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


class UserInfo:
    def __init__(self, user_id, prompt, position_id, params, tokenizer):
        self.user_id = user_id
        self.prompt = prompt
        self.position_id = position_id
        self.num_tokens_generated = 0
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
        self.stop_sequence = None
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(params.get("stop_sequence"))
        # note: sentecepiece tokenizer decode doesnt handle spaces directly
        # see: https://github.com/google/sentencepiece/blob/master/python/src/sentencepiece/__init__.py#L776
        # users must aggregate all generated tokens and decode full text each time
        # then send new chars
        self.generated_tokens = []
        self.num_generated_chars = 0


class PrefillDecodeBackend:
    def __init__(
        self,
        model_version,
        batch_size,
        num_layers,
        max_seq_len,
        cache_root,
        verbose=False,
    ) -> None:
        """
        Initialize pybuda model and all infracstructures to continuously run decode
        Maintain a cur_prompts for decode.
        """
        self.max_users = 32
        self.num_users = None
        self.users = [None for _ in range(self.max_users)]
        self.use_cache = True
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.num_steps = 0
        self.verbose = verbose  # enable conditional debug logging
        self.model_version = model_version
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.default_top_p = inference_config.model_config.default_top_p
        self.default_top_k = inference_config.model_config.default_top_k
        self.default_temperature = inference_config.model_config.default_temperature
        self.timestamps_start = {}
        self.timestamps_stop = {}
        self.enable_profile_logging = True
        self.device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # tt-metal init
        self.dtype = ttnn.bfloat8_b
        self.instruct_mode = True
        if not os.environ.get("MOCK_MODEL"):
            self.init_tt_metal()
        self.iteration = 0
        self.rot_emb_matrix_list = []

    def get_users(self):
        return [u for u in self.users if u]

    def get_user_param(self, param):
        return [
            user.generation_params[param] if user is not None else None
            for user in self.users
        ]

    def timer_start(self, name):
        self.timestamps_start[name] = time.time()

    def timer_stop(self, name, log=False):
        if name in self.timestamps_start.keys():
            self.timestamps_stop[name] = time.time()
            timedelta = self.timestamps_stop[name] - self.timestamps_start[name]
            if log or self.enable_profile_logging:
                print(f"timedelta: {name}: {timedelta} seconds")
                logger.info(f"timedelta: {name}: {timedelta} seconds")

    def model_location_generator(self, model_version, model_subdir=""):
        model_cache_path = Path(self.cache_root) / "tt-metal-models" / model_version
        model_cache_path.mkdir(parents=True, exist_ok=True)
        return model_cache_path

    def get_tt_cache_path(self, model_version, model_subdir="", default_dir=""):
        tt_cache_path = Path(self.cache_root) / "tt-metal-cache" / model_version
        tt_cache_path.mkdir(parents=True, exist_ok=True)
        return tt_cache_path

    def teardown(self):
        logger.info("teardown ...")
        if not os.environ.get("MOCK_MODEL"):
            self.teardown_tt_metal_device()

    def teardown_tt_metal_device(self):
        logger.info("teardown_tt_metal_device ...")
        import tt_lib as ttl

        ttl.device.DumpDeviceProfiler(self.device, True)
        ttl.device.DeallocateBuffers(self.device)
        ttl.device.Synchronize(self.device)
        ttl.device.CloseDevice(self.device)

    def init_tt_metal_device(self):
        import tt_lib as ttl

        logger.info("init_tt_metal_device ...")
        device_ids = ttnn.get_device_ids()
        device_id = device_ids[0]
        num_devices = ttl.device.GetNumPCIeDevices()
        assert device_id < num_devices, "CreateDevice not supported for non-mmio device"
        self.device = ttl.device.CreateDevice(device_id)
        ttl.device.SetDefaultDevice(self.device)
        self.device.enable_program_cache()

    def init_tt_metal(self):
        self.init_tt_metal_device()

        logger.info("init_tt_metal model ...")
        model_base_path = Path(self.cache_root) / "mistral-7b-instruct"
        self.model_args = TtModelArgs(
            self.device, model_base_path=model_base_path, instruct=self.instruct_mode
        )
        self.tokenizer = Tokenizer(self.model_args.tokenizer_path)

        logger.info("Loading weights...")
        state_dict = torch.load(self.model_args.consolidated_weights_path)
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if (
                any([f"layers.{i}." in k for i in range(self.model_args.n_layers)])
                or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
            )
        }
        logger.info("Loading weights finished!")

        # TODO Should we keep initial embedding on host?
        self.embd = Emb()
        self.embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

        self.generation_start_pos = 0
        # needs full batchsize inputs always
        compile_prompts = ["COMPILE_PROMPT"] * self.batch_size

        # Preprocess initial prompt inputs
        (
            tt_decode_input,
            pt_encoded_input,
            input_mask,
            self.rot_emb_matrix_list,
        ) = preprocess_inputs(
            compile_prompts,
            self.tokenizer,
            self.model_args,
            self.dtype,
            self.embd,
            self.instruct_mode,
            self.device,
        )

        if self.instruct_mode:
            self.tokenizer._model.pad_id = self.tokenizer._model.eos_id

        # Load TTNN mistral model
        logger.info("Loading weights to device...")
        self.tt_model = TtTransformer(
            args=self.model_args,
            device=self.device,
            dtype=self.dtype,
            state_dict=state_dict,
            weight_cache_path=self.model_args.weight_cache_path(
                self.dtype, instruct=self.instruct_mode
            ),
            layers=list(range(self.model_args.n_layers)),
            rot_mat=self.rot_emb_matrix_list,
            start_pos=self.generation_start_pos,
        )
        # Load TTNN embedding module
        self.tt_embd = TtMistralEmbedding(
            device=self.device,
            args=self.model_args,
            weight_cache_path=self.model_args.weight_cache_path(
                self.dtype, instruct=self.instruct_mode
            ),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )
        logger.info("Finished loading weights to device. Starting inference...")

    def _get_user_by_id(self, user_id):
        for user in self.users:
            if user is not None and user.user_id == user_id:
                return user
        return None

    def _get_num_of_users(self):
        # find num of non None users
        return sum([1 for user in self.users if user is not None])

    def _find_free_user_slot(self):
        """return the index of the first free user slot"""
        for i, user in enumerate(self.users):
            if user is None:
                return i

    def _add_users_from_non_empty_queue(self, prompt_q):
        """add users from prompt_q to self.users"""
        while not prompt_q.empty() and self._get_num_of_users() < self.max_users:
            user_id, prompt, params = prompt_q.get()

            # Cancel on special stop token
            if prompt == "<|stop|>":
                if any(
                    (user is not None) and (user_id == user.user_id)
                    for user in self.users
                ):
                    logger.info(f"Cancelling input from user {user_id}")
                    self._get_user_by_id(user_id).cancel = True
                else:
                    logger.info(f"Unexpected cancelling for non-activte user {user_id}")
                continue

            # Don't accept a prompt from a user that's already being procesed
            if any(
                (user is not None) and (user_id == user.user_id) for user in self.users
            ):
                logger.warning(f"Ignoring duplicate input from user {user_id}")
                continue

            user_info = UserInfo(user_id, prompt, 0, params, self.tokenizer)
            idx = self._find_free_user_slot()
            self.users[idx] = user_info
            if self.verbose:
                logger.debug(
                    f"Added user {user_id} to slot {idx} with prompt: {prompt}"
                )

    def pick_prompts(self, prompt_q: Queue):
        if self._get_num_of_users() == self.max_users:
            return

        if self._get_num_of_users() == 0:
            # no users generating currently
            while prompt_q.empty():
                # wait for users
                time.sleep(0.02)
            # batch start delay
            time.sleep(0.5)
            self._add_users_from_non_empty_queue(prompt_q)
        else:
            if prompt_q.empty():
                # no users to add
                return
            else:
                self._add_users_from_non_empty_queue(prompt_q)

        # Check for duplicate user_ids and log it
        user_ids = [user.user_id for user in self.users if user is not None]
        if len(user_ids) != len(set(user_ids)):
            logger.warning(f"WARNING: Duplicate user ids: {user_ids}")

    def prepare_inputs(self):
        # input_prompts = [user_info.prompt for user_info in self.users if user_info]
        # note: current implementation assumes full 32 prompts input always
        # breakpoint()
        input_prompts = [
            user_info.prompt if user_info is not None else ""
            for user_info in self.users
        ]
        self.timer_start("preprocess_inputs")
        (
            self.tt_decode_input,
            self.pt_encoded_input,
            self.input_mask,
            self.rot_emb_matrix_list,
        ) = preprocess_inputs(
            input_prompts,
            self.tokenizer,
            self.model_args,
            self.dtype,
            self.embd,
            self.instruct_mode,
            self.device,
        )
        self.timer_stop("preprocess_inputs")
        self.iteration = 0

    def prefill(self):
        # prefill via decode
        pass

    def decode(self):
        curr_pos = self.generation_start_pos + self.iteration
        self.timer_stop("all_but_decode")
        self.timer_start("decode_preprocessing")
        decode_input, current_pos = prepare_inputs_ttnn(
            self.tt_decode_input,
            curr_pos,
            self.model_args.dim,
            self.model_args.sliding_window,
            self.tt_model.device,
        )
        self.timer_stop("decode_preprocessing")
        self.timer_start("decode")
        # Run ttnn mistral model
        tt_out = self.tt_model(decode_input, current_pos)
        self.timer_stop("decode")
        self.timer_start("decode_get_logits")
        tt_output_torch = (
            ttnn.to_torch(tt_out).permute(2, 1, 0, 3).squeeze(1)
        )  # [batch, seq, hidden_dim]
        self.timer_stop("decode_get_logits")
        self.timer_start("token_selection")

        # TODO argmax on device
        # tt_out = ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)
        # tt_out = ttnn.permute(tt_out, (2, 1, 0, 3))
        # tt_out = ttnn.reshape(tt_out, (tt_out.shape[0], tt_out.shape[2], tt_out.shape[3]))  # Squeeze(1)
        # tt_out_argmax = ttnn.experimental.tensor.argmax(tt_out, dim=-1)
        # Typecast from bf16 to uint32 for embedding
        # tt_out_tok = ttnn.clone(tt_out_argmax, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.uint32)
        # tt_out_tok = ttnn.experimental.tensor.typecast(tt_out_tok, dtype=ttnn.uint32)
        
        out_tok = self.select_tokens(
            logits=tt_output_torch,
            skip_token=self.tokenizer.eos_id,
        ).reshape([self.batch_size, 1])

        self.timer_stop("token_selection")
        self.timer_start("embeddings_on_device")
        # TODO send tensor to host can be remove when argmax on device is working
        tt_out_tok = ttnn.from_torch(
            out_tok,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.tt_decode_input = self.tt_embd(tt_out_tok)
        self.timer_stop("embeddings_on_device")
        self.iteration += 1
        self.timer_start("all_but_decode")

    def select_tokens(
        self,
        logits,
        skip_token,
        return_probs=False,
    ):
        out_tokens = []
        for idx, user in enumerate(self.users):
            if not user:
                # skip None users, fill with skip token
                token = torch.tensor([skip_token])
            elif not user.prefill_complete:
                token = self.pt_encoded_input[idx, self.iteration].unsqueeze(0)
                if user.return_prompt:
                    user.generated_tokens.append(token.item())
                    user.num_tokens_generated += 1
                # TODO: better way of counting prefill that handles input mask being non-contiguous
                if self.iteration == (
                    torch.count_nonzero(self.input_mask[idx]).item() - 1
                ):
                    user.prefill_complete = True
            elif user.decode_complete:
                logger.error(
                    f"user.decode_complete={user.decode_complete}, and is still generating. Should be None"
                )
            else:
                token = top_pk_logits_efficient(
                    logits[idx],
                    user.generation_params.get("top_p"),
                    user.generation_params.get("top_k"),
                    user.generation_params.get("temperature"),
                    return_probs=return_probs,
                    skip_token=skip_token,
                )
                user.generated_tokens.append(token.item())
                user.num_tokens_generated += 1
                if token == self.tokenizer.eos_id:
                    user.decode_complete = True
                elif user.num_tokens_generated > user.max_tokens:
                    user.decode_complete = True
                elif (user.stop_sequence is not None) and (token == user.stop_sequence):
                    user.decode_complete = True
            out_tokens.append(token)
        return torch.concat(out_tokens)

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for idx, user in enumerate(self.users):
            if user is None or not user.generated_tokens:
                continue
            if user.generated_tokens[-1] == self.tokenizer.eos_id:
                # must pass end_of_sequence_str to frontend to close response
                out_text = inference_config.end_of_sequence_str
            else:
                full_text = self.tokenizer.decode(user.generated_tokens)
                out_text = full_text[user.num_generated_chars :]
                user.num_generated_chars = len(full_text)
            out = (user.user_id, out_text)
            output_q.put(out)
            if (
                user.decode_complete
                and out_text != inference_config.end_of_sequence_str
            ):
                # send eos str to frontend in all cases
                output_q.put((user.user_id, inference_config.end_of_sequence_str))
            if self.verbose:
                logger.debug(f"user_id:{user.user_id}, {out_text}")

    def reset_user_memory(self, idx, user):
        # not needed for this implementation
        pass

    def log_user_stats(self, idx, user):
        # TODO: record user stats, e.g. prompt length, num generated tokens, time
        pass

    def update_users(self):
        for idx, user in enumerate(self.users):
            if user is None or not user.generated_tokens:
                continue
            token_id = user.generated_tokens[-1]
            if (token_id == self.tokenizer.eos_id) or user.decode_complete:
                if not user.decode_complete:
                    logger.error(
                        f"user_id: {user.user_id} from index {idx} had EOS token but decode_complete=False."
                    )
                if not (token_id == self.tokenizer.eos_id):
                    logger.error(
                        f"user_id: {user.user_id} from index {idx} did not have EOS token but decode_complete=True."
                    )
                if self.verbose:
                    logger.debug(
                        f"Evicted user_id: {user.user_id} from index {idx} in user list"
                    )
                self.reset_user_memory(idx, user)
                self.log_user_stats(idx, user)
                self.users[idx] = None

    def send_status(self, prompt_q, status_q):
        if time.time() - self.time_last_status > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
            )
            status_q.put(cur_status)
            # udpate cur time
            self.time_last_status = time.time()

    def run_generate(self, prompt_q, output_q, status_q,run_once=False):
        """
        Continuously pop prompt from prompt_q and push generated tokens to output_q
        while running decode. Automatically swap users from prompt_q
        prompt_q: {'user_id1': 'prompt1', 'user_id2': 'prompt2'...}
        output_q: {'user_id1': 'generated_1', 'user_id3': 'generated_1', 'user_id1': 'generated_2'...}
        """
        logger.info("starting run_generate ...")
        LOOP_FOREVER = True
        while LOOP_FOREVER:
            if self.verbose:
                logger.debug(f"run_generate step: {self.num_steps}")
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prepare_inputs()
            # if any([not user.prefill_complete for user in self.get_users()]):
            #     self.prefill()
            logger.info("Running inference decode and pushing results ...")
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.num_steps += 1
            if run_once:
                break


def top_pk_logits_efficient(
    logits,
    p,
    k,
    temperature,
    return_probs=False,
    skip_token=11,
):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    # replace any nans with 0's
    top_k_values = torch.where(
        torch.isnan(top_k_values), torch.zeros_like(top_k_values), top_k_values
    )
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    return token


def run_backend(prompt_q, output_q, status_q, verbose=True, run_once=False):
    logger.info("starting run_backend ...")
    with torch.no_grad():
        backend = PrefillDecodeBackend(
            model_version=inference_config.model_config.model_version,
            batch_size=inference_config.model_config.batch_size,
            num_layers=inference_config.model_config.num_layers,
            max_seq_len=inference_config.model_config.max_seq_len,
            cache_root=inference_config.cache_root,
            verbose=verbose,
        )
        try:
            # run generate
            backend.run_generate(prompt_q, output_q, status_q, run_once)
        except Exception as e:
            logger.error(e)
            # Capture the stack trace
            stack_trace = traceback.format_exc()
            logger.error(stack_trace)
            # Re-raise the exception if you want the process to exit with an error
            raise e
        finally:
            backend.teardown()

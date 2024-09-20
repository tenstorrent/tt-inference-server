# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

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
import json

import torch
import torch.nn.functional as F
from transformers.generation.utils import top_k_top_p_filtering

import ttnn

# import tt_lib as ttl
from ttnn import ReplicateTensorToMesh

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
    ChatFormat,
    Message,
)
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_mesh_device,
    setup_llama_env,
)
from models.demos.t3000.llama2_70b.tt.llama_generation import (
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.demos.t3000.llama2_70b.demo.demo_continuous_batching import (
    ModelArgs,
    TTArgs,
    DataArgs,
    DemoArgs,
    construct_arg,
    build_generator,
    initialize_prefill_input,
    initialize_decode_input,
)
from conftest import get_dispatch_core_type

from model_weights_handler import get_model_weights_and_tt_cache_paths
from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


def get_t3k_mesh_device(num_devices_requested):
    logger.info("get_t3k_mesh_device ...")
    assert ttnn.get_num_devices() == 8
    assert num_devices_requested == 8
    # device_params is empty dict in llama3 70B demo pytest execution
    device_params = {}
    mesh_shape = ttnn.MeshShape(1, num_devices_requested)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape,
        dispatch_core_type=get_dispatch_core_type(),
        **device_params,
    )
    logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
    return mesh_device


def close_t3k_mesh_device(mesh_device):
    for device in mesh_device.get_devices():
        device.disable_and_clear_program_cache()
        ttnn.DumpDeviceProfiler(device)
    ttnn.close_mesh_device(mesh_device)
    del mesh_device


class UserRow:
    def __init__(
        self,
        user_id,
        user_index,
        prompt,
        rag_context,
        context_tokens,
        params,
        tokenizer,
        max_context: int = inference_config.model_config.max_seq_len,
        send_eot: bool = True,
        send_end_seq: bool = True,
        send_user_stats: bool = inference_config.send_user_stats,
    ):
        self.user_id = user_id
        self.user_index = user_index
        self.prompt = prompt
        self.rag_context = rag_context
        self.prompt_tokens = context_tokens
        self.position_id = 0
        self.generated_tokens = []
        self.generated_logits = torch.tensor([])
        self.num_generated_chars = 0
        self.num_tokens_decoded = 0
        self.num_tokens_prefilled = 0
        self.num_prefill_tokens = len(self.prompt_tokens)
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
        self.send_eot = send_eot
        self.send_end_seq = send_end_seq
        self.send_user_stats = send_user_stats
        # timer
        self.prefill_start_time = None
        self.prefill_stop_time = None
        self.prefill_via_decode_start_time = None
        self.prefill_via_decode_stop_time = None
        self.decode_start_time = None
        self.decode_stop_time = None
        self.first_decode_time = None
        self.user_start_time = time.time()
        # this may change for each tokenizer
        self.eos_token_id = tokenizer.eos_id
        self.stop_tokens = tokenizer.stop_tokens
        self.stop_sequence = None
        self.stats = {}
        # include prefill offset to account for prefill gen token and 1st token in decode
        PREFILL_OFFSET = 2
        if self.num_prefill_tokens > (max_context - PREFILL_OFFSET):
            logger.error(
                f"Truncating prompt: user_id:={user_id} has prompt_len:= {self.num_prefill_tokens} > max_context:= {max_context}"
            )
            self.prompt_tokens = self.prompt_tokens[: (max_context - PREFILL_OFFSET)]
            self.num_prefill_tokens = len(self.prompt_tokens)
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(
                params.get("stop_sequence"), bos=False, eos=False
            )

    def timer_start(self, name):
        self.timestamps_start[name] = time.time()

    def timer_stop(self, name, log=False):
        if name in self.timestamps_start.keys():
            self.timestamps_stop[name] = time.time()

    def start_prefill_timer(self):
        self.prefill_start_time = time.time()

    def stop_prefill_timer(self):
        self.prefill_stop_time = time.time()

    def start_prefill_via_decode_timer(self):
        self.prefill_via_decode_start_time = time.time()

    def stop_prefill_via_decode_timer(self):
        self.prefill_via_decode_stop_time = time.time()

    def start_decode_timer(self):
        self.decode_start_time = time.time()

    def stop_decode_timer(self):
        self.decode_stop_time = time.time()

    def get_user_stats(self, batch_size: int, context_length: int, log: bool = True):
        prefill_time = self.prefill_stop_time - self.prefill_start_time
        decode_time = self.decode_stop_time - self.decode_start_time
        ttft_e2e_ms = round((self.first_decode_time - self.user_start_time) * 1000, 0)
        ttft_ms = round((self.first_decode_time - self.prefill_start_time) * 1000, 0)
        user_tps = round(self.num_tokens_decoded / decode_time, 3)
        stats = {
            "user_ttft_ms": ttft_ms,
            "user_tps": user_tps,
            "user_ttft_e2e_ms": ttft_e2e_ms,
            "prefill": {
                "tokens_prefilled": self.num_tokens_prefilled,
                "tps": round(self.num_tokens_prefilled / prefill_time, 3),
            },
            "decode": {"tokens_decoded": self.num_tokens_decoded, "tps": user_tps},
            "batch_size": batch_size,
            "context_length": context_length,
        }
        if log:
            logger.info(stats)
        return stats


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
        # # inputs to model
        self.batch_token_indices = None
        # backend status
        self.time_last_status = time.time()
        self.update_period = 1  # status message period in seconds
        self.verbose = verbose  # enable conditional debug logging
        # new init:
        self.model_version = model_version
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.default_top_p = inference_config.model_config.default_top_p
        self.default_top_k = inference_config.model_config.default_top_k
        self.default_temperature = inference_config.model_config.default_temperature
        #
        self.timestamps_start = {}
        self.timestamps_stop = {}
        self.timer_sums = defaultdict(int)
        self.enable_profile_logging = False
        self.batch_counter = 0
        self.decode_counter = 0
        self.prev_decode_counter = 0
        self.prefill_seq_len = None
        self.prefill_batch_size = None
        #
        self.t3k_mesh_device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization
        self.decode_only = False
        self.model_config = None
        self.chat = True
        self.batch_token_indices = [0] * self.batch_size
        self.batch_token_inputs = [0] * self.batch_size
        self.page_table_tt = None
        self.init_model()

    def get_users(self):
        return [u for u in self.users if u is not None]

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
            self.timer_sums[name] += timedelta
            if log or self.enable_profile_logging:
                logger.info(f"timedelta: {name}: {timedelta} seconds")

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

    def teardown(self):
        logger.info("teardown ...")
        if self.t3k_mesh_device is not None:
            close_t3k_mesh_device(self.t3k_mesh_device)

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        t3k_mesh_device = get_t3k_mesh_device(
            num_devices_requested=inference_config.n_devices
        )
        check_mesh_device(t3k_mesh_device, self.model_config)
        for i in t3k_mesh_device.get_device_ids():
            device = t3k_mesh_device.get_device(i)
            device.enable_async(True)
            device.enable_program_cache()
        self.t3k_mesh_device = t3k_mesh_device
        logger.info("init_tt_metal_device finished.")

    def init_model(self):
        # set up variables for model init
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
            llama_version="llama3",
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            skip_model_load=False,
            num_layers=self.num_layers,
            max_batch_size=self.batch_size,
            max_kv_context_len=inference_config.model_config.max_seq_len,
            max_output_tokens=inference_config.model_config.max_seq_len,
            prompts_file=None,
            output_at_end=None,
            top_p=None,
            top_k=None,
            temperature=None,
            chat=inference_config.model_config.chat,
            mesh_device=self.t3k_mesh_device,
            n_devices=inference_config.n_devices,
            cache_path=cache_path,
            decode_only=self.decode_only,
        )
        model_args = args.model
        tt_args = args.tt

        generator = build_generator(model_args, tt_args)
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.formatter = ChatFormat(self.tokenizer)

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
            user_id, prompt, rag_context, params = prompt_q.get()
            # Cancel on special stop token
            if prompt == "<|stop|>":
                if any(
                    (user is not None) and (user_id == user.user_id)
                    for user in self.users
                ):
                    logger.info(f"Cancelling input from user {user_id}")
                    self._get_user_by_id(user_id).decode_complete = True
                else:
                    logger.info(f"Unexpected cancelling for non-activte user {user_id}")
                continue

            # Don't accept a prompt from a user that's already being procesed
            if any(
                (user is not None) and (user_id == user.user_id) for user in self.users
            ):
                logger.warning(f"Ignoring duplicate input from user {user_id}")
                continue
            context_tokens = self.tokenize_prompt(prompt, rag_context)
            idx = self._find_free_user_slot()
            user = UserRow(
                user_id=user_id,
                user_index=idx,
                prompt=prompt,
                rag_context=rag_context,
                context_tokens=context_tokens,
                params=params,
                tokenizer=self.tokenizer,
            )
            self.users[idx] = user
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

    def prefill(self):
        for user in [user for user in self.get_users() if not user.prefill_complete]:
            user.start_prefill_timer()

            seq_len = user.num_prefill_tokens
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            tokens = torch.tensor(
                user.prompt_tokens, dtype=torch.long, device="cpu"
            ).unsqueeze(0)
            prefill_ids = torch.cat(
                [tokens, torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )

            logger.info(
                f"Filling kv cache for user_id:= {user.user_index}, prefill_ids.shape:={prefill_ids.shape}"
            )
            logits = self.model.prefill_forward_single_user(
                prefill_ids,
                start_pos=0,
                user_id=user.user_index,
                last_token_idx=last_token_idx,
                page_table=None,
                kv_cache=None,
            )
            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits = logits[:, last_token_idx % 32 : last_token_idx % 32 + 1, :]
            next_logits = output_logits[:, -1, :]  # 1, seq_len, vocab -> 1, vocab
            # TODO: add params
            next_token = batch_top_pk_logits_efficient_same_params(
                next_logits,
                p=user.generation_params.get("top_p"),
                k=user.generation_params.get("top_k"),
                temperature=user.generation_params.get("temperature"),
            ).item()  # shape = (1,)
            user.prefill_stop_time = time.time()
            user.generated_tokens.append(next_token)
            user.num_tokens_decoded += 1
            # only record actual prefill tokens for metrics, not padded tokens
            user.num_tokens_prefilled = seq_len
            user.prefill_complete = True
            self.batch_token_inputs[user.user_index] = next_token
            # start index at the end of the actual prefill tokens, not including padding
            self.batch_token_indices[user.user_index] = seq_len
            # TODO: better way to handle more prefill users changing decode time
            user.start_decode_timer()

    def decode(self):
        """
        self.cur_pos is the batch level position
        each user has a generation_pos
        """
        self.decode_counter += 1
        self.timer_start("decode")
        tokens_tensor, indices_tensor = initialize_decode_input(
            self.batch_token_inputs, self.batch_token_indices
        )
        logger.info(f"Decoding batch with indices {self.batch_token_indices}")
        logits = self.model.decode_forward(
            tokens_tensor, indices_tensor, page_table=None
        )
        self.timer_stop("decode", log=False)
        next_tokens = (
            batch_top_pk_logits_efficient(
                logits,
                top_ps=self.get_user_param("top_p"),
                top_ks=self.get_user_param("top_k"),
                temperatures=self.get_user_param("temperature"),
            )
            .reshape(self.batch_size)
            .tolist()
        )
        self.batch_token_inputs = next_tokens
        for idx, (user, user_decode_id) in enumerate(
            zip(self.users, self.batch_token_inputs)
        ):
            if user is None:
                continue

            if user.num_tokens_decoded == 1:
                # tokens decoded starts at 1 because we are using the
                # decoded token from the prefill
                user.first_decode_time = time.time()
            user.num_tokens_decoded += 1
            user.generated_tokens.append(user_decode_id)
            self.batch_token_indices[idx] += 1
            if user_decode_id in user.stop_tokens:
                # generated stop token
                user.decode_complete = True
                if not user.send_eot:
                    # remove eot token
                    user.generated_tokens.pop()
            elif user.num_tokens_decoded > user.max_tokens:
                # request specified max generation
                user.decode_complete = True
            elif (
                user.num_tokens_decoded + user.num_tokens_prefilled
            ) == self.max_seq_len:
                # reached max context length
                user.decode_complete = True
            elif user.stop_sequence is not None:
                # check request specified stop_sequence
                last_n_tokens = user.generated_tokens[-(len(user.stop_sequence) - 1) :]
                last_n_tokens.append(user_decode_id)
                if last_n_tokens == user.stop_sequence:
                    user.decode_complete = True

            if user.decode_complete:
                # user just finished
                user.stop_decode_timer()
                user.stats = user.get_user_stats(
                    batch_size=self.batch_size, context_length=self.max_seq_len
                )

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for user, user_decode_id in zip(self.users, self.batch_token_indices):
            if user is None:
                continue
            full_text = self.tokenizer.decode(user.generated_tokens)
            return_text = full_text[user.num_generated_chars :]
            user.num_generated_chars = len(full_text)
            if user.decode_complete:
                if user.send_end_seq:
                    # send special EOS string to frontend to delineate end of response
                    # TODO: add handling in client logs for when this is not set
                    return_text += inference_config.end_of_sequence_str
                if user.send_user_stats:
                    # optionally send user stats, add extra EOS string to delineate
                    return_text += json.dumps(user.stats)
                    return_text += inference_config.end_of_sequence_str
            output_q.put((user.user_id, return_text))
            if self.verbose:
                logger.debug(f"user_id:{user.user_id}, {return_text}")

    def reset_user_slot(self, user_idx, user):
        self.batch_token_indices[user_idx] = 0
        self.batch_token_inputs[user_idx] = 0
        self.users[user_idx] = None

    def update_users(self):
        for idx, token_id in enumerate(self.batch_token_indices):
            if self.users[idx] is None:
                continue

            if self.users[idx].decode_complete:
                self.reset_user_slot(idx, self.users[idx])
            elif (
                token_id in self.users[idx].stop_tokens
                and not self.users[idx].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[idx].user_id} from index {idx} had EOS token but decode_complete=False."
                )
                self.reset_user_slot(idx, self.users[idx])

    def send_status(self, prompt_q, status_q):
        if time.time() - self.time_last_status > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
                self.batch_token_indices,
            )
            status_q.put(cur_status)
            # udpate cur time
            self.time_last_status = time.time()

    def run_generate(self, prompt_q, output_q, status_q, loop_once):
        """
        Continuously pop prompt from prompt_q and push generated tokens to output_q
        while running decode. Automatically swap users from prompt_q
        prompt_q: {'user_id1': 'prompt1', 'user_id2': 'prompt2'...}
        output_q: {'user_id1': 'generated_1', 'user_id3': 'generated_1', 'user_id1': 'generated_2'...}
        stop_event: threading.Event, set to stop safely
        """
        logger.info("starting run_generate ...")
        LOOP_FORVEVER = True
        while LOOP_FORVEVER:
            self.pick_prompts(prompt_q)  # we update to self.users
            self.prefill()
            self.decode()
            self.push_outputs(output_q)
            self.update_users()
            self.send_status(prompt_q, status_q)
            if loop_once:
                break


def batch_top_pk_logits_efficient_multi_params(
    logits,
    top_ps=[0.9],
    top_ks=[10],
    temperatures=[1.0],
    return_probs=False,
    skip_token=11,
):
    """
    Handle top_p and top_k sampling when a given batch has different params.
    This is quite rare as few users send non-default top_p and top_k values.
    """
    out_tokens = []
    for b_logits, p, k, temperature in zip(logits, top_ps, top_ks, temperatures):
        if p is None or k is None:
            # skip None users
            token = torch.tensor([skip_token])
        else:
            token = batch_top_pk_logits_efficient_same_params(
                b_logits, p=p, k=k, temperature=temperature
            )

        out_tokens.append(token)
    return torch.concat(out_tokens)


def batch_top_pk_logits_efficient_same_params(logits, p=0.9, k=40, temperature=1.0):
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


def check_if_all_equal(top_ps, top_ks, temperatures):
    # Remove None values from the lists
    top_ps = [p for p in top_ps if p is not None]
    top_ks = [k for k in top_ks if k is not None]
    temperatures = [t for t in temperatures if t is not None]
    if not top_ps or not top_ks or not temperatures:
        return False
    # Check if all elements in the list are equal
    all_top_ps_equal = all(p == top_ps[0] for p in top_ps)
    all_top_ks_equal = all(k == top_ks[0] for k in top_ks)
    all_temperatures_equal = all(t == temperatures[0] for t in temperatures)
    return all_top_ps_equal and all_top_ks_equal and all_temperatures_equal


def first_non_none(seq):
    return next((x for x in seq if x is not None), None)


def batch_top_pk_logits_efficient(
    logits, top_ps=[0.9], top_ks=[40], temperatures=[1.0]
):
    if check_if_all_equal(top_ps, top_ks, temperatures):
        # logits seq_len dimension is removed
        return batch_top_pk_logits_efficient_same_params(
            logits[:, -1, :],
            p=first_non_none(top_ps),
            k=first_non_none(top_ks),
            temperature=first_non_none(temperatures),
        )
    else:
        return batch_top_pk_logits_efficient_multi_params(
            logits, top_ps=top_ps, top_ks=top_ks, temperatures=temperatures
        )


def run_backend(prompt_q, output_q, status_q, loop_once=False, verbose=True):
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
            backend.run_generate(prompt_q, output_q, status_q, loop_once)
        except Exception as e:
            logger.error(e)
            # Capture the stack trace
            stack_trace = traceback.format_exc()
            logger.error(stack_trace)
            # Re-raise the exception if you want the process to exit with an error
            raise e
        finally:
            backend.teardown()

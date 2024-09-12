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
from conftest import get_dispatch_core_type

from inference_config import inference_config
from inference_logger import get_logger
from device_manager import DeviceManager, DeviceType

from model_adapters.llama3_1_8b_n150 import Llama3_1_8B_N150
from model_adapters.llama3_1_70b_t3k import Llama3_70B_T3K

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


class UserRow:
    def __init__(
        self,
        user_id,
        prompt,
        rag_context,
        context_tokens,
        params,
        tokenizer,
    ):
        self.user_id = user_id
        self.prompt = prompt
        self.rag_context = rag_context
        self.prompt_tokens = context_tokens
        self.position_id = 0
        self.generated_tokens = []
        self.generated_logits = torch.tensor([])
        self.num_tokens_decoded = 0
        self.num_tokens_prefilled_via_decode = 0
        self.num_tokens_prefilled = 0
        self.num_prefill_tokens = len(self.prompt_tokens)
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
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

    def get_user_stats(self, log=True):
        prefill_time = self.prefill_stop_time - self.prefill_start_time
        decode_time = self.decode_stop_time - self.decode_start_time
        ttft_e2e_ms = round((self.first_decode_time - self.user_start_time) * 1000, 0)
        ttft_ms = round((self.first_decode_time - self.prefill_start_time) * 1000, 0)
        user_tps = round(self.num_tokens_decoded / decode_time, 3)
        if self.prefill_via_decode_start_time:
            prefill_via_decode_time = (
                self.prefill_via_decode_stop_time - self.prefill_via_decode_start_time
            )
            stats_prefill_via_decode = (
                {
                    "tokens_prefilled_via_decode": self.num_tokens_prefilled_via_decode,
                    "tps": round(
                        self.num_tokens_prefilled_via_decode / prefill_via_decode_time,
                        3,
                    ),
                },
            )
        else:
            assert self.num_tokens_prefilled_via_decode == 0
            stats_prefill_via_decode = {
                "tokens_prefilled_via_decode": self.num_tokens_prefilled_via_decode,
                "tps": "nan",
            }
        stats = {
            "user_ttft_ms": ttft_ms,
            "user_tps": user_tps,
            "user_ttft_e2e_ms": ttft_e2e_ms,
            "prefill": {
                "tokens_prefilled": self.num_tokens_prefilled,
                "tps": round(self.num_tokens_prefilled / prefill_time, 3),
            },
            "prefill_via_decode": stats_prefill_via_decode,
            "decode": {"tokens_decoded": self.num_tokens_decoded, "tps": user_tps},
        }
        if log:
            logger.info(stats)
        return


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
        self.max_users = batch_size
        self.num_users = None
        self.users = [None for _ in range(self.max_users)]
        self.use_cache = True
        # # inputs to model
        self.decode_ids = None
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
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization
        self.decode_only = False
        self.max_prompt_len = None
        self.model_config = None
        self.chat = True
        self.device_manager = DeviceManager(
            model_name=inference_config.model_config.model_name,
            device_type_str=inference_config.model_config.device_type,
        )
        self.device = self.device_manager.open_device()
        self.model = self.init_model()

    def init_model(self):
        key = (
            inference_config.model_config.model_name,
            self.device_manager.device_type,
        )
        model_class_map = {
            ("llama-3.1-8b-instruct", DeviceType.n150): Llama3_1_8B_N150,
            ("llama-3.1-8b", DeviceType.n150): Llama3_1_8B_N150,
            ("llama-3-8b-instruct", DeviceType.n150): Llama3_1_8B_N150,
            ("llama-3-8b", DeviceType.n150): Llama3_1_8B_N150,
            ("llama-3.1-70b-instruct", DeviceType.t3k_mesh_device): Llama3_70B_T3K,
            ("llama-3.1-70b", DeviceType.t3k_mesh_device): Llama3_70B_T3K,
            ("llama-3-70b-instruct", DeviceType.t3k_mesh_device): Llama3_70B_T3K,
            ("llama-3-70b", DeviceType.t3k_mesh_device): Llama3_70B_T3K,
            ("llama-3.1-8b-instruct", DeviceType.cpu): Llama3_1_8B_N150,
            ("llama-3.1-8b", DeviceType.cpu): Llama3_1_8B_N150,
            ("llama-3-8b-instruct", DeviceType.cpu): Llama3_1_8B_N150,
            ("llama-3-8b", DeviceType.cpu): Llama3_1_8B_N150,
            ("llama-3.1-70b-instruct", DeviceType.cpu): Llama3_70B_T3K,
            ("llama-3.1-70b", DeviceType.cpu): Llama3_70B_T3K,
            ("llama-3-70b-instruct", DeviceType.cpu): Llama3_70B_T3K,
            ("llama-3-70b", DeviceType.cpu): Llama3_70B_T3K,
        }
        model_class = model_class_map[key]
        # instantiate model
        model = model_class(
            device=self.device,
            inference_config=inference_config,
        )
        return model

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

    def teardown(self):
        logger.info("teardown ...")
        if self.device is not None:
            self.device_manager.close_device(self.device)

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        self.device = self.device_manager.get_device(
            num_devices_requested=inference_config.n_devices,
            enable_async=True,
            enable_program_cache=True,
        )
        logger.info("init_tt_metal_device finished.")

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
            context_tokens = self.model.tokenize_prompt(prompt, rag_context)
            user = UserRow(
                user_id=user_id,
                prompt=prompt,
                rag_context=rag_context,
                context_tokens=context_tokens,
                params=params,
                tokenizer=self.model.tokenizer,
            )
            idx = self._find_free_user_slot()
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

    def batch_preprocessing(self):
        # TODO: investigate changing when continous batching supported
        # note: the cur_pos index if shared between all users
        # this may change for the continuous batching implementation
        self.batch_start_time = time.time()
        self.num_users = len(self.get_users())
        assert self.num_users <= self.max_users
        self.max_prompt_len = max(
            [user.num_prefill_tokens for user in self.get_users()]
        )
        self.min_prompt_len = min(
            [user.num_prefill_tokens for user in self.get_users()]
        )
        self.prepare_batch_inputs()

        self.cur_pos = 0
        self.batch_counter += 1

    def prepare_batch_inputs(self):
        input_prompts = [user.prompt_tokens for user in self.get_users()]
        input_prompt_strs = [user.prompt for user in self.get_users()]
        # pad inputs, empty users get pad id
        prefill_tokens = self.model.initialize_inputs(input_prompt_strs)
        self.prefill_ids = prefill_tokens
        # decode_ids are padded to batch_size
        decode_ids = torch.full(
            (self.batch_size, 1),
            self.model.tokenizer.pad_id,
            dtype=torch.long,
            device="cpu",
        )
        if prefill_tokens.numel() > 0:
            decode_ids[: self.num_users, :1] = prefill_tokens[:, :1].clone()
        self.decode_ids = decode_ids

    def prefill(self):
        self.timer_start("prefill")
        self.model.prefill()
        self.timer_stop("prefill")
        self.prefill_seq_len = self.model.prefill_seq_len
        self.prefill_batch_size = self.model.batch_size

    def start_decode_loop(self):
        for user in self.get_users():
            if user.prefill_complete:
                user.start_decode_timer()
        self.timer_start("decode_batch")
        logger.info("Running inference decode and pushing results ...")

    def decode(self):
        """
        self.cur_pos is the batch level position
        each user has a generation_pos
        """
        self.decode_counter += 1
        self.timer_start("decode")
        next_tokens = self.model.decode(self.decode_ids, self.cur_pos)
        self.timer_stop("decode", log=False)
        self.decode_ids = next_tokens
        for idx, (user, user_decode_id) in enumerate(
            zip(self.users, self.decode_ids.reshape(self.batch_size).tolist())
        ):
            if user is None:
                continue

            if not user.prefill_complete:
                user.num_tokens_prefilled_via_decode += 1
                prefill_via_decode_idx = (
                    user.num_tokens_prefilled + user.num_tokens_prefilled_via_decode
                )
                self.decode_ids[idx][0] = user.prompt_tokens[prefill_via_decode_idx - 1]
                if prefill_via_decode_idx >= user.num_prefill_tokens:
                    user.stop_prefill_via_decode_timer()
                    user.prefill_complete = True
                    # overwrite decode timer for user
                    user.start_decode_timer()
            else:
                if user.num_tokens_decoded == 0:
                    user.first_decode_time = time.time()
                user.num_tokens_decoded += 1
                user.generated_tokens.append(user_decode_id)
                if user_decode_id in user.stop_tokens:
                    # generated stop token
                    user.decode_complete = True
                elif user.num_tokens_decoded > user.max_tokens:
                    # request specified max generation
                    user.decode_complete = True
                elif (
                    user.num_tokens_decoded
                    + user.num_tokens_prefilled
                    + user.num_tokens_prefilled_via_decode
                ) == self.max_seq_len:
                    # reached max context length
                    user.decode_complete = True
                elif user.stop_sequence is not None:
                    # check request specified stop_sequence
                    last_n_tokens = user.generated_tokens[
                        -(len(user.stop_sequence) - 1) :
                    ]
                    last_n_tokens.append(user_decode_id)
                    if last_n_tokens == user.stop_sequence:
                        user.decode_complete = True

                if user.decode_complete:
                    # user just finished
                    self.decode_ids[idx][0] = user.eos_token_id
                    user.stop_decode_timer()
                    user.get_user_stats()

        self.cur_pos += 1

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for user, user_decode_id in zip(self.users, self.decode_ids):
            if user is None:
                continue
            elif user.num_tokens_decoded < 1:
                # still prefilling via decode
                continue
            last_token = user_decode_id.item()
            full_text = self.model.tokenizer.decode(user.generated_tokens)
            return_text = full_text[user.num_generated_chars :]
            user.num_generated_chars = len(full_text)
            # send special EOS string to frontend
            if (last_token in user.stop_tokens) or (user.decode_complete):
                return_text = inference_config.end_of_sequence_str
            output_q.put((user.user_id, return_text))
            if self.verbose:
                logger.debug(f"user_id:{user.user_id}, {return_text}")

    def reset_user_slot(self, user_idx, user):
        self.decode_ids[user_idx, 0] = 0
        self.users[user_idx] = None

    def update_users(self):
        for idx, token_id in enumerate(
            self.decode_ids.reshape(self.batch_size).tolist()
        ):
            if self.users[idx] is None:
                continue

            if (
                token_id in self.users[idx].stop_tokens
                and self.users[idx].decode_complete
            ):
                self.reset_user_slot(idx, self.users[idx])
            elif (
                token_id in self.users[idx].stop_tokens
                and not self.users[idx].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[idx].user_id} from index {idx} had EOS token but decode_complete=False."
                )
                self.reset_user_slot(idx, self.users[idx])
            elif (
                token_id not in self.users[idx].stop_tokens
                and self.users[idx].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[idx].user_id} from index {idx} did not have EOS token but decode_complete=True."
                )
                self.reset_user_slot(idx, self.users[idx])

    def get_batch_stats(self, log=True):
        self.timer_stop("decode_batch")
        batch_duration = time.time() - self.batch_start_time

        # actual prefill tokens
        prefill_batch_tokens = self.prefill_batch_size * self.prefill_seq_len
        prefill_time = (
            self.timestamps_stop["prefill"] - self.timestamps_start["prefill"]
        )

        # prefill-via-decode + decode generation tokens
        decode_batches = self.decode_counter - self.prev_decode_counter
        decode_batch_tokens = decode_batches * self.batch_size
        decode_batch_e2e_time = (
            self.timestamps_stop["decode_batch"] - self.timestamps_start["decode_batch"]
        )
        decode_batch_time = self.timer_sums["decode"]
        self.timer_sums["decode"] = 0

        self.prev_decode_counter = self.decode_counter

        batch_stats = {
            "batch_counter": self.batch_counter,
            "decode_counter": self.decode_counter,
            "batch_duration": round(batch_duration, 3),
            "batch_users": self.num_users,
            "prefill": {
                "prefill_batch_size": self.prefill_batch_size,
                "prefill_batch_tokens": prefill_batch_tokens,
                "e2e_throughput_tps": round(prefill_batch_tokens / prefill_time, 3),
            },
            "decode": {
                "decode_batch_tokens": decode_batch_tokens,
                "e2e_throughput_tps": round(
                    decode_batch_tokens / decode_batch_e2e_time, 3
                ),
                "e2e_latency_ms": round(
                    (decode_batch_e2e_time / decode_batches) * 1000, 2
                ),
                "decode_throughput_tps": round(
                    decode_batch_tokens / decode_batch_time, 3
                ),
                "decode_latency_ms": round(
                    (decode_batch_time / decode_batches) * 1000, 2
                ),
            },
        }
        if log:
            logger.info(batch_stats)
        return batch_stats

    def send_status(self, prompt_q, status_q):
        if time.time() - self.time_last_status > self.update_period:
            # send status queue which includes the (length of the prompt_q, the number of users being decoded rn, the user_ids being decoded)
            cur_status = (
                prompt_q.qsize(),
                self._get_num_of_users(),
                [user.user_id for user in self.users if user is not None],
                self.cur_pos,
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
            self.batch_preprocessing()
            self.prefill()
            self.start_decode_loop()
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.get_batch_stats(log=True)
            if loop_once:
                break


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

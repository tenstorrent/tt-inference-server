import os
import time
import traceback
import threading
from multiprocessing import Queue
from functools import partial
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers.generation.utils import top_k_top_p_filtering

import ttnn
import tt_lib as ttl

# from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
# from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
# from models.demos.t3000.llama2_70b.tt.llama_common import load_llama_state_dict
from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import ChatFormat, Message
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_device_mesh,
    setup_llama_env,
    # string_similarity_score,
)

from models.demos.t3000.llama2_70b.demo.demo import (
    ModelArgs,
    TTArgs,
    DataArgs,
    DemoArgs,
    construct_arg,
    build_generator,
    # get_sampling_func,
    # initialize_inputs,
    # prepare_next_input,
    # top_pk_logits_efficient,
)
from conftest import get_dispatch_core_type

from model_weights_handler import get_model_weights_and_tt_cache_paths
from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


def get_t3k_device_mesh(num_devices_requested):
    logger.info("get_t3k_device_mesh ...")
    assert ttnn.get_num_devices() == 8
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    # device_params is empty dict in llama3 70B demo pytest execution
    device_params = {}
    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested], dispatch_core_type=get_dispatch_core_type(), **device_params
    )
    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    return device_mesh


def close_t3k_device_mesh(device_mesh):
    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)
    ttnn.close_device_mesh(device_mesh)
    del device_mesh

def initialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t[:total_len], dtype=torch.long, device="cpu").clone().detach()
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask, eos_reached

class UserInfo:
    def __init__(self, user_id, prompt, position_id, params, tokenizer, formatter=None):
        self.user_id = user_id
        self.prompt = prompt
        self.position_id = position_id
        self.num_tokens_decoded = 0
        self.generated_tokens = []
        self.num_generated_chars = 0
        self.num_tokens_prefilled = 0
        self.generation_params = params
        self.max_tokens = params["max_tokens"]
        self.return_prompt = params["return_prompt"]
        self.cancel = False
        self.prefill_complete = False
        self.decode_complete = False
        self.sent_stop = False
        self.chat_format = True
        # timer
        self.prefill_start_time = None
        self.prefill_stop_time = None
        self.decode_start_time = None
        self.decode_stop_time = None
        # this may change for each tokenizer
        self.eos_token_id = tokenizer.eos_id
        self.stop_tokens = tokenizer.stop_tokens
        self.stop_sequence = None
        if params.get("stop_sequence"):
            self.stop_sequence = tokenizer.encode(
                params.get("stop_sequence"), bos=False, eos=False
            )
        # tokenize input here
        if self.chat_format and inference_config.model_config.llama_version == "llama3":
            dialog = [{"role": "user", "content": prompt}]
            self.prompt_tokens = formatter.encode_dialog_prompt(dialog)
        else:
            self.prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
        # strip eos token from prompt
        self.prompt_tokens = [
            tok for tok in self.prompt_tokens if tok not in self.stop_tokens
        ]
        self.num_prefill_tokens = len(self.prompt_tokens)
      
    def start_prefill_timer(self):
        self.prefill_start_time = time.time()

    def stop_prefill_timer(self):
        self.prefill_stop_time = time.time()
    
    def start_decode_timer(self):
        self.decode_start_time = time.time()

    def stop_decode_timer(self):
        self.decode_stop_time = time.time()

    def get_user_stats(self, log=True):
        prefill_time = self.prefill_stop_time - self.prefill_start_time
        decode_time = self.decode_stop_time - self.decode_start_time
        stats = {
            "prefill": {"tokens_prefilled": self.num_tokens_prefilled, "tps": round(self.num_tokens_prefilled/prefill_time, 3)},
            "decode": {"tokens_decoded": self.num_tokens_decoded, "tps": round(self.num_tokens_decoded/decode_time, 3)},
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
        self.max_users = 32
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
        self.enable_profile_logging = False
        self.batch_counter = 0
        self.decode_counter = 0
        self.prev_decode_counter = 0
        self.prefill_seq_len = None
        self.prefill_batch_size = None
        #
        self.device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization
        self.decode_only = False
        self.max_prompt_len = None
        self.model_config = None
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
            if log or self.enable_profile_logging:
                # print(f"timedelta: {name}: {timedelta} seconds")
                logger.info(f"timedelta: {name}: {timedelta} seconds")

    def teardown(self):
        logger.info("teardown ...")
        if self.t3k_device_mesh is not None:
            close_t3k_device_mesh(self.t3k_device_mesh)

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        t3k_device_mesh = get_t3k_device_mesh(
            num_devices_requested=inference_config.n_devices
        )
        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            device.enable_async(True)
            device.enable_program_cache()
        self.t3k_device_mesh = t3k_device_mesh
        check_device_mesh(self.t3k_device_mesh, self.model_config)
        logger.info("init_tt_metal_device finished.")

    def init_model(self):
        # set up variables for model init
        # set weights from tt-studio backend using
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
            num_tokens=None,
            prompts_file=None,
            output_at_end=None,
            top_p=None,
            top_k=None,
            temperature=None,
            chat=inference_config.model_config.chat,
            device_mesh=self.t3k_device_mesh,
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

            user_info = UserInfo(
                user_id, prompt, 0, params, self.tokenizer, formatter=self.formatter
            )
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


    def batch_preprocessing(self):
        # TODO: investigate changing when continous batching supported
        # note: the cur_pos index if shared between all users
        # this may change for the continuous batching implementation
        self.batch_start_time = time.time()
        self.prepare_batch_inputs()
        self.prev_pos = 0
        self.cur_pos = self.prev_pos + 1
        self.batch_counter += 1

    def prepare_batch_inputs(self):
        self.num_users = len(self.get_users())
        assert self.num_users <= self.max_users
        input_prompts = [user_info.prompt_tokens for user_info in self.get_users()]
        self.max_prompt_len = max([user_info.num_prefill_tokens for user_info in self.get_users()])
        self.min_prompt_len = min([user_info.num_prefill_tokens for user_info in self.get_users()])
        # initialize_inputs:
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
        decode_ids = torch.full((self.batch_size, 1), self.tokenizer.pad_id, dtype=torch.long, device="cpu")
        decode_ids[:self.num_users, :1] = prefill_tokens[:, :1].clone()
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
            self.prev_pos = seq_len + 1
            self.cur_pos = self.prev_pos + 1

        self.prefill_complete = True
        for user in self.get_users():
            user.num_tokens_prefilled = self.prefill_seq_len
            if user.num_prefill_tokens <= user.num_tokens_prefilled:
                user.stop_prefill_timer()
                user.prefill_complete = True
            
        self.prefill_ids = None
        self.timer_stop("prefill")

    def start_decode_loop(self):
        for user in self.get_users():
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
        logits = self.model.forward(self.decode_ids, self.prev_pos)
        self.timer_stop("decode")
        self.timer_start("token_selection")
        self.timer_start("batch_top_pk_logits_efficient")
        next_tokens = batch_top_pk_logits_efficient(
            logits,
            top_ps=self.get_user_param("top_p"),
            top_ks=self.get_user_param("top_k"),
            temperatures=self.get_user_param("temperature"),
        ).reshape(self.batch_size, 1)
        self.timer_stop("batch_top_pk_logits_efficient")
        self.decode_ids = next_tokens
        for idx, (user_info, user_decode_id) in enumerate(
            zip(self.users, self.decode_ids.reshape(self.batch_size).tolist())
        ):
            if user_info is None:
                continue

            if not user_info.prefill_complete:
                # take next token for prefill
                self.decode_ids[idx][0] = user_info.prompt_tokens[
                    user_info.num_tokens_prefilled
                ]
                user_info.num_tokens_prefilled += 1
                if user_info.num_tokens_prefilled >= user_info.num_prefill_tokens:
                    user_info.stop_prefill_timer()
                    user_info.prefill_complete = True
                    # overwrite decode timer
                    user_info.start_decode_timer()
            else:
                user_info.num_tokens_decoded += 1
                if user_decode_id in user_info.stop_tokens:
                    # generated stop token
                    user_info.decode_complete = True
                elif user_info.num_tokens_decoded > user_info.max_tokens:
                    # request specified max generation
                    user_info.decode_complete = True
                elif (user_info.num_tokens_decoded + user_info.num_tokens_prefilled) == self.max_seq_len:
                    # reached max context length
                    user_info.decode_complete = True
                elif user_info.stop_sequence is not None:
                    # check request specified stop_sequence
                    last_n_tokens = user_info.generated_tokens[
                        -(len(user_info.stop_sequence) - 1) :
                    ]
                    last_n_tokens.append(user_decode_id)
                    if last_n_tokens == user_info.stop_sequence:
                        user_info.decode_complete = True

                if user_info.decode_complete:
                    # user just finished
                    self.decode_ids[idx][0] = user_info.eos_token_id
                    user_info.stop_decode_timer()
                    user_info.get_user_stats()

        self.cur_pos += 1
        self.prev_pos += 1
        self.timer_stop("token_selection")

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for user_info, user_decode_id in zip(self.users, self.decode_ids):
            if user_info is None:
                continue
            elif user_info.num_tokens_decoded < 1:
                # still prefilling via decode
                continue
            last_token = user_decode_id.item()
            user_info.generated_tokens.append(last_token)
            full_text = self.tokenizer.decode(user_info.generated_tokens)
            return_text = full_text[user_info.num_generated_chars :]
            user_info.num_generated_chars = len(full_text)
            # send special EOS string to frontend
            if (last_token in user_info.stop_tokens) or (user_info.decode_complete):
                return_text = inference_config.end_of_sequence_str
            output_q.put((user_info.user_id, return_text))
            if self.verbose:
                logger.debug(f"user_id:{user_info.user_id}, {return_text}")

    def reset_user_slot(self, user_idx, user):
        self.decode_ids[user_idx, 0] = 0
        self.users[user_idx] = None

    def update_users(self):
        for idx, token_id in enumerate(
            self.decode_ids.reshape(self.batch_size).tolist()
        ):
            if self.users[idx] is None:
                continue

            if token_id in self.users[idx].stop_tokens and self.users[idx].decode_complete:
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
        prefill_time = self.timestamps_stop["prefill"] - self.timestamps_start["prefill"]

        # prefill-via-decode + decode generation tokens
        decode_batch_tokens = (self.decode_counter - self.prev_decode_counter) * self.batch_size
        decode_batch_time = self.timestamps_stop["decode_batch"] - self.timestamps_start["decode_batch"]

        self.prev_decode_counter = self.decode_counter
        
        batch_stats = {
            "batch_counter": self.batch_counter,
            "decode_counter": self.decode_counter,
            "batch_duration": round(batch_duration, 3),
            "batch_users": self.num_users,
            "prefill": {"prefill_batch_size": self.prefill_batch_size, "prefill_batch_tokens": prefill_batch_tokens, "tps": round(prefill_batch_tokens/prefill_time, 3)},
            "decode": {"decode_batch_tokens": decode_batch_tokens, "tps": round(decode_batch_tokens/decode_batch_time, 3)},
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


def batch_top_pk_logits_efficient(
    logits,
    top_ps=[0.9],
    top_ks=[10],
    temperatures=[1.0],
    return_probs=False,
    skip_token=11,
):
    out_tokens = []
    for b_logits, p, k, temperature in zip(logits, top_ps, top_ks, temperatures):
        if p is None:
            # skip None users
            token = torch.tensor([skip_token])
        else:
            # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
            top_k_values, top_k_indices = torch.topk(b_logits, k=k)
            # replace any nans with 0's
            top_k_values = torch.where(
                torch.isnan(top_k_values), torch.zeros_like(top_k_values), top_k_values
            )
            top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
            probs = F.softmax(top_p_values / temperature, dim=-1)
            top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)

        out_tokens.append(token)
    return torch.concat(out_tokens)


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

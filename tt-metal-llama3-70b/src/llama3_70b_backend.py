import os
import time
import traceback
import threading
from multiprocessing import Queue
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

import tt_lib as ttl
import ttnn

from tt_metal_impl.reference.llama import Llama
from transformers.generation.utils import top_k_top_p_filtering
from tt_metal_impl.tt.llama_generation import TtLlamaModelForGeneration
from tt_metal_impl.reference.llama.tokenizer3 import ChatFormat
from tt_metal_impl.tt.llama_common import (
    setup_llama_env,
    check_device_mesh,
    string_similarity_score,
    load_llama_state_dict,
)
from transformers.generation.utils import top_k_top_p_filtering

from model_weights_handler import get_model_weights_and_tt_cache_paths
from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


def intialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = (
            torch.tensor(t, dtype=torch.long, device="cpu").clone().detach()
        )
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask, eos_reached


def prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token

    eos_reached = (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def get_t3k_device_mesh(num_devices_requested):
    assert ttnn.get_num_devices() == 8
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    device_mesh = ttnn.open_device_mesh(
        ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested]
    )
    logger.info(f"multidevice with {device_mesh.get_num_devices()} devices is created")
    return device_mesh


class Args:
    def __init__(
        self,
        # model args
        implementation="tt",
        ckpt_dir=None,
        tokenizer_path=None,
        skip_model_load=False,
        max_batch_size=32,
        num_layers=None,
        max_seq_len=4096,
        # Generation args
        num_tokens=128,
        prompts_file=None,
        output_at_end=True,
        top_p=1,
        top_k=1,
        temperature=1.0,
        chat=False,
        ground_truth=None,
        sample_len=None,
        # TT args
        device_mesh=None,
        n_devices=8,
        cache_path=None,
        decode_only=False,
    ):
        self.implementation = implementation
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path
        self.skip_model_load = skip_model_load
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.prompts_file = prompts_file
        self.output_at_end = output_at_end
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.chat = chat
        self.ground_truth = ground_truth
        self.sample_len = sample_len
        self.device_mesh = device_mesh
        self.n_devices = n_devices
        self.cache_path = cache_path
        self.decode_only = decode_only


def construct_arg(**kwargs):
    return Args(**kwargs)


def build_generator(args):
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        skip_model_load=args.skip_model_load,
        n_layers=1 if args.implementation == "tt" else args.num_layers,
    )

    state_dict = load_llama_state_dict(args.ckpt_dir, n_layers=args.num_layers)

    if args.implementation == "tt":
        generator.model = TtLlamaModelForGeneration(
            configuration=generator.model.params,
            state_dict=state_dict,
            device_mesh=args.device_mesh,
            n_devices=args.n_devices,
            n_layers=args.num_layers,
            cache_path=args.cache_path,
        )
    return generator


class UserInfo:
    def __init__(self, user_id, prompt, position_id, params, tokenizer, formatter=None):
        self.user_id = user_id
        self.prompt = prompt
        self.position_id = position_id
        self.num_tokens_generated = 0
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
        self.forward_counter = 0
        self.prev_forward_counter = 0
        #
        self.device = None
        self.cache_root = Path(cache_root)
        if not self.cache_root.exists():
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # initialization
        self.decode_only = True
        self.init_model()

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

    def teardown(self):
        logger.info("teardown ...")
        self.teardown_tt_metal_device()

    def teardown_tt_metal_device(self):
        logger.info("teardown_tt_metal_device ...")
        device_mesh = self.t3k_device_mesh
        for device in device_mesh.get_devices():
            ttl.device.DumpDeviceProfiler(device)
            ttl.device.DeallocateBuffers(device)

        ttnn.close_device_mesh(device_mesh)
        del device_mesh

    def init_tt_metal_device(self):
        logger.info("init_tt_metal_device ...")
        t3k_device_mesh = get_t3k_device_mesh(
            num_devices_requested=inference_config.n_devices
        )
        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            device.enable_program_cache()

        self.t3k_device_mesh = t3k_device_mesh
        logger.info("init_tt_metal_device finished.")

    def init_model(self):
        # set up variables for model init
        n_devices = inference_config.n_devices
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
        # override for tt-studio
        ckpt_dir = weights_path.as_posix()
        tokenizer_path = tokenizer_path.as_posix()
        cache_path = tt_cache_path
        self.init_tt_metal_device()
        t3k_device_mesh = self.t3k_device_mesh
        check_device_mesh(t3k_device_mesh, model_config)

        for i in t3k_device_mesh.get_device_ids():
            device = t3k_device_mesh.get_device(i)
            device.enable_async(True)

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
            device_mesh=t3k_device_mesh,
            n_devices=n_devices,
            cache_path=cache_path,
            decode_only=self.decode_only,
            ground_truth=False,
        )
        generator = build_generator(args)
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

    def batch_stats(self):
        tokens_generated = self.forward_counter - self.prev_forward_counter
        batch_duration = time.time() - self.batch_start_time
        tps = tokens_generated / batch_duration
        logger.info(
            f"batch_counter:={self.batch_counter}, forward_counter:={self.forward_counter}, tokens_generated:={tokens_generated}, tps:={tps:.4f} tokens/sec (32 users)"
        )
        self.prev_forward_counter = self.forward_counter
        self.batch_start_time = time.time()

    def prepare_inputs(self):
        # empty users get pad id
        input_prompts = [
            user_info.prompt_tokens if user_info else [self.tokenizer.pad_id]
            for user_info in self.users
        ]
        tokens, input_text_mask, eos_reached = intialize_inputs(
            tokenizer=self.tokenizer,
            prompt_tokens=input_prompts,
            bsz=self.batch_size,
            total_len=self.max_seq_len,
        )
        # TODO: when prefill is separate change
        self.cur_pos = 1
        self.prev_pos = 0
        # self.prefill_ids = tokens
        self.input_text_mask = input_text_mask
        self.tokens = tokens
        self.decode_ids = tokens[:, :1]
        self.num_users = len(self.get_users())
        self.batch_counter += 1
        # self.num_input_tokens =end num_input_tokens

    def prefill_via_decode(self):
        # the implementation uses decode
        logger.info("Running prefill_via_decode ...")

    def prefill(self):
        logger.info("Running prefill ...")
        logger.info("Done prefill")

    def decode(self):
        """
        self.cur_pos is the batch level position
        each user has a generation_pos
        """
        self.forward_counter += 1
        self.timer_stop("all_but_decode")
        self.timer_start("decode")
        logits = self.model.forward(
            self.decode_ids, self.prev_pos, decode_only=self.decode_only
        )
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
                    user_info.prefill_complete = True
            else:
                user_info.num_tokens_generated += 1
                if user_decode_id in user_info.stop_tokens:
                    user_info.decode_complete = True
                elif user_info.num_tokens_generated > user_info.max_tokens:
                    user_info.decode_complete = True
                elif user_info.stop_sequence is not None:
                    last_n_tokens = user_info.generated_tokens[
                        -(len(user_info.stop_sequence) - 1) :
                    ]
                    last_n_tokens.append(user_decode_id)
                    if last_n_tokens == user_info.stop_sequence:
                        user_info.decode_complete = True
            if user_info.decode_complete:
                self.decode_ids[idx][0] = user_info.eos_token_id

        self.cur_pos += 1
        self.prev_pos += 1
        self.timer_stop("token_selection")
        self.timer_start("all_but_decode")

    def push_outputs(self, output_q):
        # Sentencepiece tokenizer doesn't handle spaces per token, must decode full text
        # then push new chars to output queue
        for user_info, user_decode_id in zip(self.users, self.decode_ids):
            if user_info is None:
                continue
            elif user_info.num_tokens_generated < 1:
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

    def reset_user_memory(self, user_idx, user):
        self.decode_ids[user_idx, 0] = 0

    def update_users(self):
        for i, token_id in enumerate(
            self.decode_ids.reshape(self.batch_size).tolist()
        ):  # bc input_ids is 1x32
            if self.users[i] is None:
                continue

            if token_id in self.users[i].stop_tokens and self.users[i].decode_complete:
                self.reset_user_memory(i, self.users[i])
                if self.verbose:
                    logger.debug(
                        f"Evicted user_id: {self.users[i].user_id} from index {i} in user list"
                    )
                self.users[i] = None
            elif (
                token_id in self.users[i].stop_tokens
                and not self.users[i].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} had EOS token but decode_complete=False."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None
            elif (
                token_id not in self.users[i].stop_tokens
                and self.users[i].decode_complete
            ):
                logger.error(
                    f"user_id: {self.users[i].user_id} from index {i} did not have EOS token but decode_complete=True."
                )
                self.reset_user_memory(i, self.users[i])
                self.users[i] = None

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
            self.batch_start_time = time.time()
            self.prepare_inputs()
            logger.info("Running inference decode and pushing results ...")
            while not all([user.decode_complete for user in self.get_users()]):
                self.decode()
                self.push_outputs(output_q)
                self.update_users()
                self.send_status(prompt_q, status_q)
            self.batch_stats()
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

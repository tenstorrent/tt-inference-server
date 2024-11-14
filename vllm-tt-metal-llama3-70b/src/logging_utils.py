import os
import json
from datetime import datetime
from vllm.engine.metrics import logger
from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo

# imports for new__init__ function to add logging to the MQLLMEngine
import zmq
import time
import threading
from typing import Optional
from vllm import LLMEngine
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.engine.multiprocessing import (
    IPC_DATA_EXT,
    IPC_HEALTH_EXT,
    IPC_INPUT_EXT,
    IPC_OUTPUT_EXT,
)


# new init function for MQLLMEngine to be used in vllm api server (online inference)
def new__init__(
    self,
    ipc_path: str,
    use_async_sockets: bool,
    *args,
    log_requests: bool = True,
    **kwargs,
) -> None:
    # For MQLLMEngine, we can use cached outputs, since each new request
    # output is immediately pickled and send over the socket, which frees
    # the python object to be reused again.
    kwargs["use_cached_outputs"] = True

    self.engine = LLMEngine(*args, **kwargs)
    num_scheduler_steps = self.engine.scheduler_config.num_scheduler_steps
    batch_size = self.engine.scheduler_config.max_num_seqs
    self.engine.stat_loggers["raw_logging"] = RawStatLogger(
        num_scheduler_steps, batch_size
    )
    self.log_requests = log_requests

    self.use_async_sockets = use_async_sockets
    if self.use_async_sockets:
        self.engine.process_request_outputs_callback = (
            self._async_socket_engine_callback
        )

    self.ctx = zmq.Context()  # type: ignore[attr-defined]

    # Receive input from the client.
    self.input_socket = self.ctx.socket(zmq.constants.PULL)
    self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

    # Send output stream back to client.
    self.output_socket = self.ctx.socket(zmq.constants.PUSH)
    self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

    # Send heartbeats back to client.
    self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
    self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

    # IPC path for the data socket.
    self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

    # Error state.
    self._errored_with: Optional[BaseException] = None

    # Heartbeat thread
    self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
    self._heartbeat_stop_event = threading.Event()
    # The heartbeat needs to be faster than what the client will wait for
    # The VLLM_RPC_TIMEOUT duration is in ms, and we need one in seconds
    self.heartbeat_interval_seconds = VLLM_RPC_TIMEOUT / 5000.0

    self._last_alive_time = time.time()
    # The heartbeats can tolerate a long period of the engine chugging
    # away at a generation request.
    # The VLLM_RPC_TIMEOUT duration is in ms, and we need one in seconds
    self.last_alive_threshold = VLLM_RPC_TIMEOUT * 3.0 / 1000.0


class RawStatLogger(StatLoggerBase):
    def __init__(self, num_scheduler_steps, batch_size) -> None:
        self.time_to_first_token = []
        self.time_per_output_token = []
        self.num_scheduler_steps = num_scheduler_steps
        self.batch_size = batch_size
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.filepath = f"/home/user/tests/statistics_{timestamp}.jsonl"
        self.num_total_grouped_step = (
            0  # number of iterations of size num_scheduler_steps
        )
        self.num_inference = (
            0  # number of times inference is done (ie. how many batches)
        )

    def log(self, stats: Stats, log_to_stdout=True) -> None:
        if len(stats.time_to_first_tokens_iter) > 0:
            self.time_to_first_token.append(
                stats.time_to_first_tokens_iter
            )  # Add all values to the list

            if log_to_stdout:
                for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
                    logger.info(f"User {user_idx}: Time to first token {ttft:.2f} s\n")

        if len(stats.time_per_output_tokens_iter) > 0:
            tpot = [
                time / self.num_scheduler_steps
                for time in stats.time_per_output_tokens_iter
            ]
            self.time_per_output_token.append(tpot)  # Add all values to the list

        self._write_to_json(stats)

    def _write_to_json(self, stats):
        data = {}

        # to record time per output token (decode stage)
        if len(stats.time_per_output_tokens_iter) > 0:
            data["tpot"] = {}
            data["tpot"][f"Total_step_num:{self.num_total_grouped_step}"] = {}
            for user_idx, tpot in enumerate(stats.time_per_output_tokens_iter):
                data["tpot"][f"Total_step_num:{self.num_total_grouped_step}"][
                    f"user_{user_idx}"
                ] = tpot

            self.num_total_grouped_step += 1

        # to record time to first token (prefill stage)
        if len(stats.time_to_first_tokens_iter) > 0:
            # if inference is done online, need to handle case where not all user requests are made at same engine step call
            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as file:
                    lines = file.readlines()
                    # load in last line if time to first token not completed for all users
                    if lines:  # ensure there is data
                        last_line = lines[-1]
                        last_data = json.loads(last_line)
                        if (
                            "ttft" in last_data
                        ):  # if still in prefill stage (incomplete for all users) or only doing prefill and no decode
                            if (
                                len(list(last_data["ttft"].values())[0])
                                < self.batch_size
                            ):  # if incomplete prefill for all users
                                self._append_new_users(data)
                                # find the index of the last user for whicht the first token was computed
                                last_user_processed = len(
                                    list(last_data["ttft"].values())[0]
                                )

                            else:  # if prefill already complete for all users
                                last_user_processed = 0
                                self._append_new_users(data)

                        else:  # if in decode stage
                            last_user_processed = 0
                            self._append_new_users(data)
            else:  # if first forward pass
                last_user_processed = 0
                self._append_new_users(data)

            for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
                data["ttft"][f"Inference_num:{self.num_inference}"][
                    f"user_{user_idx + last_user_processed}"
                ] = ttft

            self.num_inference += 1  # increase number of inference passes

        if data:
            with open(self.filepath, "a") as file:
                json.dump(data, file)
                file.write("\n")  # Ensure each JSON object is on a new line

    def _append_new_users(self, data):
        data["ttft"] = {}
        data["ttft"][f"Inference_num:{self.num_inference}"] = {}

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

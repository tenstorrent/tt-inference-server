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
    self.engine.stat_loggers["raw_logging"] = RawStatLogger(num_scheduler_steps)
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
    def __init__(self, num_scheduler_steps) -> None:
        self.time_to_first_token = []
        self.time_per_output_token = []
        self.num_scheduler_steps = num_scheduler_steps
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

        if len(stats.time_per_output_tokens_iter) > 0:
            data["time per output token"] = {}
            data["time per output token"][
                f"Total step num:{self.num_total_grouped_step}"
            ] = {}
            for user_idx, tpot in enumerate(stats.time_per_output_tokens_iter):
                data["time per output token"][
                    f"Total step num:{self.num_total_grouped_step}"
                ][f"user {user_idx}"] = tpot

            self.num_total_grouped_step += 1

        if len(stats.time_to_first_tokens_iter) > 0:
            # if inference is done online, need to handle case where not all user requests are made at same engine step call
            if os.path.exists(self.filepath):
                with open(self.filepath, "r") as file:
                    lines = file.readlines()
                    # load in last line if time to first token not completed for all users
                    if lines:
                        last_line = lines[-1]
                        last_data = json.loads(last_line)
                        if "time to first token" in last_data:
                            data = last_data
                            # find the index of the last user for whicht the first token was computed
                            last_user_processed = len(
                                data["time to first token"][
                                    f"Inference num:{self.num_inference}"
                                ]
                            )
                        else:
                            last_user_processed = 0
                            data["time to first token"] = {}
                            data["time to first token"][
                                f"Inference num:{self.num_inference}"
                            ] = {}
            else:
                last_user_processed = 0
                data["time to first token"] = {}
                data["time to first token"][f"Inference num:{self.num_inference}"] = {}

            for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
                data["time to first token"][f"Inference num:{self.num_inference}"][
                    f"user {user_idx + last_user_processed}"
                ] = ttft

            if (
                len(data["time to first token"][f"Inference num:{self.num_inference}"])
                == 32
            ):  # if batch size == num users processed
                self.num_inference += 1

        if data:
            with open(self.filepath, "a") as file:
                json.dump(data, file)
                file.write("\n")  # Ensure each JSON object is on a new line

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError
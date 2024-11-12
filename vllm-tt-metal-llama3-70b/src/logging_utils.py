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
        self.filepath = f"/home/user/tests/statistics_{timestamp}.json"

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
        """
        Called in step() by TTWorker every time log() is called to write
        time to first token and time per output token per num scheduler steps.
        """
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = {}  # if empty or something wrong
        else:
            data = {}

        if len(stats.time_per_output_tokens_iter) > 0:
            if "time per output token" in data:
                # get current set of num scheduler steps number
                if len(data["time per output token"].keys()) > 0:
                    num_total_grouped_step = len(data["time per output token"].keys())
                else:
                    return  # Exit the function if there are no inferences
            else:
                num_total_grouped_step = 0
                data["time per output token"] = {}

            data["time per output token"][
                f"Total step num:{num_total_grouped_step}"
            ] = {}
            for user_idx, tpot in enumerate(stats.time_per_output_tokens_iter):
                data["time per output token"][
                    f"Total step num:{num_total_grouped_step}"
                ][f"user {user_idx}"] = tpot

        if len(stats.time_to_first_tokens_iter) > 0:
            if "time to first token" in data:
                # Get the current inference number to use as the key
                if len(data["time to first token"].keys()) > 0:
                    num_inference = len(data["time to first token"].keys())
                else:
                    return  # Exit the function if there are no inferences
            else:
                # Initialize the "time to first token" dictionary if it doesn't exist
                num_inference = 0
                data["time to first token"] = {}

            data["time to first token"][f"Inference num:{num_inference}"] = {}
            for user_idx, ttft in enumerate(stats.time_to_first_tokens_iter):
                data["time to first token"][f"Inference num:{num_inference}"][
                    f"user {user_idx}"
                ] = ttft

        with open(self.filepath, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

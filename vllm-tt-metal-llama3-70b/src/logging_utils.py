import os
import json
from datetime import datetime
from vllm.engine.metrics import logger
from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo


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

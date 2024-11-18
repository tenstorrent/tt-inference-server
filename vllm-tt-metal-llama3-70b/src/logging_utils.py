import os
import datetime
import json

from vllm.engine.metrics_types import StatLoggerBase, Stats, SupportsMetricsInfo
from vllm.engine.metrics import logger
from vllm.engine.llm_engine import LLMEngine


# new init function for LLMEngine to be used in vllm api server (online inference) when init in MQLLMEngine
original_init = LLMEngine.__init__


def init_wrapper(self, *args, **kwargs):
    original_init(self, *args, **kwargs)  # Call the original __init__
    num_scheduler_steps = self.scheduler_config.num_scheduler_steps
    batch_size = self.scheduler_config.max_num_seqs
    self.stat_loggers["raw_logging"] = RawStatLogger(num_scheduler_steps, batch_size)


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

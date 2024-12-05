# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import threading
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from transformers import AutoTokenizer

from prompt_configs import BatchConfig
from prompt_client import PromptClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BatchProcessor:
    def __init__(self, prompt_client: PromptClient, batch_config: BatchConfig):
        self.prompt_client = prompt_client
        self.batch_config = batch_config
        self.responses_lock = threading.Lock()

    def _calculate_batch_sizes(self, num_prompts: int) -> List[int]:
        if self.batch_config.vary_batch_size:
            mean_workers = self.batch_config.batch_size / 2
            std_dev = self.batch_config.batch_size / 4

            batch_sizes = []
            remaining = num_prompts

            while remaining > 0:
                size = int(
                    np.clip(
                        np.random.normal(mean_workers, std_dev),
                        1,
                        self.batch_config.batch_size,
                    )
                )
                if size > remaining:
                    size = remaining
                batch_sizes.append(size)
                remaining -= size

            return batch_sizes

        return [self.batch_config.batch_size] * (
            num_prompts // self.batch_config.batch_size
        )

    def process_batch(
        self,
        prompts: List[str],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
    ) -> List[dict]:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_fpath = (
            Path(self.prompt_client.env_config.cache_root)
            / f"alpaca_eval_responses_{timestamp}.json"
        )

        total_prompts = len(prompts) * self.batch_config.num_full_iterations
        response_counter = 0
        all_responses = []

        with open(json_fpath, "a") as f:
            f.write("[\n")

        if self.batch_config.batch_size == 1:
            all_responses = self._process_single_thread(
                prompts,
                input_seq_lengths,
                tokenizer,
                json_fpath,
                total_prompts,
                response_counter,
            )
        else:
            all_responses = self._process_multi_thread(
                prompts,
                input_seq_lengths,
                tokenizer,
                json_fpath,
                total_prompts,
                response_counter,
            )

        with open(json_fpath, "a") as f:
            f.write("\n]")

        return all_responses

    def _process_single_thread(
        self,
        prompts: List[str],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
        json_fpath: Path,
        total_prompts: int,
        response_counter: int,
    ) -> List[dict]:
        all_responses = []

        for iter_num in range(self.batch_config.num_full_iterations):
            for i, (prompt, isl) in enumerate(zip(prompts, input_seq_lengths)):
                if self.batch_config.inter_batch_delay > 0:
                    time.sleep(self.batch_config.inter_batch_delay)

                response_idx = iter_num * len(prompts) + i
                response_data = self.prompt_client.call_inference(
                    prompt=prompt,
                    response_idx=response_idx,
                    prompt_len=isl,
                    max_tokens=self.batch_config.output_seq_lens[i],
                    stream=self.batch_config.stream,
                    vll_model=self.prompt_client.env_config.vllm_model,
                    tokenizer=tokenizer,
                )

                self._save_response(
                    response_data, all_responses, json_fpath, response_counter
                )
                response_counter += 1
                self._log_progress(response_counter, total_prompts, response_data)

        return all_responses

    def _process_multi_thread(
        self,
        prompts: List[str],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
        json_fpath: Path,
        total_prompts: int,
        response_counter: int,
    ) -> List[dict]:
        all_responses = []

        if self.batch_config.vary_batch_size:
            batch_sizes = self._calculate_batch_sizes(len(prompts))

            for iter_num in range(self.batch_config.num_full_iterations):
                batch_start = 0

                for bsz in batch_sizes:
                    batch_end = min(batch_start + bsz, len(prompts))
                    self._process_batch_chunk(
                        prompts[batch_start:batch_end],
                        input_seq_lengths[batch_start:batch_end],
                        iter_num,
                        bsz,
                        tokenizer,
                        all_responses,
                        json_fpath,
                        total_prompts,
                        response_counter,
                    )
                    batch_start = batch_end
        else:
            with ThreadPoolExecutor(
                max_workers=self.batch_config.batch_size
            ) as executor:
                futures = []

                for iter_num in range(self.batch_config.num_full_iterations):
                    for i, (prompt, isl) in enumerate(zip(prompts, input_seq_lengths)):
                        response_idx = iter_num * len(prompts) + i
                        future = executor.submit(
                            self.prompt_client.call_inference,
                            prompt=prompt,
                            response_idx=response_idx,
                            prompt_len=isl,
                            max_tokens=self.batch_config.output_seq_lens[i],
                            stream=self.batch_config.stream,
                            vll_model=self.prompt_client.env_config.vllm_model,
                            tokenizer=tokenizer,
                        )
                        futures.append(future)

                for future in as_completed(futures):
                    try:
                        response_data = future.result()
                        self._save_response(
                            response_data, all_responses, json_fpath, response_counter
                        )
                        response_counter += 1
                        self._log_progress(
                            response_counter, total_prompts, response_data
                        )
                    except Exception as e:
                        logger.error(f"Error processing response: {e}")

        return all_responses

    def _process_batch_chunk(
        self,
        batch_prompts: List[str],
        batch_input_seq_lengths: List[int],
        iter_num: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
        all_responses: List[dict],
        json_fpath: Path,
        total_prompts: int,
        response_counter: int,
    ):
        if self.batch_config.inter_batch_delay > 0:
            time.sleep(self.batch_config.inter_batch_delay)

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []

            for i, (prompt, isl) in enumerate(
                zip(batch_prompts, batch_input_seq_lengths)
            ):
                response_idx = iter_num * len(batch_prompts) + i
                future = executor.submit(
                    self.prompt_client.call_inference,
                    prompt=prompt,
                    response_idx=response_idx,
                    prompt_len=isl,
                    max_tokens=self.batch_config.output_seq_lens[i],
                    stream=self.batch_config.stream,
                    vll_model=self.prompt_client.env_config.vllm_model,
                    tokenizer=tokenizer,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    response_data = future.result()
                    self._save_response(
                        response_data, all_responses, json_fpath, response_counter
                    )
                    response_counter += 1
                    self._log_progress(response_counter, total_prompts, response_data)
                except Exception as e:
                    logger.error(f"Error processing response: {e}")

    def _save_response(
        self,
        response_data: dict,
        all_responses: List[dict],
        json_fpath: Path,
        response_counter: int,
    ):
        with self.responses_lock:
            all_responses.append(response_data)
            with open(json_fpath, "a") as f:
                if response_counter > 0:
                    f.write(",")
                json.dump(response_data, f, indent=4)

    def _log_progress(
        self, response_counter: int, total_prompts: int, response_data: dict
    ):
        logger.info(
            f"Processed {response_counter}/{total_prompts} responses. "
            f"decode_tps: {response_data['decode_tps']:.2f}, "
            f"total_tps: {response_data['total_tps']:.2f}, "
            f"ttft: {response_data['ttft']:.2f}, "
            f"input_seq_len: {response_data['input_seq_len']}, "
            f"output_seq_len: {response_data['output_seq_len']}"
        )

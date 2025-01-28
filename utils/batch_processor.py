# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import threading
import logging
import json
import time
from pathlib import Path
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from transformers import AutoTokenizer

from utils.prompt_configs import BatchConfig
from utils.prompt_client import PromptClient

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

    def _calculate_max_concurrents(self, num_prompts: int) -> List[int]:
        if self.batch_config.vary_max_concurrent:
            mean_workers = self.batch_config.max_concurrent / 2
            std_dev = self.batch_config.max_concurrent / 4

            max_concurrents = []
            remaining = num_prompts

            while remaining > 0:
                size = int(
                    np.clip(
                        np.random.normal(mean_workers, std_dev),
                        1,
                        self.batch_config.max_concurrent,
                    )
                )
                if size > remaining:
                    size = remaining
                max_concurrents.append(size)
                remaining -= size

            return max_concurrents

        return [self.batch_config.max_concurrent] * (
            num_prompts // self.batch_config.max_concurrent
        )

    def process_batch(
        self,
        prompts: List[str],
        images: List[List[str]],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
        output_path: Union[Path, str] = None,
    ) -> List[dict]:
        total_prompts = len(prompts) * self.batch_config.num_full_iterations
        response_counter = 0
        all_responses = []

        if output_path:
            with open(output_path, "a") as f:
                f.write("[\n")

        if self.batch_config.max_concurrent == 1:
            all_responses = self._process_single_thread(
                prompts,
                images,
                input_seq_lengths,
                tokenizer,
                output_path,
                total_prompts,
                response_counter,
            )
        else:
            all_responses = self._process_multi_thread(
                prompts,
                images,
                input_seq_lengths,
                tokenizer,
                output_path,
                total_prompts,
                response_counter,
            )

        if output_path:
            with open(output_path, "a") as f:
                f.write("\n]")

        return all_responses

    def _process_single_thread(
        self,
        prompts: List[str],
        images: List[List[str]],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
        output_path: Union[Path, str],
        total_prompts: int,
        response_counter: int,
    ) -> List[dict]:
        all_responses = []

        for iter_num in range(self.batch_config.num_full_iterations):
            for i, (prompt, img, isl) in enumerate(
                zip(prompts, images, input_seq_lengths)
            ):
                if self.batch_config.inter_batch_delay > 0:
                    time.sleep(self.batch_config.inter_batch_delay)

                response_idx = iter_num * len(prompts) + i
                response_data = self.prompt_client.call_inference(
                    prompt=prompt,
                    images=img,
                    response_idx=response_idx,
                    prompt_len=isl,
                    max_tokens=self.batch_config.output_seq_lens[i],
                    stream=self.batch_config.stream,
                    vll_model=self.prompt_client.env_config.vllm_model,
                    tokenizer=tokenizer,
                    use_chat_api=self.batch_config.use_chat_api,
                )

                self._save_response(
                    response_data, all_responses, output_path, response_counter
                )
                response_counter += 1
                self._log_progress(response_counter, total_prompts, response_data)

        return all_responses

    def _process_multi_thread(
        self,
        prompts: List[str],
        images: List[List[str]],
        input_seq_lengths: List[int],
        tokenizer: AutoTokenizer,
        output_path: Union[Path, str],
        total_prompts: int,
        response_counter: int,
    ) -> List[dict]:
        all_responses = []

        if self.batch_config.vary_max_concurrent:
            max_concurrents = self._calculate_max_concurrents(len(prompts))

            for iter_num in range(self.batch_config.num_full_iterations):
                batch_start = 0

                for maxcon in max_concurrents:
                    batch_end = min(batch_start + maxcon, len(prompts))
                    self._process_batch_chunk(
                        prompts[batch_start:batch_end],
                        input_seq_lengths[batch_start:batch_end],
                        images[batch_start:batch_end],
                        iter_num,
                        maxcon,
                        tokenizer,
                        all_responses,
                        output_path,
                        total_prompts,
                        response_counter,
                    )
                    batch_start = batch_end
        else:
            with ThreadPoolExecutor(
                max_workers=self.batch_config.max_concurrent
            ) as executor:
                futures = []

                for iter_num in range(self.batch_config.num_full_iterations):
                    for i, (prompt, img, isl) in enumerate(
                        zip(prompts, images, input_seq_lengths)
                    ):
                        response_idx = iter_num * len(prompts) + i
                        future = executor.submit(
                            self.prompt_client.call_inference,
                            prompt=prompt,
                            images=img,
                            response_idx=response_idx,
                            prompt_len=isl,
                            max_tokens=self.batch_config.output_seq_lens[i],
                            stream=self.batch_config.stream,
                            vll_model=self.prompt_client.env_config.vllm_model,
                            tokenizer=tokenizer,
                            use_chat_api=self.batch_config.use_chat_api,
                        )
                        futures.append(future)

                for future in as_completed(futures):
                    try:
                        response_data = future.result()
                        self._save_response(
                            response_data, all_responses, output_path, response_counter
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
        batch_images: List[List[str]],
        batch_input_seq_lengths: List[int],
        iter_num: int,
        max_concurrent: int,
        tokenizer: AutoTokenizer,
        all_responses: List[dict],
        output_path: Union[Path, str],
        total_prompts: int,
        response_counter: int,
    ):
        if self.batch_config.inter_batch_delay > 0:
            time.sleep(self.batch_config.inter_batch_delay)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []

            for i, (prompt, images, isl) in enumerate(
                zip(batch_prompts, batch_images, batch_input_seq_lengths)
            ):
                response_idx = iter_num * len(batch_prompts) + i
                future = executor.submit(
                    self.prompt_client.call_inference,
                    prompt=prompt,
                    images=images,
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
                        response_data, all_responses, output_path, response_counter
                    )
                    response_counter += 1
                    self._log_progress(response_counter, total_prompts, response_data)
                except Exception as e:
                    logger.error(f"Error processing response: {e}")

    def _save_response(
        self,
        response_data: dict,
        all_responses: List[dict],
        output_path: Union[Path, str],
        response_counter: int,
    ):
        with self.responses_lock:
            all_responses.append(response_data)
            if output_path:
                with open(output_path, "a") as f:
                    if response_counter > 0:
                        f.write(",")
                    json.dump(response_data, f, indent=4)

    def _log_progress(
        self, response_counter: int, total_prompts: int, response_data: dict
    ):
        logger.info(
            f"Processed {response_counter}/{total_prompts} responses. "
            f"TPOT: {response_data['tpot_ms']:.4f}, "
            f"TTFT: {response_data['ttft_ms']:.4f}, "
            f"input_seq_len: {response_data['input_seq_len']}, "
            f"output_seq_len: {response_data['output_seq_len']}"
        )

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import json
import time
from typing import List

import requests
import jwt
from transformers import AutoTokenizer

from prompt_generation import generate_prompts
from prompt_configs import PromptConfig, EnvironmentConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptClient:
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config
        self.headers = {"Authorization": f"Bearer {self._get_authorization()}"}
        self.completions_url = self._get_api_completions_url()
        self.health_url = self._get_api_health_url()
        self.server_ready = False

    def _get_authorization(self) -> str:
        if self.env_config.authorization:
            return self.env_config.authorization

        if self.env_config.jwt_secret:
            json_payload = json.loads(
                '{"team_id": "tenstorrent", "token_id":"debug-test"}'
            )
            encoded_jwt = jwt.encode(
                json_payload, self.env_config.jwt_secret, algorithm="HS256"
            )
            return encoded_jwt

        raise ValueError(
            "Neither AUTHORIZATION or JWT_SECRET environment variables are set."
        )

    def _get_api_base_url(self) -> str:
        return f"{self.env_config.deploy_url}:{self.env_config.service_port}/v1"

    def _get_api_completions_url(self) -> str:
        return f"{self._get_api_base_url()}/completions"

    def _get_api_health_url(self) -> str:
        return f"{self.env_config.deploy_url}:{self.env_config.service_port}/health"

    def get_health(self) -> requests.Response:
        return requests.get(self.health_url, headers=self.headers)

    def wait_for_healthy(self, timeout: int = 300, interval: int = 10) -> bool:
        if self.server_ready:
            return True

        start_time = time.time()
        total_time_waited = 0

        while time.time() - start_time < timeout:
            req_time = time.time()
            try:
                response = requests.get(
                    self.health_url, headers=self.headers, timeout=interval
                )
                if response.status_code == 200:
                    startup_time = time.time() - start_time
                    logger.info(
                        f"vLLM service is healthy. startup_time:= {startup_time} seconds"
                    )
                    self.server_ready = True
                    return True
                else:
                    logger.warning(f"Health check failed: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check failed: {e}")

            total_time_waited = time.time() - start_time
            sleep_interval = max(2 - (time.time() - req_time), 0)
            logger.info(
                f"Service not ready after {total_time_waited:.2f} seconds, "
                f"waiting {sleep_interval:.2f} seconds before polling ..."
            )
            time.sleep(sleep_interval)

        logger.error(f"Service did not become healthy within {timeout} seconds")
        return False

    def capture_traces(
        self,
        input_sizes: List[int] = None,
        prompts_per_size: int = 1,
        output_seq_len: int = 1,
    ) -> None:
        logger.info("Capturing input sizes ...")

        # Default input sizes based on get_padded_prefill_len()
        if input_sizes is None:
            input_sizes = [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096]

        # Check service health before starting
        if not self.wait_for_healthy():
            raise RuntimeError("vLLM did not start correctly!")

        for size in input_sizes:
            logger.info(f"Capture input size: {size}")

            # Create prompt config for current size
            prompt_config = PromptConfig(
                input_seq_len=size,
                max_prompt_length=size,
                num_prompts=prompts_per_size,
                distribution="fixed",
                dataset="random",
                tokenizer_model=self.env_config.vllm_model,
                template=None,
                save_path=None,
                print_prompts=False,
            )

            # Generate prompts for current size
            prompts, prompt_lengths = generate_prompts(prompt_config)

            # Process each prompt
            for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lengths)):
                try:
                    logger.info(f"Starting capture for input_seq_len: {prompt_len}")
                    response_data = self.call_inference(
                        prompt=prompt,
                        response_idx=i,
                        prompt_len=prompt_len,
                        max_tokens=output_seq_len,
                        stream=True,
                        vll_model=self.env_config.vllm_model,
                        tokenizer=None,
                        force_max_tokens=True,
                    )
                    logger.info(
                        f"Input size: {size}, "
                        f"input_seq_len: {prompt_len}, "
                        f"TTFT: {response_data['ttft']:.3f}s"
                    )
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")

    def call_inference(
        self,
        prompt: str,
        response_idx: int,
        prompt_len: int,
        max_tokens: int,
        stream: bool,
        vll_model: str,
        tokenizer: AutoTokenizer,
        force_max_tokens: bool = True,
        include_usage: bool = True,
    ) -> dict:
        json_data = {
            "model": vll_model,
            "prompt": prompt,
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": stream,
            "stream_options": {"include_usage": include_usage},
        }

        if force_max_tokens:
            json_data["stop"] = "<|reserved_special_token_249|>"

        req_time = time.perf_counter()
        response = requests.post(
            self.completions_url,
            json=json_data,
            headers=self.headers,
            stream=stream,
            timeout=600,
        )

        return self._process_response(
            response, req_time, response_idx, prompt, prompt_len, max_tokens, stream
        )

    def _process_response(
        self,
        response: requests.Response,
        req_time: float,
        response_idx: int,
        prompt: str,
        prompt_len: int,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        full_text = ""
        num_completion_tokens = 0
        first_token_time = 0
        ttft = 0
        usage_dict = {}

        if stream:
            assert (
                response.headers.get("transfer-encoding") == "chunked"
            ), "Response is not chunked"
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    if num_completion_tokens == 0:
                        first_token_time = time.perf_counter()
                        ttft = first_token_time - req_time

                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if data["choices"]:
                            full_text += data["choices"][0].get("text", "")
                            num_completion_tokens += 1
                        else:
                            usage_dict = data.get("usage", {})
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                        continue
        else:
            data = response.json()
            full_text = data["choices"][0]["text"]
            usage_dict = data["usage"]
            first_token_time = req_time

        decode_time = max(time.perf_counter() - first_token_time, 0.0001)
        total_time = max(time.perf_counter() - req_time, 0.0001)

        # verify the number of input tokens
        isl_diff = usage_dict["prompt_tokens"] - prompt_len
        if isl_diff != 0:
            logger.warning(
                f"response_idx=:{response_idx}, isl_diff(actual - expected) =: {isl_diff}"
            )

        # verify the number of output tokens
        usage_completion_tokens = usage_dict["completion_tokens"]
        if num_completion_tokens > 0:
            osl_diff = usage_completion_tokens - num_completion_tokens
            if osl_diff != 0:
                logger.warning(
                    f"response_idx=:{response_idx}, osl_diff(actual - expected) =: {osl_diff}"
                )
            if (
                max_tokens != usage_completion_tokens
                or max_tokens != num_completion_tokens
            ):
                logger.warning(
                    f"response_idx=:{response_idx}, max_tokens=:{max_tokens}, num_completion_tokens=:{num_completion_tokens}, usage_completion_tokens:={usage_completion_tokens}"
                )

        return {
            "response_idx": response_idx,
            "prompt": prompt,
            "response": full_text,
            "input_seq_len": prompt_len,
            "output_seq_len": num_completion_tokens,
            "decode_tps": (max(num_completion_tokens, 1)) / decode_time,
            "total_tps": (max(num_completion_tokens, 1)) / total_time,
            "ttft": ttft,
        }

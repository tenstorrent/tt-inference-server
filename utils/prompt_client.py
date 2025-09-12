# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import json
import time
from typing import List, Tuple, Optional
from pathlib import Path

import requests
import jwt
from transformers import AutoTokenizer

from utils.prompt_generation import generate_prompts
from utils.prompt_configs import PromptConfig, EnvironmentConfig
from utils.cache_monitor import CacheMonitor, CacheGenerationStatus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptClient:
    def __init__(self, env_config: EnvironmentConfig, model_spec=None, cache_dir: Optional[Path] = None):
        self.env_config = env_config
        authorization = self._get_authorization()
        if authorization:
            self.headers = {"Authorization": f"Bearer {authorization}"}
        else:
            self.headers = {}
        self.completions_url = self._get_api_completions_url()
        self.health_url = self._get_api_health_url()
        self.cache_monitor = CacheMonitor(model_spec=model_spec, cache_dir=cache_dir)
        self.server_ready = False

    def _get_authorization(self) -> Optional[str]:
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

        logger.warning(
            "Neither AUTHORIZATION nor JWT_SECRET environment variables are set. "
            "Proceeding without authorization."
        )
        return None

    def _get_api_base_url(self) -> str:
        return f"{self.env_config.deploy_url}:{self.env_config.service_port}/v1"

    def _get_api_completions_url(self) -> str:
        return f"{self._get_api_base_url()}/completions"

    def _get_api_health_url(self) -> str:
        return f"{self.env_config.deploy_url}:{self.env_config.service_port}/health"

    def get_health(self) -> requests.Response:
        return requests.get(self.health_url, headers=self.headers)
    
    def wait_for_healthy(self, timeout: float = 1200.0, interval: int = 10, 
                        cache_generation_timeout_multiplier: float = 3.0) -> bool:
        """
        Wait for the vLLM service to become healthy with intelligent cache generation detection.
        
        Args:
            timeout: Base timeout in seconds
            interval: Health check interval in seconds  
            cache_generation_timeout_multiplier: Multiplier for timeout when cache generation is detected
            
        Returns:
            bool: True if service becomes healthy, False if timeout exceeded
        """
        timeout = float(timeout)
        if self.server_ready:
            return True

        start_time = time.time()
        original_timeout = timeout
        cache_generation_detected = False
        last_cache_status_log = 0
        cache_status_log_interval = 60  # Log cache status every 60 seconds

        logger.info(f"Waiting for vLLM service to become healthy (base timeout: {timeout}s)")

        while time.time() - start_time < timeout:
            req_time = time.time()
            
            # Check cache generation status
            cache_status = self.cache_monitor.get_cache_generation_status()
            current_time = time.time()
            
            # Log cache status periodically
            if current_time - last_cache_status_log > cache_status_log_interval:
                if cache_status.is_generating:
                    logger.info("🔄 Cache generation in progress - this may take 40-60 minutes for new models")
                    if not cache_generation_detected:
                        # First time detecting cache generation - extend timeout
                        extended_timeout = 90 * 60.0
                        timeout = extended_timeout
                        cache_generation_detected = True
                        logger.info(f"⏰ using extended timeout:={timeout}s due to cache generation")
                else:
                    logger.info(f"📁 No active cache generation detected, using standard timeout:={timeout}s")
                    timeout = original_timeout
                last_cache_status_log = current_time
            
            # Try health check
            try:
                response = requests.get(
                    self.health_url, headers=self.headers, timeout=interval
                )
                if response.status_code == 200:
                    startup_time = time.time() - start_time
                    logger.info(
                        f"✅ vLLM service is healthy. startup_time: {startup_time:.1f} seconds"
                    )
                    
                    # Mark cache as completed if it was generating
                    if cache_status.is_generating:
                        self.cache_monitor.mark_cache_completed()
                        logger.info("🎯 Marked cache generation as completed")
                    
                    self.server_ready = True
                    return True
                else:
                    logger.debug(f"Health check did not return 200: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed: {e}")

            total_time_waited = time.time() - start_time
            sleep_interval = max(interval - (time.time() - req_time), 1)
            
            # Provide different messaging based on cache status
            if cache_status.is_generating:
                logger.info(
                    f"🔄 Cache generation in progress. Waited {total_time_waited:.1f}s, "
                    f"next check in {sleep_interval:.1f}s (timeout: {timeout}s)"
                )
            else:
                logger.info(
                    f"⏳ Service not ready after {total_time_waited:.1f}s, "
                    f"waiting {sleep_interval:.1f}s before polling (timeout: {timeout}s)"
                )
            
            time.sleep(sleep_interval)

        # Final status check
        final_cache_status = self.cache_monitor.get_cache_generation_status()
        if final_cache_status.is_generating:
            logger.error(
                f"⛔ Service did not become healthy within {timeout}s. "
                f"Cache generation appears to still be in progress. "
                f"Consider increasing the timeout or checking the docker logs."
            )
        else:
            logger.error(f"⛔ Service did not become healthy within {timeout}s")
        
        return False

    def capture_traces(
        self,
        context_lens: List[Tuple[int, int]] = None,
        image_resolutions: List[Tuple[int, int]] = None,
        timeout: float = 1200.0,
    ) -> None:
        """Capture traces for text and/or image inputs at different sizes.

        Args:
            context_lens: List of (input_seq_len, output_seq_len) tuples for text lengths
            image_resolutions: List of (width, height) tuples for image resolutions
            timeout: startup timeout waiting for server, seconds.
        """
        logger.info("Capturing traces for input configurations...")

        # Default input sizes if none provided
        if context_lens is None:
            default_context_lens = {
                (32, 4),
                (64, 4),
                (128, 4),
                (256, 4),
                (512, 4),
                (1024, 4),
                (2048, 4),
                (3072, 4),
                (4096, 4),
                (8192, 4),
                (16384, 4),
            }
            # ascending order of input sequence length
            context_lens = sorted(default_context_lens)

        # Check service health before starting
        if not self.wait_for_healthy(timeout=timeout):
            raise RuntimeError("vLLM did not start correctly!")

        # Import image generation only if needed
        if image_resolutions:
            from utils.prompt_generation import generate_random_images

        # Process each text length configuration
        for isl, osl in context_lens:
            logger.info(
                f"Capturing traces for input_seq_len={isl}, output_seq_len={osl}"
            )

            # Create prompt config
            prompt_config = PromptConfig(
                input_seq_len=isl,
                max_prompt_length=isl,
                num_prompts=1,
                distribution="fixed",
                dataset="random",
                tokenizer_model=self.env_config.vllm_model,
                template=None,
                save_path=None,
                print_prompts=False,
                use_chat_api=bool(image_resolutions),  # Use chat API if we have images
            )

            # Generate prompts for current size
            prompts, prompt_lengths = generate_prompts(prompt_config)

            # If no image resolutions specified, do text-only traces
            if not image_resolutions:
                for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lengths)):
                    try:
                        logger.info(
                            f"Starting text trace capture: "
                            f"input_seq_len={prompt_len}, output_seq_len={osl}"
                        )
                        response_data = self.call_inference(
                            prompt=prompt,
                            images=[],
                            response_idx=i,
                            prompt_len=prompt_len,
                            max_tokens=osl,
                            stream=True,
                            vllm_model=self.env_config.vllm_model,
                            tokenizer=None,
                            force_max_tokens=True,
                            use_chat_api=False,
                        )
                        logger.info(
                            f"Text trace completed: "
                            f"tokens_generated={response_data['output_seq_len']}, "
                            f"TTFT={response_data['ttft_ms']:.3f}ms, "
                            f"TPOT={response_data['tpot_ms']:.3f}ms\n"
                        )
                    except Exception as e:
                        logger.error(f"Error processing text prompt: {e}")
                        continue
            else:
                # Process each image resolution with the current text length
                for width, height in image_resolutions:
                    for i, (prompt, prompt_len) in enumerate(
                        zip(prompts, prompt_lengths)
                    ):
                        try:
                            # Generate random image at current resolution
                            image_data = generate_random_images(
                                width=width,
                                height=height,
                                base64_encoded=True,
                            )

                            logger.info(
                                f"Starting image + text trace capture: "
                                f"input_seq_len={prompt_len}, output_seq_len={osl}, "
                                f"image_size={width}x{height}"
                            )

                            response_data = self.call_inference(
                                prompt=prompt,
                                images=[image_data],
                                response_idx=i,
                                prompt_len=prompt_len,
                                max_tokens=osl,
                                stream=True,
                                vllm_model=self.env_config.vllm_model,
                                tokenizer=None,
                                force_max_tokens=True,
                                use_chat_api=True,
                            )

                            logger.info(
                                f"Image + Text trace completed: "
                                f"tokens_generated={response_data['output_seq_len']}, "
                                f"TTFT={response_data['ttft_ms']:.3f}ms, "
                                f"TPOT={response_data['tpot_ms']:.3f}ms\n"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error processing prompt with image {width}x{height}: {e}"
                            )
                            continue

    def call_inference(
        self,
        prompt: str,
        images: List[str],
        response_idx: int,
        prompt_len: int,
        max_tokens: int,
        stream: bool,
        vllm_model: str,
        tokenizer: AutoTokenizer,
        force_max_tokens: bool = True,
        include_usage: bool = True,
        use_chat_api: bool = False,
    ) -> dict:
        """Unified inference call handling both regular and chat APIs, with optional image support."""

        # Prepare the request payload based on API type
        if use_chat_api:
            content = [
                {"type": "text", "text": prompt},
            ]
            for image_data in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

            json_data = {
                "model": vllm_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            completions_url = f"{self._get_api_base_url()}/chat/completions"
        else:
            assert (
                len(images) == 0
            ), "legacy API does not support images, use --use_chat_api option."
            json_data = {
                "model": vllm_model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": stream,
                "stream_options": {"include_usage": include_usage},
            }

            completions_url = self.completions_url

        if force_max_tokens:
            json_data["min_tokens"] = max_tokens
            json_data["ignore_eos"] = True

        logger.info(f"calling: {completions_url}, response_idx={response_idx}")
        logger.info(f"model: {vllm_model}")
        req_time = time.perf_counter()
        response = requests.post(
            completions_url,
            json=json_data,
            headers=self.headers,
            stream=stream,
            timeout=1800,
        )

        return self._process_response(
            response,
            req_time,
            response_idx,
            prompt,
            prompt_len,
            max_tokens,
            stream,
            use_chat_api,
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
        use_chat_api: bool = False,
    ) -> dict:
        """Unified response processing for both regular and chat APIs."""
        full_text = ""
        num_completion_tokens = 0
        first_token_time = 0
        ttft = 0
        token_timestamps = []

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    current_time = time.perf_counter()
                    if num_completion_tokens == 0:
                        first_token_time = current_time
                        ttft = first_token_time - req_time

                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if data.get("choices"):
                            if use_chat_api:
                                content = (
                                    data["choices"][0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                            else:
                                content = data["choices"][0].get("text", "")

                            full_text += content
                            if content:  # Only count non-empty content
                                token_timestamps.append(current_time)
                                num_completion_tokens += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                        continue
        else:
            data = response.json()
            if use_chat_api:
                full_text = data["choices"][0]["message"]["content"]
            else:
                full_text = data["choices"][0]["text"]
            first_token_time = req_time

        # Rest of processing remains the same
        latency = time.perf_counter() - req_time
        inter_token_latencies = []
        if len(token_timestamps) > 1:
            inter_token_latencies = [
                (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(token_timestamps))
            ]

        gen_time = max(time.perf_counter() - first_token_time, 0.0001)
        time_per_output_token = gen_time / max(num_completion_tokens - 1, 1)

        return {
            "response_idx": response_idx,
            "prompt": prompt,
            "response": full_text,
            "input_seq_len": prompt_len,
            "output_seq_len": num_completion_tokens,
            "itl_ms": inter_token_latencies,
            "tpot_ms": time_per_output_token * 1000.0,
            "ttft_ms": ttft * 1000.0,
            "latency": latency,
        }

    def _process_chat_response(
        self,
        response: requests.Response,
        req_time: float,
        response_idx: int,
        prompt: str,
        prompt_len: int,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        """Process responses from the chat completions API."""
        full_text = ""
        num_completion_tokens = 0
        first_token_time = 0
        ttft = 0
        token_timestamps = []

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    current_time = time.perf_counter()
                    if num_completion_tokens == 0:
                        first_token_time = current_time
                        ttft = first_token_time - req_time

                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if data.get("choices"):
                            content = (
                                data["choices"][0].get("delta", {}).get("content", "")
                            )
                            full_text += content
                            if content:  # Only count non-empty content
                                token_timestamps.append(current_time)
                                num_completion_tokens += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                        continue
        else:
            data = response.json()
            full_text = data["choices"][0]["message"]["content"]
            first_token_time = req_time

        # Rest of processing remains the same as in _process_response
        latency = time.perf_counter() - req_time
        inter_token_latencies = []
        if len(token_timestamps) > 1:
            inter_token_latencies = [
                (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
                for i in range(1, len(token_timestamps))
            ]

        gen_time = max(time.perf_counter() - first_token_time, 0.0001)
        time_per_output_token = gen_time / max(num_completion_tokens - 1, 1)

        return {
            "response_idx": response_idx,
            "prompt": prompt,
            "response": full_text,
            "input_seq_len": prompt_len,
            "output_seq_len": num_completion_tokens,
            "itl_ms": inter_token_latencies,
            "tpot_ms": time_per_output_token * 1000.0,
            "ttft_ms": ttft * 1000.0,
            "latency": latency,
        }

    def call_chat_inference(
        self,
        prompt: str,
        response_idx: int,
        prompt_len: int,
        max_tokens: int,
        stream: bool,
        vllm_model: str,
        tokenizer: AutoTokenizer,
        image_data: Optional[str] = None,
        force_max_tokens: bool = True,
        include_usage: bool = True,
    ) -> dict:
        """Call inference using the chat completions API format."""
        messages = []

        if image_data:
            # Create message with both text and image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            )
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})

        json_data = {
            "model": vllm_model,
            "messages": messages,
            "temperature": 1,
            "top_k": 20,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if force_max_tokens:
            json_data["min_tokens"] = max_tokens
            json_data["ignore_eos"] = True

        chat_url = f"{self._get_api_base_url()}/chat/completions"
        req_time = time.perf_counter()
        response = requests.post(
            chat_url,
            json=json_data,
            headers=self.headers,
            stream=stream,
            timeout=1800,
        )

        return self._process_chat_response(
            response, req_time, response_idx, prompt, prompt_len, max_tokens, stream
        )

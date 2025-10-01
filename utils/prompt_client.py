# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import json
import time
from typing import List, Tuple, Optional

import requests
import jwt
from transformers import AutoTokenizer

from utils.prompt_generation import generate_prompts
from utils.prompt_configs import PromptConfig, EnvironmentConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default configurations for trace capture
DEFAULT_IMAGE_RESOLUTIONS = [
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
]

# Hardcoded padded sequence lengths for trace capture
# Based on TT hardware padding requirements (powers of 2 and multiples of 1024)
PADDED_SEQ_LENS = [
    1,
    128,
    256,
    512,
] + [isl - 128 for isl in range(1024, 131072, 1024)]


def get_trace_context_lens(
    max_context: int,
    output_len: int = 4,
) -> List[Tuple[int, int]]:
    """Get trace context lengths filtered by model's max context length.
    
    Args:
        max_context: Maximum context length supported by the model
        output_len: Fixed output sequence length for trace capture
        
    Returns:
        List of (input_seq_len, output_seq_len) tuples
    """
    return [(seq_len, output_len) for seq_len in PADDED_SEQ_LENS if seq_len <= max_context]


class PromptClient:
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config
        authorization = self._get_authorization()
        if authorization:
            self.headers = {"Authorization": f"Bearer {authorization}"}
        else:
            self.headers = {}
        self.completions_url = self._get_api_completions_url()
        self.health_url = self._get_api_health_url()
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

    def wait_for_healthy(self, timeout: float = 1200.0, interval: int = 10) -> bool:
        timeout = float(timeout)
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
            sleep_interval = max(10 - (time.time() - req_time), 0)
            logger.info(
                f"Service not ready after {total_time_waited:.2f} seconds, "
                f"waiting {sleep_interval:.2f} seconds before polling ..."
            )
            time.sleep(sleep_interval)

        logger.error(f"Service did not become healthy within {timeout} seconds")
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


def run_background_trace_capture(
    hf_model_repo: str,
    service_port: int,
    jwt_secret: str = None,
    supported_modalities: List[str] = None,
    max_context: int = None,
    context_lens: List[Tuple[int, int]] = None,
    image_resolutions: List[Tuple[int, int]] = None,
    health_timeout: float = 1800.0,
    trace_timeout: float = 1200.0,
):
    """Run trace capture in a separate process after server becomes healthy.

    This function is designed to be called as a multiprocessing.Process target.
    It waits for the vLLM server to be healthy, then captures traces for the
    specified or calculated context lengths and image resolutions.

    Args:
        hf_model_repo: HuggingFace model repository ID
        service_port: Port where the vLLM server is running
        jwt_secret: JWT secret for authentication (optional)
        supported_modalities: List of supported modalities (e.g., ["text", "image"])
        max_context: Maximum context length supported by the model (for calculating traces)
        context_lens: List of (input_seq_len, output_seq_len) tuples (overrides calculation)
        image_resolutions: List of (width, height) tuples for image inputs
        health_timeout: Timeout in seconds to wait for server to become healthy
        trace_timeout: Timeout in seconds for trace capture operations
    """
    try:
        logger.info("Starting background trace capture process...")

        # Use defaults if not provided
        if supported_modalities is None:
            supported_modalities = ["text"]

        # Calculate or use provided context lengths
        if context_lens is None:
            if max_context is None:
                max_context = 131072  # Default max context
            context_lens = get_trace_context_lens(max_context=max_context, output_len=4)
            logger.info(
                f"Using {len(context_lens)} trace context lengths "
                f"(max_context={max_context})"
            )
            logger.debug(f"Context lengths: {context_lens}")
        else:
            logger.info(f"Using provided context lengths: {context_lens}")

        # Configure environment
        env_config = EnvironmentConfig()
        env_config.jwt_secret = jwt_secret if jwt_secret else None
        env_config.service_port = service_port
        env_config.vllm_model = hf_model_repo

        # Create prompt client
        prompt_client = PromptClient(env_config)

        # Wait for server to be healthy
        logger.info(
            f"Waiting for vLLM server to become healthy (timeout: {health_timeout}s)..."
        )
        if not prompt_client.wait_for_healthy(timeout=health_timeout):
            logger.error(
                "⛔️ vLLM server did not become healthy. Skipping trace capture."
            )
            return

        # Capture traces based on supported modalities
        if "image" in supported_modalities:
            if image_resolutions is None:
                image_resolutions = DEFAULT_IMAGE_RESOLUTIONS
            logger.info(
                f"Capturing traces with image support: "
                f"{len(context_lens)} context lengths, "
                f"{len(image_resolutions)} image resolutions"
            )
            prompt_client.capture_traces(
                context_lens=context_lens,
                image_resolutions=image_resolutions,
                timeout=trace_timeout,
            )
        else:
            logger.info(f"Capturing traces: {len(context_lens)} context lengths")
            prompt_client.capture_traces(
                context_lens=context_lens, timeout=trace_timeout
            )

        logger.info("✅ Background trace capture completed successfully")

    except Exception as e:
        logger.error(f"⛔️ Error during background trace capture: {e}", exc_info=True)

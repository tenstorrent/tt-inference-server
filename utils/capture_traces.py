# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import time
import logging
import requests
import argparse
from utils.prompt_generation import generate_prompts
from utils.prompt_client_cli import (
    call_inference_api,
    get_api_base_url,
    get_authorization,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_api_health_url():
    DEPLOY_URL = os.getenv("DEPLOY_URL", "http://127.0.0.1")
    health_url = f"{DEPLOY_URL}:{os.getenv('SERVICE_PORT', '8000')}/health"
    return health_url


def check_health(base_url: str, timeout: int = 300, interval: int = 10) -> bool:
    """
    Check the health endpoint until the service is ready.
    """
    health_url = get_api_health_url()
    start_time = time.time()
    headers = {"Authorization": f"Bearer {get_authorization()}"}

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, headers=headers)
            if response.status_code == 200:
                logger.info("vLLM service is healthy and ready")
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed: {e}")

        logger.info(f"Service not ready, waiting {interval} seconds...")
        time.sleep(interval)

    logger.error(f"Service did not become healthy within {timeout} seconds")
    return False


def capture_input_sizes():
    """
    Capture different input size graphs with the TT model on vLLM.
    get_padded_prefill_len() defines the different input sizes for prefill:
    https://github.com/tenstorrent/tt-metal/blob/main/models/demos/t3000/llama2_70b/tt/llama_generation.py#L341
    """
    input_sizes = [sz - 8 for sz in [32, 64, 128, 256, 512, 1024, 2048, 3072, 4096]]
    prompts_per_size = 1
    output_seq_len = 1

    base_url = get_api_base_url()
    if not check_health(base_url):
        raise RuntimeError("vLLM did not start correctly!")

    api_url = f"{base_url}/completions"
    headers = {"Authorization": f"Bearer {get_authorization()}"}
    vllm_model = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

    for size in input_sizes:
        logger.info(f"Capture input size: {size}")

        args = argparse.Namespace(
            tokenizer_model=vllm_model,
            dataset="random",
            max_prompt_length=size,
            input_seq_len=size,
            distribution="fixed",
            template=None,
            save_path=None,
            print_prompts=False,
            num_prompts=prompts_per_size,
        )

        prompts, prompt_lengths = generate_prompts(args)

        for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lengths)):
            try:
                response_data = call_inference_api(
                    prompt=prompt,
                    response_idx=i,
                    prompt_len=prompt_len,
                    stream=True,
                    headers=headers,
                    api_url=api_url,
                    max_tokens=output_seq_len,
                    vll_model=vllm_model,
                    tokenizer=None,
                )

                logger.info(
                    f"Input size: {size}, input_seq_len: {prompt_len}, TTFT: {response_data['ttft']:.3f}s"
                )

            except Exception as e:
                logger.error(f"Error processing prompt: {e}")


def main():
    try:
        capture_input_sizes()
    except Exception as e:
        logger.error(f"Capturing input sizes failed: {e}")
        raise


if __name__ == "__main__":
    main()

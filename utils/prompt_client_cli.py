# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import getpass
import threading
import logging
import json
import argparse
import time
from datetime import datetime
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import jwt
import numpy as np
from transformers import AutoTokenizer

from utils.prompt_generation import add_prompt_gen_args, generate_prompts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set numpy seed for reproducibility
np.random.seed(42)


def get_authorization():
    authorization = os.getenv("AUTHORIZATION", None)
    if authorization is None:
        jwt_secret = os.getenv("JWT_SECRET", None)
        if jwt_secret is None:
            raise ValueError(
                "Neither AUTHORIZATION or JWT_SECRET environment variables are set."
            )
        json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        authorization = f"{encoded_jwt}"
    return authorization


def get_api_base_url():
    DEPLOY_URL = os.getenv("DEPLOY_URL", "http://127.0.0.1")
    base_url = f"{DEPLOY_URL}:{os.getenv('SERVICE_PORT', '8000')}/v1"
    return base_url


def get_api_url():
    base_url = get_api_base_url()
    api_url = f"{base_url}/completions"
    return api_url


# Thread-safe data collection
responses_lock = threading.Lock()
responses = []


def call_inference_api(
    prompt,
    response_idx,
    prompt_len,
    stream,
    headers,
    api_url,
    max_tokens,
    vll_model,
    tokenizer,
):
    # set API prompt and optional parameters
    json_data = {
        "model": vll_model,
        "prompt": prompt,
        "temperature": 1,
        "top_k": 20,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    req_time = time.time()
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        api_url, json=json_data, headers=headers, stream=stream, timeout=600
    )
    # Handle chunked response
    full_text = ""
    num_completion_tokens = 0
    first_token_time = 0
    ttft = 0
    if stream:
        if response.headers.get("transfer-encoding") == "chunked":
            for line in response.iter_lines(decode_unicode=True):
                # Process each line of data as it's received
                if line:
                    # Remove the 'data: ' prefix
                    if line.startswith("data: "):
                        if num_completion_tokens == 0:
                            first_token_time = time.time()
                            ttft = first_token_time - req_time
                        num_completion_tokens += 1
                        data_str = line[len("data: ") :].strip()
                        if data_str == "[DONE]":
                            num_completion_tokens -= 1
                            break
                        try:
                            # Parse the JSON data
                            data = json.loads(data_str)
                            # Extract text from the 'choices' field
                            content = data["choices"][0].get("text", "")
                            full_text += content
                        except json.JSONDecodeError as e:
                            print(f"Failed to decode JSON: {e}")
                            continue
        else:
            # If not chunked, you can access the entire response body at once
            data = response.json()["usage"]
            raise ValueError("Response is not chunked")

    else:
        data = response.json()
        full_text = data["choices"][0]["text"]
        num_completion_tokens = data["usage"]["completion_tokens"]
        # conservatively set the first token time to the request time
        first_token_time = req_time
        logger.info(f"usage: {data['usage']}")
        # TODO: verify the number of tokens
        # num_completion_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

    num_completion_tokens = max(num_completion_tokens, 2)
    throughput_time = max(time.time() - first_token_time, 0.0001)
    response_data = {
        "response_idx": response_idx,
        "prompt": prompt,
        "response": full_text,
        "prompt_length": prompt_len,
        "num_completion_tokens": num_completion_tokens,
        "tps": (num_completion_tokens - 1) / throughput_time,
        "ttft": ttft,
    }
    with responses_lock:
        responses.append(response_data)
    return response_data


def check_json_fpath(json_fpath):
    directory = os.path.dirname(json_fpath)
    user = getpass.getuser()
    if os.access(directory, os.W_OK):
        try:
            with open(json_fpath, "w") as f:
                f.write("")  # Attempt to write an empty string to the file
            logger.info(f"The file '{json_fpath}' can be created and is writable.")
            return True, ""
        except IOError as err:
            err_msg = f"Cannot write to the file '{json_fpath}'. Reason: {err}"
    else:
        err_msg = (
            f"User:={user} cannot write to file:={json_fpath} in directory:={directory}"
        )
    logger.error(err_msg)
    return False, err_msg


def handle_delay(delay):
    if delay > 0:
        logger.info(f"Sleeping for {delay} seconds...")
        time.sleep(delay)


def calculate_batch_sizes(num_prompts, max_batch_size, vary_batch_size):
    """Calculate normally distributed batch sizes that sum to total_items"""
    if vary_batch_size:
        mean_workers = max_batch_size / 2
        std_dev = max_batch_size / 4

        batch_sizes = []
        remaining = num_prompts

        while remaining > 0:
            size = int(
                np.clip(np.random.normal(mean_workers, std_dev), 1, max_batch_size)
            )
            if size > remaining:
                size = remaining
            batch_sizes.append(size)
            remaining -= size

    else:
        batch_sizes = [max_batch_size] * (num_prompts // max_batch_size)

    return batch_sizes


def test_api_call_threaded_full_queue(
    prompts,
    prompt_lengths,
    batch_size,
    num_full_iterations,
    vary_batch_size,
    inter_batch_delay,
    call_func,
    call_func_kwargs,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_root = Path(os.getenv("CACHE_ROOT", "."))
    json_fpath = cache_root / f"alpaca_eval_responses_{timestamp}.json"
    logger.info(f"Will write output to: {json_fpath}")
    can_write, err_msg = check_json_fpath(json_fpath)
    if not can_write:
        err_msg += (
            f"\nNote: CACHE_ROOT:={cache_root}, consider setting in this shell to $PWD"
        )
    assert can_write, err_msg
    with open(json_fpath, "a") as f:
        f.write("[\n")

    total_prompts = len(prompts) * num_full_iterations
    response_counter = 0
    logger.info(
        f"Running {total_prompts} prompts in full queue with batch size {batch_size}."
    )
    num_prompts = len(prompts)
    if batch_size == 1:
        logger.info("Running with single thread")
        for iter_num in range(num_full_iterations):
            for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lengths)):
                handle_delay(inter_batch_delay)
                response_idx = iter_num * num_prompts + i
                response_data = call_func(
                    prompt=prompt,
                    response_idx=response_idx,
                    prompt_len=prompt_len,
                    **call_func_kwargs,
                )
                # Write the response data to the JSONL file
                with responses_lock:
                    with open(json_fpath, "a") as f:
                        if response_counter > 0:
                            f.write(",")
                        json.dump(response_data, f, indent=4)
                response_counter += 1
                logger.info(
                    f"Processed {response_counter}/{total_prompts} responses. Avg. TPS: {response_data['tps']:.2f}, TTFT: {response_data['ttft']:.2f}, Completion Tokens: {response_data['num_completion_tokens']}, Prompt Length: {response_data['prompt_length']}"
                )
    elif batch_size > 1 and vary_batch_size:
        logger.info(
            f"Running with ThreadPoolExecutor: batch_size={batch_size}, vary_batch_size={vary_batch_size}"
        )
        batch_sizes = calculate_batch_sizes(
            num_prompts=num_prompts,
            max_batch_size=batch_size,
            vary_batch_size=True,
        )

        # Process prompts in batches with varying sizes
        for iter_num in range(num_full_iterations):
            batch_start = 0

            for bsz in batch_sizes:
                batch_end = min(batch_start + bsz, num_prompts)
                batch_prompts = prompts[batch_start:batch_end]
                batch_prompt_lengths = prompt_lengths[batch_start:batch_end]
                handle_delay(inter_batch_delay)
                # Submit all prompts in the current batch
                logger.info(f"Sending batch requests: {bsz}")
                with ThreadPoolExecutor(max_workers=bsz) as executor:
                    futures = []

                    for i, (prompt, prompt_len) in enumerate(
                        zip(batch_prompts, batch_prompt_lengths)
                    ):
                        response_idx = iter_num * num_prompts + i
                        future = executor.submit(
                            call_func,
                            prompt=prompt,
                            response_idx=response_idx,
                            prompt_len=prompt_len,
                            **call_func_kwargs,
                        )
                        futures.append(future)
                    # Wait for all futures in this batch to complete
                    for future in as_completed(futures):
                        try:
                            response_data = future.result()
                            with responses_lock:
                                with open(json_fpath, "a") as f:
                                    if response_counter > 0:
                                        f.write(",")
                                    json.dump(response_data, f, indent=4)
                            response_counter += 1
                            logger.info(
                                f"Processed {response_counter}/{total_prompts} responses. Avg. TPS: {response_data['tps']:.2f}, TTFT: {response_data['ttft']:.2f}, Completion Tokens: {response_data['num_completion_tokens']}, Prompt Length: {response_data['prompt_length']}"
                            )
                        except Exception as e:
                            logger.error(f"Error processing response: {e}")
    elif batch_size > 1 and not vary_batch_size:
        logger.info(
            f"Running with ThreadPoolExecutor: batch_size={batch_size}, vary_batch_size={vary_batch_size}"
        )
        # Process all prompts concurrently up to batch_size limit
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []

            # Submit all prompts across all iterations
            for iter_num in range(num_full_iterations):
                for i, (prompt, prompt_len) in enumerate(zip(prompts, prompt_lengths)):
                    response_idx = iter_num * num_prompts + i
                    future = executor.submit(
                        call_func,
                        prompt=prompt,
                        response_idx=response_idx,
                        prompt_len=prompt_len,
                        **call_func_kwargs,
                    )
                    futures.append(future)

            # Process completed futures as they finish
            for future in as_completed(futures):
                try:
                    response_data = future.result()
                    with responses_lock:
                        with open(json_fpath, "a") as f:
                            if response_counter > 0:
                                f.write(",")
                            json.dump(response_data, f, indent=4)
                    response_counter += 1
                    logger.info(
                        f"Processed {response_counter}/{total_prompts} responses. Avg. TPS: {response_data['tps']:.2f}, TTFT: {response_data['ttft']:.2f}, Completion Tokens: {response_data['num_completion_tokens']}, Prompt Length: {response_data['prompt_length']}"
                    )
                except Exception as e:
                    logger.error(f"Error processing response: {e}")

    logger.info(f"Finished all requests, total responses: {response_counter}")
    with open(json_fpath, "a") as f:
        f.write("\n]")


def main():
    parser = argparse.ArgumentParser(description="Run Alpaca Evaluation Inference.")
    parser = add_client_args(parser)
    parser = add_prompt_gen_args(parser)
    args = parser.parse_args()

    # generate prompts
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    prompts, prompt_lengths = generate_prompts(args)

    headers = {"Authorization": f"Bearer {get_authorization()}"}
    api_url = get_api_url()
    logging.info(f"API_URL: {api_url}")
    test_api_call_threaded_full_queue(
        prompts=prompts,
        prompt_lengths=prompt_lengths,
        batch_size=args.batch_size,
        num_full_iterations=args.num_full_iterations,
        vary_batch_size=args.vary_batch_size,
        inter_batch_delay=args.inter_batch_delay,
        call_func=call_inference_api,
        call_func_kwargs={
            "stream": not args.no_stream,
            "headers": headers,
            "api_url": api_url,
            "max_tokens": args.output_seq_len,
            "vll_model": args.vllm_model,
            "tokenizer": tokenizer,
        },
    )


def add_client_args(parser):
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (default: streaming enabled)",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"),
        help="Model name vLLM API server is using.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--num_full_iterations",
        type=int,
        default=1,
        help="Number of full iterations over prompts.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for concurrent requests."
    )
    parser.add_argument(
        "--input_seq_len",
        type=int,
        default=-1,
        help="Length parameter of the input sequence when using random prompts (not given dataset).",
    )
    parser.add_argument(
        "--output_seq_len",
        type=int,
        default=2048,
        help="Make completions all the same pre-defined maximum length for testing.",
    )
    parser.add_argument(
        "--inter_batch_delay",
        type=int,
        default=0,
        help="Seconds of delay between batches.",
    )
    parser.add_argument(
        "--vary_batch_size",
        action="store_true",
        help="Randomize normally the batch size for each batch of prompts.",
    )
    return parser


if __name__ == "__main__":
    main()

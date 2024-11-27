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

from utils.prompt_generation import add_prompt_gen_args, generate_prompts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_client_args(parser):
    parser.add_argument(
        "--stream", type=bool, default=False, help="Set stream to True or False."
    )
    parser.add_argument(
        "--vllm_model",
        type=bool,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name vLLM API server is using.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=805,
        help="Number of samples to use from the dataset.",
    )
    parser.add_argument(
        "--num_full_iterations",
        type=int,
        default=100,
        help="Number of full iterations to run over the dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for concurrent requests."
    )
    parser.add_argument(
        "--input_seq_len",
        type=int,
        default=-1,
        help="Make prompts all the same pre-defined length for testing.",
    )
    parser.add_argument(
        "--output_seq_len",
        type=int,
        default=2048,
        help="Make completeions all the same pre-defined maximum length for testing.",
    )
    return parser


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
    prompt, response_idx, stream, headers, api_url, max_tokens, vll_model
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
        "stop": ["<|eot_id|>"],
    }
    req_time = time.time()
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        api_url, json=json_data, headers=headers, stream=stream, timeout=600
    )
    # Handle chunked response
    full_text = ""
    num_tokens = 0
    first_token_time = 0
    if stream:
        if response.headers.get("transfer-encoding") == "chunked":
            for line in response.iter_lines(decode_unicode=True):
                # Process each line of data as it's received
                if line:
                    # Remove the 'data: ' prefix
                    if line.startswith("data: "):
                        if num_tokens == 0:
                            first_token_time = time.time()
                            ttft = first_token_time - req_time
                        num_tokens += 1
                        data_str = line[len("data: ") :].strip()
                        if data_str == "[DONE]":
                            num_tokens -= 1
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
            logger.info(response.text)
            raise ValueError("Response is not chunked")

    else:
        full_text = response.text
        # TODO: get tokens from tokenizer
        num_tokens = 2

    num_tokens = max(num_tokens, 2)
    throughput_time = max(time.time() - first_token_time, 0.0001)
    response_data = {
        "response_idx": response_idx,
        "prompt": prompt,
        "response": full_text,
        "num_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
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


def test_api_call_threaded_full_queue(
    prompts,
    batch_size,
    num_full_iterations,
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
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for _ in range(num_full_iterations):
            for response_idx, instruction in enumerate(prompts):
                future = executor.submit(
                    call_func, instruction, response_idx, **call_func_kwargs
                )
                futures.append(future)

        for future in as_completed(futures):
            try:
                response_data = future.result()
                # Write the response data to the JSONL file
                with responses_lock:
                    with open(json_fpath, "a") as f:
                        if response_counter > 0:
                            f.write(",")
                        json.dump(response_data, f, indent=4)
                response_counter += 1
                logger.info(
                    f"Processed {response_counter}/{total_prompts} responses. Avg. TPS: {response_data['tps']:.2f}, TTFT: {response_data['ttft']:.2f}, Num Tokens: {response_data['num_tokens']}"
                )
            except Exception as e:
                logger.error(f"Error processing a response: {e}")

    logger.info(f"Finished all requests, total responses: {response_counter}")
    with open(json_fpath, "a") as f:
        f.write("\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alpaca Evaluation Inference.")
    parser = add_client_args(parser)
    parser = add_prompt_gen_args(parser)
    args = parser.parse_args()

    # generate prompts
    prompts = generate_prompts(args)

    headers = {"Authorization": f"Bearer {get_authorization()}"}
    api_url = get_api_url()
    logging.info(f"API_URL: {api_url}")
    test_api_call_threaded_full_queue(
        prompts=prompts,
        batch_size=args.batch_size,
        num_full_iterations=args.num_full_iterations,
        call_func=call_inference_api,
        call_func_kwargs={
            "stream": args.stream,
            "headers": headers,
            "api_url": api_url,
            "max_tokens": args.output_seq_len,
            "vll_model": args.vllm_model,
        },
    )

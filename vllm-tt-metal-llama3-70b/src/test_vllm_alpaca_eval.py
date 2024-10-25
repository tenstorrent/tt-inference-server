# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import getpass
import threading
import logging
import json
from datetime import datetime
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

DEPLOY_URL = "http://127.0.0.1"
API_BASE_URL = f"{DEPLOY_URL}:8000"
API_URL = f"{API_BASE_URL}/v1/completions"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
n_samples = 805
# alpaca_eval contains 805 evaluation samples
alpaca_ds = load_dataset(
    "tatsu-lab/alpaca_eval",
    "alpaca_eval",
    split=f"eval[:{n_samples}]",
)

# Thread-safe data collection
responses_lock = threading.Lock()
responses = []


def call_inference_api(alpaca_instruction, response_idx):
    # set API prompt and optional parameters
    prompt = alpaca_instruction
    json_data = {
        "model": "meta-llama/Meta-Llama-3.1-70B",
        "prompt": prompt,
        "temperature": 1,
        "top_k": 20,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stream": True,
        "stop": ["<|eot_id|>"],
        # "include_stop_str_in_output": True,
        # "stop_sequence": None,
        # "return_prompt": None,
    }
    # using requests stream=True, make sure to set a timeout
    response = requests.post(API_URL, json=json_data, stream=True, timeout=600)
    # Handle chunked response
    full_text = ""
    if response.headers.get("transfer-encoding") == "chunked":
        for line in response.iter_lines(decode_unicode=True):
            # Process each line of data as it's received
            if line:
                # Remove the 'data: ' prefix
                if line.startswith("data: "):
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
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
    response_data = {
        "response_idx": response_idx,
        "instruction": alpaca_instruction,
        "response": full_text,
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


def test_api_call_threaded_full_queue():
    batch_size = 32  # Maximum number of concurrent threads
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
    # Default to 1 iteration of the Alpaca eval dataset
    NUM_FULL_ITERATIONS = 1000

    total_instructions = len(alpaca_ds["instruction"]) * NUM_FULL_ITERATIONS
    response_counter = 0
    logger.info(
        f"Running {total_instructions} prompts in full queue with batch size {batch_size}."
    )
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for _ in range(NUM_FULL_ITERATIONS):
            for response_idx, instruction in enumerate(alpaca_ds["instruction"]):
                future = executor.submit(call_inference_api, instruction, response_idx)
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
                    f"Processed {response_counter}/{total_instructions} responses."
                )
            except Exception as e:
                logger.error(f"Error processing a response: {e}")

    logger.info(f"Finished all requests, total responses: {response_counter}")
    with open(json_fpath, "a") as f:
        f.write("\n]")


if __name__ == "__main__":
    test_api_call_threaded_full_queue()

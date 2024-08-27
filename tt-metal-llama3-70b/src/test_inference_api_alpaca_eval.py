import os
import getpass
import threading
import time
import logging
import json
from datetime import datetime
import requests
from pathlib import Path

from datasets import load_dataset
from inference_config import inference_config

DEPLOY_URL = "http://127.0.0.1"
API_BASE_URL = f"{DEPLOY_URL}:{inference_config.backend_server_port}"
API_URL = f"{API_BASE_URL}/inference/{inference_config.inference_route_name}"
HEALTH_URL = f"{API_BASE_URL}/health"

# must set AUTHORIZATION
# see tt-metal-llama3-70b/README.md JWT_TOKEN Authorization instructions
headers = {"Authorization": os.environ["AUTHORIZATION"]}
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
n_batches = 25
n_samples = n_batches * 32
# alpaca_eval contains 805 evaluation samples
alpaca_ds = load_dataset(
    "tatsu-lab/alpaca_eval",
    "alpaca_eval",
    split=f"eval[:{n_samples}]",
)

# Thread-safe data collection
responses_lock = threading.Lock()
responses = []


def test_api_client_perf(alpaca_instruction, response_idx, print_streaming=False):
    # set API prompt and optional parameters
    prompt = alpaca_instruction
    json_data = {
        "text": prompt,
        "temperature": 1,
        "top_k": 20,
        "top_p": 0.9,
        "max_tokens": 2048,
        "stop_sequence": None,
        "return_prompt": None,
    }
    start_time = time.time()
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL, json=json_data, headers=headers, stream=True, timeout=600
    )
    # Handle chunked response
    full_text = ""
    if response.headers.get("transfer-encoding") == "chunked":
        for idx, chunk in enumerate(
            response.iter_content(chunk_size=None, decode_unicode=True)
        ):
            # Process each chunk of data as it's received
            full_text += chunk
            if print_streaming:
                # for debugging when writting to file after each batch isnt enough
                print(full_text)
    else:
        # If not chunked, you can access the entire response body at once
        logger.error("NOT CHUNKED!")
        logger.info(response.text)

    with responses_lock:
        responses.append(
            {
                "response_idx": response_idx,
                "instruction": alpaca_instruction,
                "output": full_text,
            }
        )


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


def test_api_call_threaded():
    batch_size = 32
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cache_root = Path(os.getenv("CACHE_ROOT"))
    json_fpath = cache_root / f"alpaca_eval_responses_{timestamp}.json"
    logger.info(f"will write output to: {json_fpath}")
    can_write, err_msg = check_json_fpath(json_fpath)
    if not can_write:
        err_msg += (
            f"\nNote: CACHE_ROOT:={cache_root}, consider setting in this shell to $PWD"
        )
    assert can_write, err_msg
    # default to 1 iteration of alpaca eval dataset
    NUM_FULL_ITERATIONS = 1
    for _ in range(NUM_FULL_ITERATIONS):
        for batch_idx in range(0, len(alpaca_ds) // batch_size):
            threads = []
            batch = alpaca_ds[
                (batch_idx * batch_size) : (batch_idx * batch_size) + batch_size
            ]
            logger.info(f"starting batch {batch_idx} ...")
            for i in range(0, batch_size):
                response_idx = (batch_idx * batch_size) + i
                thread = threading.Thread(
                    target=test_api_client_perf,
                    args=[batch["instruction"][i], response_idx, False],
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads in the current batch to complete
            for thread in threads:
                thread.join()

            logger.info(f"finished batch {batch_idx}.")

            # Save the responses to a JSON file incrementally
            with responses_lock:
                with open(json_fpath, "a") as f:
                    json.dump(responses, f, indent=4)
                    responses.clear()  # Clear responses after writing to file
        logger.info(f"finished all batches, batch_idx:={batch_idx}")


if __name__ == "__main__":
    test_api_call_threaded()

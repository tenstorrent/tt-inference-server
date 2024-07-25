import os
import threading
import time
import logging

import requests
from inference_config import inference_config

DEPLOY_URL = "http://127.0.0.1"
API_BASE_URL = f"{DEPLOY_URL}:{inference_config.backend_server_port}"
API_URL = f"{API_BASE_URL}/inference/{inference_config.inference_route_name}"
HEALTH_URL = f"{API_BASE_URL}/health"

headers = {"Authorization": os.environ.get("AUTHORIZATION")}
logger = logging.getLogger(__name__)


def test_api_client_perf(prompt_extra="", print_streaming=True):
    # set API prompt and optional parameters
    json_data = {
        "text": "What is in Austin Texas?" + prompt_extra,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.9,
        "max_tokens": 256,
        "stop_sequence": None,
        "return_prompt": None,
    }
    start_time = time.time()
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL, json=json_data, headers=headers, stream=True, timeout=240
    )
    # Handle chunked response
    full_text = ""
    if response.headers.get("transfer-encoding") == "chunked":
        print("processing chunks ...")
        for idx, chunk in enumerate(
            response.iter_content(chunk_size=None, decode_unicode=True)
        ):
            # Process each chunk of data as it's received
            full_text += chunk
            if print_streaming:
                print(full_text)
        end_time = time.time()
        tps = idx / (end_time - start_time)
        print(full_text)
        print(f"Client side tokens per second (TPS): {tps}")
    else:
        # If not chunked, you can access the entire response body at once
        print("NOT CHUNKED!")
        print(response.text)


def test_api_call_threaded():
    threads = []
    batch_size = 32
    for i in range(batch_size):
        thread = threading.Thread(target=test_api_client_perf, args=["", False])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished execution.")


if __name__ == "__main__":
    test_api_call_threaded()
    # test_api_client_perf()

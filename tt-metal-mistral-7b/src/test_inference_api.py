import os
import threading

import requests
from inference_config import inference_config

DEPLOY_URL = "http://127.0.0.1"
API_BASE_URL = f"{DEPLOY_URL}:{inference_config.backend_server_port}"
API_URL = f"{API_BASE_URL}{inference_config.inference_route}"
# API_URL="https://falcon-api--tenstorrent-playground.workload.tenstorrent.com/inference/falcon40b"
HEALTH_URL = f"{API_BASE_URL}/health"

headers = {"Authorization": os.environ.get("AUTHORIZATION")}
# headers = {"Authorization": os.environ.get("APIM_KEY")}


def test_valid_api_call(prompt_extra="", print_output=True):
    # set API prompt and optional parameters
    json_data = {
        "text": "what are dogs (the animal)?" + prompt_extra,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.9,
        "max_tokens": 128,
        "stop_sequence": None,
        "return_prompt": None,
    }
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL, json=json_data, headers=headers, stream=True, timeout=350
    )
    # Initialize an empty string to collect the response chunks
    full_response = ""

    # Handle chunked response
    if response.headers.get("transfer-encoding") == "chunked":
        print("processing chunks ...")
        for idx, chunk in enumerate(
            response.iter_content(chunk_size=None, decode_unicode=True)
        ):
            # Process each chunk of data as it's received
            # if print_output:
            #     print(f"chunk:{idx}")
            #     print(chunk)
            full_response += chunk 
    else:
        # If not chunked, you can access the entire response body at once
        print("NOT CHUNKED!")
        print(response.text)

    print("Full response:")
    print(full_response)


def test_bad_params_types_api_calls(prompt_extra="", print_output=True):
    # set API prompt and optional parameters
    json_data_list = [
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "temperature": "sdfgnskdgjn",
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "top_k": "ddgsd",
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "top_p": "3333ffaa",
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "max_tokens": "dg2",
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "max_tokens": "gsgsgg",
        },
    ]
    for jd in json_data_list:
        response = requests.post(
            API_URL, json=jd, headers=headers, stream=True, timeout=35
        )
        print(response.text)
        assert response.status_code == 400


def test_bad_params_bounds_api_calls(prompt_extra="", print_output=True):
    # set API prompt and optional parameters
    json_data_list = [
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "temperature": "0",
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "top_k": 0,
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "top_p": -1,
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "max_tokens": -1,
        },
        {
            "text": "Where should I go in Austin when I visit?" + prompt_extra,
            "max_tokens": 9999999,
        },
    ]
    for jd in json_data_list:
        response = requests.post(
            API_URL, json=jd, headers=headers, stream=True, timeout=35
        )
        print(response.text)
        assert response.status_code == 400


def test_api_call_threaded():
    threads = []

    for i in range(128):
        thread = threading.Thread(target=test_valid_api_call, args=[str(i), False])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished execution.")


def test_get_health():
    # breakpoint() 
    response = requests.get(HEALTH_URL, headers=headers, timeout=35)
    assert response.status_code == 200


if __name__ == "__main__":
    test_get_health()
    test_valid_api_call()
    # test_bad_params_types_api_calls()
    # test_bad_params_bounds_api_calls()

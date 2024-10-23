# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from collections import defaultdict

import json 


import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, FalconForCausalLM, pipeline


import requests
from inference_config import inference_config

DEPLOY_URL = "http://127.0.0.1"
API_BASE_URL = f"{DEPLOY_URL}:{inference_config.backend_server_port}"
API_URL = f"{API_BASE_URL}{inference_config.inference_route}"
# API_URL="https://falcon-api--tenstorrent-playground.workload.tenstorrent.com/inference/falcon40b"
HEALTH_URL = f"{API_BASE_URL}/health"

headers = {"Authorization": os.environ.get("AUTHORIZATION")}
# headers = {"Authorization": os.environ.get("APIM_KEY")}


def test_alpaca_eval_api_call(prompt_extra="", print_output=True):
    alpaca_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval[:5]")
    alpaca_set = list(alpaca_set)

    results = []

    # set API prompt and optional parameters
    i = 0 
    for example in alpaca_set: 
        print("prompt number: ", i)
        if i % 32 == 0:
            print(f"Reached {i} prompts!")

        if i == 512:
            print(f"Reached 512 prompts...now saving")
        json_data = {
            "text": example["instruction"] + prompt_extra,
            "temperature": 1,
            "top_k": 10,
            "top_p": 0.9,
            "max_tokens": 512,
            "stop_sequence": None,
            "return_prompt": None,
        }
        # using requests stream=True, make sure to set a timeout
        response = requests.post(
            API_URL, json=json_data, headers=headers, stream=True, timeout=3500
        )
        # # Handle chunked response
        # if response.headers.get("transfer-encoding") == "chunked":
        #     print("processing chunks ...")
        #     for idx, chunk in enumerate(
        #         response.iter_content(chunk_size=None, decode_unicode=True)
        #     ):
        #         # Process each chunk of data as it's received
        #         if print_output:
        #             print(f"chunk:{idx}")
        #             print(chunk)
        # else:
        #     # If not chunked, you can access the entire response body at once
        #     print("NOT CHUNKED!")
        #     print(response.text)
        response_text = response.text
        results.append({
            "instruction": example["instruction"],
            "response": response_text
        })

        # Append result to a file after each response
        file_name = "./api_responses_18_07_2024.json"
        with open(file_name, "a") as outfile:
            json.dump({
                "instruction": example["instruction"],
                "response": response_text
            }, outfile, indent=4)
            outfile.write("\n")  # Add newline for clarity

        i += 1

    print(f"Responses saved to {file_name}")



def test_get_health():
    breakpoint() 
    response = requests.get(HEALTH_URL, headers=headers, timeout=35)
    assert response.status_code == 200


if __name__ == "__main__":
    test_get_health()
    test_alpaca_eval_api_call()

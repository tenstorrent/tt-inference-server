# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import logging
import json
import time
import requests
from pprint import pprint

import jwt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    api_url = f"{base_url}/chat/completions"
    return api_url


def main():
    # message using openai api format
    # see: https://platform.openai.com/docs/api-reference/chat
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What'''s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        },
    ]

    headers = {"Authorization": f"Bearer {get_authorization()}"}
    api_url = get_api_url()
    stream = True
    logging.info(f"API_URL: {api_url}")
    # set API prompt and optional parameters
    json_data = {
        "model": os.environ.get("HF_MODEL_REPO_ID"),
        "messages": messages,
        "temperature": 1,
        "top_k": 20,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": stream,
    }
    req_time = time.perf_counter()
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
                            first_token_time = time.perf_counter()
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
                            content = data["choices"][0].get("delta").get("content")
                            full_text += content
                            logger.info(full_text)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON: {e}")
                            continue
        else:
            # If not chunked, you can access the entire response body at once
            logger.info(response.text)
            raise ValueError("Response is not chunked")

    else:
        full_text = response.text
        # TODO: get tokens from tokenizer
        ttft = 0
        num_tokens = 2
    end_time = time.perf_counter()
    num_tokens = max(num_tokens, 2)
    throughput_time = max(end_time - first_token_time, 0.0001)
    e2el = end_time - req_time
    response_data = {
        "response": full_text,
        "output_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
        "ttft": ttft,
        "e2el": e2el,
    }
    pprint(response_data)


if __name__ == "__main__":
    main()

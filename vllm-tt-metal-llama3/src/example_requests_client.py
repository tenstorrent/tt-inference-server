# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import logging
import json
import time
import requests
import argparse
from pprint import pprint
import asyncio
import aiohttp
import statistics

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


def make_request(api_url, headers, json_data, user_input_prompt):
    stream = json_data.get("stream", True)
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
        "prompt": user_input_prompt,
        "response": full_text,
        "output_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
        "ttft": ttft,
        "e2el": e2el,
    }
    return response_data


async def async_make_request(
    session, api_url, headers, json_data, user_input_prompt, request_id
):
    stream = json_data.get("stream", True)
    req_time = time.perf_counter()

    async with session.post(
        api_url, json=json_data, headers=headers, timeout=600
    ) as response:
        full_text = ""
        num_tokens = 0
        first_token_time = 0

        if stream:
            async for line in response.content:
                line = line.decode("utf-8").strip()
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
                        data = json.loads(data_str)
                        content = data["choices"][0].get("delta").get("content", "")
                        full_text += content
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to decode JSON in request {request_id}: {e}"
                        )
                        continue
        else:
            full_text = await response.text()
            ttft = 0
            num_tokens = 2

    end_time = time.perf_counter()
    num_tokens = max(num_tokens, 2)
    throughput_time = max(end_time - first_token_time, 0.0001)
    e2el = end_time - req_time

    response_data = {
        "request_id": request_id,
        "prompt": user_input_prompt,
        "response": full_text,
        "output_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
        "ttft": ttft,
        "e2el": e2el,
    }

    logger.info(
        f"Request {request_id} completed - TTFT: {ttft:.4f}s, TPS: {(num_tokens - 1) / throughput_time:.2f}"
    )
    return response_data


async def run_concurrent_requests(n, api_url, headers, json_data, user_input_prompt):
    start_time = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(n):
            task = asyncio.create_task(
                async_make_request(
                    session, api_url, headers, json_data, user_input_prompt, i + 1
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        return results, total_time


async def run_batched_requests(
    total_requests, concurrent_limit, api_url, headers, json_data, user_input_prompt
):
    all_results = []
    start_time = time.perf_counter()
    request_id = 1

    async with aiohttp.ClientSession() as session:
        # Process requests in batches of size concurrent_limit
        for batch_start in range(0, total_requests, concurrent_limit):
            batch_size = min(concurrent_limit, total_requests - batch_start)
            logger.info(
                f"Processing batch of {batch_size} requests ({batch_start+1}-{batch_start+batch_size} of {total_requests})..."
            )

            batch_tasks = []
            for i in range(batch_size):
                task = asyncio.create_task(
                    async_make_request(
                        session,
                        api_url,
                        headers,
                        json_data,
                        user_input_prompt,
                        request_id,
                    )
                )
                batch_tasks.append(task)
                request_id += 1

            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)

    total_time = time.perf_counter() - start_time
    return all_results, total_time


def main():
    parser = argparse.ArgumentParser(
        description="Client for interacting with the LLM API"
    )
    parser.add_argument("--prompt", type=str, help="Prompt to send to the model")
    parser.add_argument(
        "--num_concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (sets temperature to 0.0)",
    )
    parser.add_argument(
        "--n_requests",
        type=int,
        help="Total number of requests to make (minimum: num_concurrent)",
    )
    args = parser.parse_args()

    model = os.environ.get("HF_MODEL_REPO_ID")
    print("\n")

    # Use command-line argument if provided, otherwise prompt user
    user_input_prompt = args.prompt
    if user_input_prompt is None:
        user_input_prompt = input(f"Enter your prompt for {model}: ")

    # message using openai api format
    # see: https://platform.openai.com/docs/api-reference/chat
    messages = [
        {
            "role": "user",
            "content": user_input_prompt,
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
        "temperature": 0.0 if args.greedy else 0.9,
        "top_k": 20,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": stream,
    }

    if args.num_concurrent > 1:
        logger.info(f"Making {args.num_concurrent} concurrent requests...")

        # Determine total number of requests to make
        total_requests = (
            args.n_requests if args.n_requests is not None else args.num_concurrent
        )
        total_requests = max(total_requests, args.num_concurrent)

        if total_requests > args.num_concurrent:
            logger.info(
                f"Running a total of {total_requests} requests in batches of {args.num_concurrent}..."
            )
            results, total_time = asyncio.run(
                run_batched_requests(
                    total_requests,
                    args.num_concurrent,
                    api_url,
                    headers,
                    json_data,
                    user_input_prompt,
                )
            )
        else:
            results, total_time = asyncio.run(
                run_concurrent_requests(
                    args.num_concurrent, api_url, headers, json_data, user_input_prompt
                )
            )

        # Calculate aggregate statistics
        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        avg_e2el = sum(r["e2el"] for r in results) / len(results)

        # Calculate min/max values
        min_ttft = min(r["ttft"] for r in results)
        max_ttft = max(r["ttft"] for r in results)
        min_tps = min(r["tps"] for r in results)
        max_tps = max(r["tps"] for r in results)
        min_e2el = min(r["e2el"] for r in results)
        max_e2el = max(r["e2el"] for r in results)

        # Calculate standard deviations
        if len(results) > 1:
            std_ttft = statistics.stdev(r["ttft"] for r in results)
            std_tps = statistics.stdev(r["tps"] for r in results)
            std_e2el = statistics.stdev(r["e2el"] for r in results)
        else:
            std_ttft = std_tps = std_e2el = 0

        # Calculate total tokens and system throughput
        total_tokens = sum(r["output_tokens"] for r in results)
        system_throughput = total_tokens / total_time if total_time > 0 else 0

        # Print individual results if needed
        for i, result in enumerate(results):
            print(f"\nRequest {i+1} Results:")
            pprint(result)

        # Print comprehensive summary statistics
        print("\n" + "=" * 50)
        print(
            f"SUMMARY STATISTICS FOR {args.num_concurrent} CONCURRENT REQUESTS, {total_requests} TOTAL REQUESTS"
        )
        print("=" * 50)
        print(f"Total time for all requests: {total_time:.4f}s")
        print(f"Total tokens generated: {total_tokens}")
        print(f"System throughput: {system_throughput:.2f} tokens/sec")
        print("\nTime To First Token (TTFT):")
        print(f"  Average Total: {avg_ttft:.4f}s")
        print(f"  Average Per Request: {avg_ttft / args.num_concurrent:.4f}s")
        print(f"  Min: {min_ttft:.4f}s")
        print(f"  Max: {max_ttft:.4f}s")
        print(f"  Std Dev: {std_ttft:.4f}s")
        print("\nTokens Per Second (TPS):")
        print(f"  Average: {avg_tps:.2f}")
        print(f"  Min: {min_tps:.2f}")
        print(f"  Max: {max_tps:.2f}")
        print(f"  Std Dev: {std_tps:.2f}")
        print("\nEnd-to-End Latency (E2EL):")
        print(f"  Average: {avg_e2el:.4f}s")
        print(f"  Min: {min_e2el:.4f}s")
        print(f"  Max: {max_e2el:.4f}s")
        print(f"  Std Dev: {std_e2el:.4f}s")
        print("=" * 50)
    else:
        # Single request mode (original behavior)
        response_data = make_request(api_url, headers, json_data, user_input_prompt)
        pprint(response_data)


if __name__ == "__main__":
    main()

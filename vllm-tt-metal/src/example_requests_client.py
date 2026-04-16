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
            logger.warning(
                "Neither AUTHORIZATION nor JWT_SECRET environment variables are set. "
                "Proceeding without authorization."
            )
            return None
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
    usage = None
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
                            # Check if this is the special usage chunk (will have usage data and empty choices)
                            if (
                                "usage" in data
                                and data["usage"] is not None
                                and data.get("choices", []) == []
                            ):
                                usage = data["usage"]
                                num_tokens -= 1  # Don't count this as a token
                                continue

                            # Extract text from the 'choices' field
                            content = (
                                data["choices"][0].get("delta", {}).get("content", "")
                            )
                            if content:
                                full_text += content
                                logger.info(full_text)

                            # Note: Other chunks may have usage: null
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to decode JSON: {e}")
                            continue
        else:
            # If not chunked, you can access the entire response body at once
            logger.info(response.text)
            raise ValueError("Response is not chunked")

    else:
        response_json = response.json()
        full_text = (
            response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if "usage" in response_json:
            usage = response_json["usage"]
        # TODO: get tokens from tokenizer
        ttft = 0
        num_tokens = 2
    end_time = time.perf_counter()
    num_tokens = max(num_tokens, 2)
    throughput_time = max(end_time - first_token_time, 0.0001)
    e2el = end_time - req_time

    # Create a default usage dict if none was returned by the API
    if usage is None:
        usage = {
            "prompt_tokens": 0,  # We don't know the count
            "completion_tokens": num_tokens,
            "total_tokens": num_tokens,  # This is an underestimate without prompt tokens
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        }

    response_data = {
        "prompt": user_input_prompt,
        "response": full_text,
        "output_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
        "ttft": ttft,
        "e2el": e2el,
        "usage": usage,
    }
    return response_data


async def async_make_request(
    session, api_url, headers, json_data, user_input_prompt, request_id, prompts=None
):
    # If prompts is provided and not empty, use the appropriate prompt based on request_id
    prompt_to_use = user_input_prompt
    if prompts and len(prompts) > 0:
        prompt_to_use = prompts[(request_id - 1) % len(prompts)]

    # Create a copy of json_data to avoid modifying the original
    request_json_data = json_data.copy()

    # Update the prompt in the messages
    if prompt_to_use != user_input_prompt:
        request_json_data["messages"] = [
            {
                "role": "user",
                "content": prompt_to_use,
            },
        ]

    stream = request_json_data.get("stream", True)
    req_time = time.perf_counter()

    async with session.post(
        api_url, json=request_json_data, headers=headers, timeout=600
    ) as response:
        full_text = ""
        num_tokens = 0
        first_token_time = 0
        usage = None

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
                        # Check if this is the special usage chunk (will have usage data and empty choices)
                        if (
                            "usage" in data
                            and data["usage"] is not None
                            and data.get("choices", []) == []
                        ):
                            usage = data["usage"]
                            num_tokens -= 1  # Don't count this as a token
                            continue

                        content = data["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            full_text += content

                        # Note: Other chunks may have usage: null
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to decode JSON in request {request_id}: {e}"
                        )
                        continue
            assert num_tokens > 0, "No tokens were generated. response.content: " + str(
                line
            )
        else:
            response_json = await response.json()
            full_text = (
                response_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if "usage" in response_json:
                usage = response_json["usage"]
            ttft = 0
            num_tokens = 2

    end_time = time.perf_counter()
    num_tokens = max(num_tokens, 2)
    throughput_time = max(end_time - first_token_time, 0.0001)
    e2el = end_time - req_time

    # Create a default usage dict if none was returned by the API
    if usage is None:
        usage = {
            "prompt_tokens": 0,  # We don't know the count
            "completion_tokens": num_tokens,
            "total_tokens": num_tokens,  # This is an underestimate without prompt tokens
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        }

    response_data = {
        "request_id": request_id,
        "prompt": prompt_to_use,
        "response": full_text,
        "output_tokens": num_tokens,
        "tps": (num_tokens - 1) / throughput_time,
        "ttft": ttft,
        "e2el": e2el,
        "usage": usage,
    }

    logger.info(
        f"Request {request_id} completed - TTFT: {ttft:.4f}s, TPS: {(num_tokens - 1) / throughput_time:.2f}"
    )
    return response_data


async def run_concurrent_requests(
    n, api_url, headers, json_data, user_input_prompt, prompts=None
):
    start_time = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(n):
            task = asyncio.create_task(
                async_make_request(
                    session,
                    api_url,
                    headers,
                    json_data,
                    user_input_prompt,
                    i + 1,
                    prompts,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        return results, total_time


async def run_batched_requests(
    total_requests,
    concurrent_limit,
    api_url,
    headers,
    json_data,
    user_input_prompt,
    prompts=None,
):
    all_results = []
    start_time = time.perf_counter()
    request_id = 1

    async with aiohttp.ClientSession() as session:
        # Process requests in batches of size concurrent_limit
        for batch_start in range(0, total_requests, concurrent_limit):
            batch_size = min(concurrent_limit, total_requests - batch_start)
            logger.info(
                f"Processing batch of {batch_size} requests ({batch_start + 1}-{batch_start + batch_size} of {total_requests})..."
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
                        prompts,
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
        "--model",
        type=str,
        help="Model name to use (overrides HF_MODEL_REPO_ID environment variable)",
    )
    parser.add_argument(
        "--prompt_json_path",
        type=str,
        help="Path to a JSON file containing an array of prompts",
    )
    parser.add_argument(
        "--num_concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)",
    )
    parser.add_argument(
        "--n_requests",
        type=int,
        default=1,
        help="Total number of requests to make (minimum: num_concurrent)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Force generation of max_tokens regardless of stop tokens",
    )
    args = parser.parse_args()

    assert args.temperature >= 0.0 and args.temperature <= 1.0, (
        "temperature:={args.temperature} must be between 0.0 and 1.0"
    )
    assert args.max_tokens > 0, f"max_tokens:={args.max_tokens} must be greater than 0"
    assert args.num_concurrent > 0, (
        f"num_concurrent:={args.num_concurrent} must be greater than 0"
    )
    assert args.n_requests > 0, f"n_requests:={args.n_requests} must be greater than 0"

    model = args.model
    if model is None:
        model = os.environ.get("HF_MODEL_REPO_ID")
        if model is None:
            raise ValueError(
                "Model name is not specified via --model argument or HF_MODEL_REPO_ID environment variable. "
            )
    print("\n")

    # Load prompts from JSON file if specified
    prompts = []
    if args.prompt_json_path:
        try:
            with open(args.prompt_json_path, "r") as f:
                prompts_data = json.load(f)
                prompts = [item["prompt"] for item in prompts_data]
                if not prompts:
                    logger.error("No prompts found in JSON file")
                    return
                logger.info(
                    f"Loaded {len(prompts)} prompts from {args.prompt_json_path}"
                )

                # If n_requests is specified, show how prompts will be used
                if args.n_requests and args.n_requests > len(prompts):
                    logger.info(
                        f"Will cycle through prompts ({len(prompts)}) to fulfill {args.n_requests} requests"
                    )
        except Exception as e:
            logger.error(f"Error loading prompts from JSON file: {e}")
            return

    # Use command-line argument if provided, otherwise prompt user
    user_input_prompt = args.prompt
    if user_input_prompt is None:
        if prompts:
            user_input_prompt = prompts[0]
        else:
            user_input_prompt = input(f"Enter your prompt for {model}: ")

    # message using openai api format
    # see: https://platform.openai.com/docs/api-reference/chat
    messages = [
        {
            "role": "user",
            "content": user_input_prompt,
        },
    ]

    headers = {}
    authorization = get_authorization()
    if authorization is not None:
        headers["Authorization"] = f"Bearer {authorization}"
    api_url = get_api_url()
    stream = True
    logging.info(f"API_URL: {api_url}")

    # set API prompt and optional parameters
    json_data = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": stream,
    }
    # Add ignore_eos parameter if specified
    if args.ignore_eos:
        json_data["ignore_eos"] = True

    if stream:
        json_data["stream_options"] = {
            "include_usage": True,
        }

    # Determine total number of requests to make
    total_requests = max(args.n_requests, args.num_concurrent)
    if args.num_concurrent > 1:
        logger.info(f"Making {args.num_concurrent} concurrent requests...")

        # Modify the run_batched_requests and run_concurrent_requests to use the prompts list
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
                    prompts,
                )
            )
        else:
            results, total_time = asyncio.run(
                run_concurrent_requests(
                    args.num_concurrent,
                    api_url,
                    headers,
                    json_data,
                    user_input_prompt,
                    prompts,
                )
            )

        # Calculate aggregate statistics
        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        avg_e2el = sum(r["e2el"] for r in results) / len(results)

        # Calculate token usage statistics
        total_prompt_tokens = sum(
            r.get("usage", {}).get("prompt_tokens", 0) for r in results
        )
        total_completion_tokens = sum(
            r.get("usage", {}).get("completion_tokens", 0) for r in results
        )
        total_tokens_used = sum(
            r.get("usage", {}).get("total_tokens", 0) for r in results
        )

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

        # Print individual results if needed
        for i, result in enumerate(results):
            print(f"\nRequest {i + 1} Results:")
            pprint(result)

        # Print comprehensive summary statistics
        print("\n" + "=" * 50)
        print(
            f"SUMMARY STATISTICS FOR {args.num_concurrent} CONCURRENT REQUESTS, {total_requests} TOTAL REQUESTS"
        )
        print("=" * 50)
        print(f"Total time for all requests: {total_time:.4f}s")
        print(f"Total tokens generated: {total_tokens}")

        print("\nToken Usage Statistics (avg is per user):")
        print(f"  Total Prompt Tokens: {total_prompt_tokens}")
        print(f"  Avg Prompt Tokens: {total_prompt_tokens / total_requests}")
        print(f"  Total Completion Tokens: {total_completion_tokens}")
        print(f"  Avg Completion Tokens: {total_completion_tokens / total_requests}")
        print(f"  Total context: {total_tokens_used}")
        print(f"  Avg context: {total_tokens_used / total_requests}")

        print("\nTime To First Token (TTFT):")
        print(f"  Average Total: {avg_ttft:.4f}s")
        print(f"  Average per user: {avg_ttft / args.num_concurrent:.4f}s")
        print(f"  Min: {min_ttft:.4f}s")
        print(f"  Max: {max_ttft:.4f}s")
        print(f"  Std Dev: {std_ttft:.4f}s")
        print(
            f"  Prefill Tput t/s: {(total_prompt_tokens / total_requests) / (avg_ttft / args.num_concurrent):.4f}s"
        )
        print("\nTokens Per Second (t/s/user):")
        print(f"  Average per user: {avg_tps:.2f}")
        print(f"  Min: {min_tps:.2f}")
        print(f"  Max: {max_tps:.2f}")
        print(f"  Std Dev: {std_tps:.2f}")
        print("\nEnd-to-End Latency (E2EL):")
        print(f"  Average per user: {avg_e2el:.4f}s")
        print(f"  Min: {min_e2el:.4f}s")
        print(f"  Max: {max_e2el:.4f}s")
        print(f"  Std Dev: {std_e2el:.4f}s")
        print("=" * 50)
    else:
        # Single request mode (original behavior)
        # If prompts list is available, use the first prompt
        if prompts:
            # Update the messages with the first prompt from the list
            json_data["messages"] = [
                {
                    "role": "user",
                    "content": prompts[0],
                }
            ]
            user_input_prompt = prompts[0]

        for loop_num in range(total_requests):
            logger.info(f"request: {loop_num}/{total_requests}")
            response_data = make_request(api_url, headers, json_data, user_input_prompt)
            pprint(response_data)


if __name__ == "__main__":
    main()

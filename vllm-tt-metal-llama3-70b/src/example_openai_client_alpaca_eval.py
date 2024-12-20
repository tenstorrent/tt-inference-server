# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import threading
import logging
import time

from openai import OpenAI

from example_requests_client_alpaca_eval import (
    parse_args,
    get_api_base_url,
    load_dataset_samples,
    get_authorization,
    test_api_call_threaded_full_queue,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Thread-safe data collection
responses_lock = threading.Lock()
responses = []


def call_inference_api(prompt, response_idx, stream=True, headers=None, client=None):
    # set API prompt and optional parameters
    req_time = time.time()
    full_text = ""
    num_tokens = 0
    try:
        # Use OpenAI client to call API
        completion = client.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct",
            prompt=prompt,
            temperature=1,
            max_tokens=2048,
            top_p=0.9,
            stream=stream,
        )
        if stream:
            for event in completion:
                if event.choices[0].finish_reason is not None:
                    break
                if num_tokens == 0:
                    first_token_time = time.time()
                    ttft = first_token_time - req_time
                num_tokens += 1
                content = event.choices[0].text
                full_text += content
        else:
            full_text = completion.choices[0].text
            # Assuming tokens were returned with response (using len to mock token length)
            num_tokens = len(full_text.split())
            first_token_time = req_time  # Simplify for non-stream
            ttft = time.time() - req_time
    except Exception as e:
        logger.error(f"Error calling API: {e}")
        elapsed_time = time.time() - req_time
        logger.error(
            f"Before error: elapsed_time={elapsed_time}, num_tokens: {num_tokens}, full_text: {full_text}"
        )
        full_text = "ERROR"
        num_tokens = 0
        first_token_time = time.time()
        ttft = 0.001

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


if __name__ == "__main__":
    logger.info(
        "Note: OpenAI API client adds additional latency of ~10 ms to the API call."
    )
    args = parse_args()
    prompts = load_dataset_samples(args.n_samples)
    headers = {"Authorization": f"Bearer {get_authorization()}"}
    base_url = get_api_base_url()
    logging.info(f"BASE_API_URL: {base_url}")
    client = OpenAI(
        base_url=base_url,
        api_key=get_authorization(),
    )
    test_api_call_threaded_full_queue(
        prompts=prompts,
        batch_size=args.batch_size,
        num_full_iterations=args.num_full_iterations,
        call_func=call_inference_api,
        call_func_kwargs={
            "stream": args.stream,
            "headers": headers,
            "client": client,
            "max_tokens": args.output_seq_len,
        },
    )

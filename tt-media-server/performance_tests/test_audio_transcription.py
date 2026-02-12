# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#!/usr/bin/env python3
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp
import pytest

BASE_URL = "http://localhost:8014"
API_URL = f"{BASE_URL}/audio/transcriptions"

output_dir = "./audio_transcription_eval_results"
logs_output_dir = f"{output_dir}/run_logs"
results_output_dir = f"{output_dir}/run_results"

# TODO: Update this configuration based on actual test environment - currently hardcoded for 4 devices
num_devices_list = ["4"]
batch_size_list = ["1", "2", "4", "8", "16", "32"]

# Load test payload
with open("static/data/audio_test.json", "r") as f:
    payload = json.load(f)

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


def check_server_health():
    max_attempts = 3000
    attempt = 1
    wait_seconds = 5

    while attempt <= max_attempts:
        try:
            result = subprocess.run(
                ["curl", "-s", f"{BASE_URL}/tt-liveness"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                response = result.stdout
                if '"status":"alive"' in response and '"model_ready":true' in response:
                    print("Server is healthy and ready!...")
                    return True

            print("Server not ready yet. Waiting 5 seconds...")
            time.sleep(wait_seconds)
            attempt += 1

        except Exception:
            time.sleep(wait_seconds)
            attempt += 1

    print(
        f"ERROR: Server failed to start within {max_attempts * wait_seconds} seconds..."
    )
    return False


def print_to_file(message: str, output_file: str):
    with open(output_file, "a") as f:
        f.write(f"{message}\n")


@pytest.mark.asyncio
async def test_concurrent_audio_transcription(
    batch_size, results_output_file, log_output_file
):
    async def timed_request(session, index):
        print_to_file(f"Starting request {index}", log_output_file)
        try:
            start = time.perf_counter()
            async with session.post(API_URL, json=payload, headers=headers) as response:
                duration = time.perf_counter() - start
                if response.status == 200:
                    data = await response.json()
                    # Save response data to the results_output_file JSON
                    results_dir = os.path.dirname(os.path.abspath(results_output_file))
                    os.makedirs(results_dir, exist_ok=True)  # Ensure directory exists
                    with open(results_output_file, "w") as f:
                        json.dump(data, f, indent=4)
                print_to_file(
                    f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    log_output_file,
                )
                return duration

        except Exception as e:
            print_to_file(e, log_output_file)
            duration = time.perf_counter() - start
            print_to_file(
                f"[{index}] Error after {duration:.2f}s: {e}", log_output_file
            )
            assert False, f"Request {index} failed"

    # First iteration is warmup, second is measured (original behavior)
    for iteration in range(2):
        session_timeout = aiohttp.ClientTimeout(total=2000)
        async with aiohttp.ClientSession(
            headers=headers, timeout=session_timeout
        ) as session:
            tasks = [timed_request(session, i + 1) for i in range(batch_size)]
            results = await asyncio.gather(*tasks)
            requests_duration = max(results)
            total_duration = sum(results)
            avg_duration = total_duration / batch_size
        if iteration == 0:
            print_to_file("\n Warm up run done.", log_output_file)

    print_to_file(
        f"\n🚀 Time taken for individual concurrent requests : {results}",
        log_output_file,
    )
    print_to_file(
        f"\n🚀 Total time for {batch_size} concurrent requests: {requests_duration:.2f}s",
        log_output_file,
    )
    print_to_file(
        f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s",
        log_output_file,
    )


def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    for i in num_devices_list:
        print(f"Running localhost server on {BASE_URL}...")

        for j in batch_size_list:
            if check_server_health():
                print(f"Running API test with batch size: {j}...")

                results_output_file = (
                    f"{results_output_dir}/num_dev_{i}_batch_size_{j}.json"
                )
                log_output_file = f"{logs_output_dir}/num_dev_{i}_batch_size_{j}.log"

                # Clear the log file if it exists
                if os.path.exists(log_output_file):
                    with open(log_output_file, "w") as f:
                        f.write("")

                try:
                    asyncio.run(
                        test_concurrent_audio_transcription(
                            int(j), results_output_file, log_output_file
                        )
                    )
                    print("API test completed successfully...")
                except Exception as e:
                    print(f"API test failed: {e}")
                    sys.exit(1)

            else:
                print("Server health check failed...")
                sys.exit(1)

        print(f"Test run completed for device configuration {i}...")


if __name__ == "__main__":
    main()

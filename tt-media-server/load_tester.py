# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import requests
import time
import sys

API_URL = "http://localhost:8000/image/generations"
AUTH_TOKEN = "your-secret-key"
LOG_FILE = "api_status.log"

# JSON payload you want to send
PAYLOAD = {"prompt": "Michael Jordan blocked by Spud Webb", "num_inference_steps": 5}

# Headers matching your curl command
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json",
}


def check_api():
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=PAYLOAD, headers=HEADERS, timeout=30)
        elapsed = time.time() - start_time
        return "ok", elapsed if response.status_code == 200 else "nok"
    except Exception as e:
        return e


def main():
    # Get number of runs from command line argument or default to 150
    num_run_times = 150
    if len(sys.argv) > 1:
        try:
            num_run_times = int(sys.argv[1])
        except ValueError:
            print("Invalid argument. Using default value of 150.")

    print(f"Running inference {num_run_times} times...")

    for i in range(num_run_times):
        status, elapsed = check_api()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - Run {i + 1}/{num_run_times} - {status} time: {elapsed}")

    print(f"Completed {num_run_times} runs.")


if __name__ == "__main__":
    main()

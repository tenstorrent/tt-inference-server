# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os

from data_reader import DataReader
from locust import FastHttpUser, events, tag, task


# Constants for timeouts and API configuration
NETWORK_TIMEOUT = 300.0
CONNECTION_TIMEOUT = 300.0
AUTHORIZATION_HEADER = {"Authorization": os.environ["AUTHORIZATION"]}
API_ENDPOINT = "/inference/llama3-70b"
DEFAULT_PARAMS = {
    "temperature": 1.0,
    "top_k": 10,
    "top_p": 0.9,
    "stop_sequence": None,
    "return_prompt": None,
}

# Global variable to store data iterator
data_iter = None

# Event listener to load custom data before tests start
@events.test_start.add_listener
def load_custom_data(**kwargs):
    global data_iter
    data_iter = DataReader(with_shuffle=False)

class ServeUser(FastHttpUser):
    # Set test parameters
    network_timeout = NETWORK_TIMEOUT
    connection_timeout = CONNECTION_TIMEOUT
    headers = AUTHORIZATION_HEADER

    def post_request(self, prompt: str, max_tokens: int):
        """Helper method to send a POST request to the API with the given prompt and token limit."""
        json_data = {
            "text": prompt,
            **DEFAULT_PARAMS,  # Merge default parameters
            "max_tokens": max_tokens,
        }
        response = self.client.post(API_ENDPOINT, json=json_data, headers=self.headers)
        return response

    @tag("static")
    @task
    def basic_test(self):
        """Test using a static prompt."""
        prompt = "What is in Austin Texas?"
        self.post_request(prompt, max_tokens=128)

    @tag("dynamic")
    @task
    def dataset_test(self):
        """Test using dynamic prompts from a data iterator."""
        prompt = next(data_iter)
        print(f"Prompt: {prompt}")
        self.post_request(prompt, max_tokens=256)

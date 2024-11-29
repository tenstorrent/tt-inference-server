# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

from data_reader import DataReader
from locust import FastHttpUser, events, tag, task


# Constants for timeouts and API configuration
NETWORK_TIMEOUT = 300.0
CONNECTION_TIMEOUT = 300.0
API_ENDPOINT = "/v1/completions"
DEFAULT_PARAMS = {
    "model": "meta-llama/Meta-Llama-3.1-70B",
    "temperature": 1.0,
    "top_k": 10,
    "top_p": 0.9,
}

# Global variable to store data iterator
data_iter = None

# Event listener to load custom data before tests start
@events.test_start.add_listener
def load_custom_data(environment, **kwargs):
    global data_iter
    if "dynamic" in environment.parsed_options.tags:
        data_iter = DataReader()
        print("Dynamic data loaded.")
    else:
        print("Dynamic data not loaded (no 'dynamic' tag).")

class ServeUser(FastHttpUser):
    # Set test parameters
    network_timeout = NETWORK_TIMEOUT
    connection_timeout = CONNECTION_TIMEOUT

    def post_request(self, prompt: str, max_tokens: int):
        """Helper method to send a POST request to the API with the given prompt and token limit."""
        json_data = {
            "prompt": prompt,
            **DEFAULT_PARAMS,  # Merge default parameters
            "max_tokens": max_tokens,
        }
        response = self.client.post(API_ENDPOINT, json=json_data)
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
        self.post_request(prompt, max_tokens=128)

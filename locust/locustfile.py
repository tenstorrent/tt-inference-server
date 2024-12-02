# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import json
import os

import jwt

from data_reader import DataReader
from locust import FastHttpUser, events, task

# Constants for timeouts and API configuration
NETWORK_TIMEOUT = 300.0
CONNECTION_TIMEOUT = 300.0
API_ENDPOINT = "/v1/completions"
DEFAULT_PARAMS = {
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "temperature": 1.0,
    "top_k": 10,
    "top_p": 0.9,
}

# Global variable to store data iterator
data_iter = None

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


# Event listener to load custom data before tests start
@events.test_start.add_listener
def load_custom_data(**kwargs):
    global data_iter
    data_iter = DataReader()


class ServeUser(FastHttpUser):
    # Set test parameters
    network_timeout = NETWORK_TIMEOUT
    connection_timeout = CONNECTION_TIMEOUT
    headers = {"Authorization": f"Bearer {get_authorization()}"}

    def post_request(self, prompt: str, max_tokens: int):
        """Helper method to send a POST request to the API with the given prompt and token limit."""
        json_data = {
            "prompt": prompt,
            **DEFAULT_PARAMS,  # Merge default parameters
            "max_tokens": max_tokens,
        }
        response = self.client.post(API_ENDPOINT, json=json_data, headers=self.headers)
        return response

    @task
    def dataset_test(self):
        """Test using generated prompts from a data iterator."""
        prompt = next(data_iter)
        self.post_request(prompt, max_tokens=128)

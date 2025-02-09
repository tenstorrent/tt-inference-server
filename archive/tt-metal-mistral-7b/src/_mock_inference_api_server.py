# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from time import sleep
from unittest.mock import Mock, patch

from mistral_7b_backend import PrefillDecodeBackend
from inference_api_server import (
    app,
    initialize_decode_backend,
)
from inference_config import inference_config

"""
This script runs the flask server and initialize_decode_backend()
with the actual model mocked out.

This allows for rapid testing of the server and backend implementation.
"""


def mock_decoder(self):
    # mock with repeating previous token
    tps = 3000  # simulate a given tokens per second per user
    sleep(1 / tps)
    output_tokens = self.input_ids[-1].unsqueeze(0)
    # if user has hit max_length, send eos token
    for idx, user in enumerate(self.users):
        if user is not None:
            output_tokens[0, idx] = self.tokenizer(
                str(user.position_id % 10)
            ).input_ids[0]
            if (user.position_id - user.prompt_length + 1) >= user.max_tokens:
                output_tokens[0, idx] = self.tokenizer.eos_token_id
            elif (
                (user.stop_sequence is not None)
                and (user.position_id - user.prompt_length + 1) > 0
                and (output_tokens[0, idx] == user.stop_sequence)
            ):
                output_tokens[0, idx] = self.tokenizer.eos_token_id
    # update the new tokens generated to the input id
    self.input_ids = output_tokens.view(1, self.max_users)


def mock_load_model_and_tokenizer(self, args):
    from transformers import AutoTokenizer

    # # mock model
    model = None
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)
    return model, tokenizer


def mock_post_init_pybudify(self, args):
    pass


backend_initialized = False
api_log_dir = os.path.join(inference_config.log_cache, "api_logs")


def global_backend_init():
    global backend_initialized
    if not backend_initialized:
        # Create server log directory
        if not os.path.exists(api_log_dir):
            os.makedirs(api_log_dir)
        # override_args = get_backend_override_args()
        # initialize_decode_backend(override_args)
        initialize_decode_backend()
        backend_initialized = True


@patch.object(PrefillDecodeBackend, "decode", new=mock_decoder)
# @patch.object(PrefillDecodeBackend, "_post_init_pybudify", new=mock_post_init_pybudify)
@patch.object(
    PrefillDecodeBackend, "load_model_and_tokenizer", new=mock_load_model_and_tokenizer
)
def create_test_server():
    from flask_cors import CORS

    # CORS for swagger-ui local testing
    CORS(
        app,
        supports_credentials=True,
        resources={r"/predictions/*": {"origins": "http://localhost:8080"}},
    )
    global_backend_init()
    return app


if __name__ == "__main__":
    app = create_test_server()
    app.run(
        port=inference_config.backend_server_port,
        host="0.0.0.0",
    )

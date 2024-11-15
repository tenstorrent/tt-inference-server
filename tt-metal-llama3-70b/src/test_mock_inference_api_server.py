# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from unittest.mock import Mock, patch


from llama3_70b_backend import PrefillDecodeBackend
from inference_api_server import (
    app,
    initialize_decode_backend,
)
from inference_config import inference_config

from test_llama3_70b_backend_mock import mock_init_model

"""
This script runs the flask server and initialize_decode_backend()
with the actual model mocked out.

This allows for rapid testing of the server and backend implementation.
"""

backend_initialized = False
api_log_dir = os.path.join(inference_config.log_cache, "api_logs")


def global_backend_init():
    global backend_initialized
    if not backend_initialized:
        # Create server log directory
        if not os.path.exists(api_log_dir):
            os.makedirs(api_log_dir)
        initialize_decode_backend()
        backend_initialized = True


@patch.object(PrefillDecodeBackend, "init_model", new=mock_init_model)
@patch.object(PrefillDecodeBackend, "teardown", new=Mock(return_value=None))
def create_test_server():
    global_backend_init()
    return app


if __name__ == "__main__":
    app = create_test_server()
    app.run(
        port=inference_config.backend_server_port,
        host="0.0.0.0",
    )

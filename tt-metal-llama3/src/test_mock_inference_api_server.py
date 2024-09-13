import os
from time import sleep
from unittest.mock import Mock, patch

import torch

from inference_api_server import (
    global_backend_init,
    app,
    initialize_decode_backend,
)
from inference_config import inference_config

from model_adapters.llama3_1_8b_n150 import Llama3_1_8B_N150
from model_adapters.llama3_1_70b_t3k import Llama3_70B_T3K
from device_manager import DeviceManager, DeviceType

"""
This script runs the flask server and initialize_decode_backend()
with the actual model mocked out.

This allows for rapid testing of the server and backend implementation.
"""

backend_initialized = False
api_log_dir = os.path.join(inference_config.log_cache, "api_logs")


from model_adapters.llama3_1_8b_n150 import MockTtTransformer

mock_return_tensor = lambda tensor, **kwargs: tensor


@patch.object(Llama3_1_8B_N150, "embed_on_device", new=False)
@patch.object(DeviceManager, "get_device_type", return_value=DeviceType.cpu)
@patch("llama3_backend.ttnn.from_torch", new=mock_return_tensor)
@patch(
    "models.demos.wormhole.llama31_8b.tt.llama_common.ttnn.from_torch",
    new=mock_return_tensor,
)
@patch(
    "models.demos.wormhole.llama31_8b.demo.demo_with_prefill.ttnn.from_torch",
    new=mock_return_tensor,
)
@patch("model_adapters.llama3_1_8b_n150.TtTransformer", new=MockTtTransformer)
@patch(
    "model_adapters.llama3_1_8b_n150.cache_attention", new=lambda *args, **kwargs: None
)
@patch("model_adapters.llama3_1_8b_n150.ttnn.untilize", new=mock_return_tensor)
@patch("model_adapters.llama3_1_8b_n150.ttnn.to_torch", new=mock_return_tensor)
@patch("model_adapters.llama3_1_8b_n150.ttnn.linear", new=torch.matmul)
def create_test_server(mock_get_device_type):
    global_backend_init()
    return app


if __name__ == "__main__":
    app = create_test_server()
    app.run(
        port=inference_config.backend_server_port,
        host="0.0.0.0",
    )

import queue
import os
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from inference_api_server import get_user_parameters
from inference_logger import get_logger

from inference_config import inference_config

from model_weights_handler import get_model_weights_and_tt_cache_paths

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
    Tokenizer3,
    ChatFormat,
    Message,
)

from llama3_backend import PrefillDecodeBackend, run_backend


logger = get_logger(__name__)
logger.info(f"importing {__name__}")

backend_logger = logging.getLogger("llama3_backend")
backend_logger.setLevel(logging.DEBUG)


from model_adapters.llama3_1_8b_n150 import Llama3_1_8B_N150
from model_adapters.llama3_1_70b_t3k import Llama3_70B_T3K
from device_manager import DeviceManager, DeviceType
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
def test_llama3_backend_mock(mock_get_device_type):
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 64})
    default_params["max_tokens"] = 128
    min_prompt_tokens = 1
    rag_context = "test rag context"
    for i in range(0, inference_config.model_config.batch_size * 2, 1):
        prompt_q.put(
            (
                f"INIT_ID-{i}",
                "test " * (i + min_prompt_tokens),
                rag_context,
                default_params,
            )
        )
    run_backend(prompt_q, output_q, status_q, verbose=True, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama3_backend_mock()

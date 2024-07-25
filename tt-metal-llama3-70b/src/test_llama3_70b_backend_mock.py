import queue
import os
from pathlib import Path
from time import sleep
from unittest.mock import Mock, patch
import logging

import torch

from inference_api_server import get_user_parameters
from inference_logger import get_logger

from model_weights_handler import get_model_weights_and_tt_cache_paths

# from tt_metal_impl.reference.llama.tokenizer import Tokenizer
from tt_metal_impl.reference.llama.tokenizer3 import Tokenizer3, ChatFormat

from llama3_70b_backend import PrefillDecodeBackend, run_backend


logger = get_logger(__name__)
logger.info(f"importing {__name__}")

test_prompts_outputs = [
    ("This is a test prompt.", "this is test output, much longer now"),
    ("Another prompt.", "also test output"),
]

backend_logger = logging.getLogger("llama2_70b_backend")
backend_logger.setLevel(logging.DEBUG)


class MockModel:

    def __init__(self):
        self.forward_counter = 0

    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        assert len(tokens.shape) == 2
        # mock with repeating previous token
        sleep(1.0 / 500)  # 32 TPS
        # update the new tokens generated to the input id
        # vocab size = tokenizer.nwords
        logits = torch.randn([32, 1, 128256])
        EOT_ID = 128009
        EOS_ID = 128001
        period_ID = 13
        if self.forward_counter % 10 == 0:
            print(f"sending {EOT_ID}")
            logits[:, :, EOT_ID] = 100.0

        self.forward_counter += 1
        return logits


def mock_init_model(self):
    weights_path, tt_cache_path = get_model_weights_and_tt_cache_paths()
    tokenizer_path = weights_path.joinpath("tokenizer.model")
    # vocab_size = 32000
    self.tokenizer = Tokenizer3(model_path=tokenizer_path.as_posix())
    self.formatter = ChatFormat(self.tokenizer)
    self.model = MockModel()


@patch.object(PrefillDecodeBackend, "init_model", new=mock_init_model)
@patch.object(
    PrefillDecodeBackend, "teardown_tt_metal_device", new=Mock(return_value=None)
)
def test_llama2_70b_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 64})
    default_params["max_tokens"] = 128
    # default_params["stop_sequence"] = "."
    for i in range(0, 31, 1):
        prompt_q.put((f"INIT_ID-{i}", "test", default_params))
    run_backend(prompt_q, output_q, status_q, verbose=True, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama2_70b_backend()

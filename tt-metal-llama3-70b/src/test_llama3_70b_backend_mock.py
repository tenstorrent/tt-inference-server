import queue
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch
import logging

import torch

from inference_api_server import get_user_parameters
from inference_logger import get_logger

from model_weights_handler import get_model_weights_and_tt_cache_paths

from models.demos.t3000.llama2_70b.reference.llama.llama.tokenizer3 import (
    Tokenizer3,
    ChatFormat,
    Message,
)

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
        batch, seqlen = tokens.shape
        forward_start = time.time()
        simulated_tps = 10.0
        simulated_duration = 1.0 / simulated_tps
        # update the new tokens generated to the input id
        # vocab_size = tokenizer.nwords
        # logits: [batch, seqlen, vocab_size]
        logits = torch.randn([batch, seqlen, 128256])
        # send a token every period loops
        EOT_ID = 128009
        EOS_ID = 128001
        period = 100
        send_token = EOT_ID
        if self.forward_counter % period == 0:
            print(f"sending {send_token}")
            logits[:, :, send_token] = 100.0
        self.forward_counter += 1
        actual_duration = time.time() - forward_start
        # simulate forward latency
        time.sleep(max(simulated_duration - actual_duration, 0))
        return logits


def mock_init_model(self):
    weights_path, tt_cache_path = get_model_weights_and_tt_cache_paths()
    tokenizer_path = weights_path.joinpath("tokenizer.model")
    # vocab_size = 32000
    self.tokenizer = Tokenizer3(model_path=tokenizer_path.as_posix())
    self.formatter = ChatFormat(self.tokenizer)
    self.model = MockModel()


@patch.object(PrefillDecodeBackend, "init_model", new=mock_init_model)
@patch.object(PrefillDecodeBackend, "teardown", new=Mock(return_value=None))
def test_llama2_70b_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 64})
    default_params["max_tokens"] = 128
    for i in range(0, 32, 1):
        prompt_q.put((f"INIT_ID-{i}", "test " * (i + 1), default_params))
    run_backend(prompt_q, output_q, status_q, verbose=True, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama2_70b_backend()

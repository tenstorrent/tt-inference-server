import queue
import os
from pathlib import Path
from time import sleep
from unittest.mock import Mock, patch

import torch
from mistral_7b_backend import PrefillDecodeBackend, run_backend
from inference_api_server import get_user_parameters
from inference_logger import get_logger
from inference_config import inference_config
from tt_metal_impl.reference.tokenizer import Tokenizer
from tt_metal_impl.tt.model_config import TtModelArgs




logger = get_logger(__name__)
logger.info(f"importing {__name__}")

test_prompts_outputs = [
    ("This is a test prompt.", "this is test output, much longer now"),
    ("Another prompt.", "also test output"),
]
class MockModel:
    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        assert len(tokens.shape) == 2
        # mock with repeating previous token
        sleep(1.0 / 32)  # 32 TPS
        # update the new tokens generated to the input id
        logits = torch.randn([32, 1, 32000])
        return None


def mock_init_model(self):
    model_base_path = Path(inference_config.cache_root) / "mistral-7b-instruct"
    model_args = TtModelArgs(None, model_base_path=model_base_path, instruct=True)
    # vocab_size = 32000
    self.tokenizer = Tokenizer(model_args.tokenizer_path)
    self.model = MockModel()


@patch.object(PrefillDecodeBackend, "init_tt_metal", new=mock_init_model)
@patch.object(
    PrefillDecodeBackend, "teardown_tt_metal_device", new=Mock(return_value=None)
)
def test_mistral_7b_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 64})
    prompt_q.put(("INIT_ID-1", "How do you get to Carnegie Hall?", default_params))
    prompt_q.put(("INIT_ID-2", "Another prompt", default_params))
    run_backend(prompt_q, output_q, status_q, verbose=False, run_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_mistral_7b_backend()

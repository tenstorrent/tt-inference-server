import queue
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch

import torch
from llama3_70b_backend import PrefillDecodeBackend, run_backend
from inference_api_server import get_user_parameters
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")

test_prompts_outputs = [
    ("This is a test prompt.", "this is test output, much longer now"),
    ("Another prompt.", "also test output"),
]


def test_llama2_70b_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 64})
    prompt_q.put(("INIT_ID-1", "How do you get to Carnegie Hall?", default_params))
    prompt_q.put(("INIT_ID-2", "Another prompt", default_params))
    run_backend(prompt_q, output_q, status_q, verbose=False, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama2_70b_backend()

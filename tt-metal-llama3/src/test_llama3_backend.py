import queue
import os
import logging
import time
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from llama3_backend import PrefillDecodeBackend, run_backend
from inference_api_server import get_user_parameters
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")

backend_logger = logging.getLogger("llama3_backend")
backend_logger.setLevel(logging.DEBUG)

def test_llama3_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 128})
    rag_context = ""
    prompt_q.put(
        ("INIT_ID-1", "How do you get to Carnegie Hall?", rag_context, default_params)
    )
    prompt_q.put(("INIT_ID-2", "Another prompt", rag_context, default_params))
    run_backend(prompt_q, output_q, status_q, verbose=True, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama3_backend()

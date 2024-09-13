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

long_prompt = "When considering the rapid advancements in artificial intelligence and its increasing integration into various sectors such as healthcare, finance, education, and even entertainment, what ethical considerations should be at the forefront of these developments to ensure that AI serves humanity in a just and equitable manner? Specifically, how do we address potential biases in AI algorithms that could perpetuate existing societal inequalities, such as racial, gender, or socioeconomic disparities, and what role should governments and regulatory bodies play in enforcing ethical standards in the development and deployment of AI technologies? Additionally, how can we balance innovation with accountability, ensuring that companies developing AI systems are held responsible for the consequences of their creations, both positive and negative? Should there be a global consensus or framework guiding AI ethics, or is it more practical for individual nations to develop their own regulations, considering their unique cultural, legal, and social contexts? Finally, in the face of concerns about job displacement due to AI automation, what strategies should governments and businesses adopt to support workforce transitions and prevent large-scale unemployment, while fostering an environment where humans and AI can work together collaboratively to create new opportunities for economic growth and societal advancement?"
short_prompt = "What is in Austin Texas?"

def test_llama3_backend():
    prompt_q = queue.Queue()
    output_q = queue.Queue()
    status_q = queue.Queue()

    # user_id, prompt, params
    default_params, _ = get_user_parameters({"max_tokens": 512})
    rag_context = ""
    prompt_q.put(
        ("INIT_ID-1", short_prompt, rag_context, default_params)
    )
    prompt_q.put(("INIT_ID-2", "Another prompt", rag_context, default_params))
    run_backend(prompt_q, output_q, status_q, verbose=True, loop_once=True)
    logger.info("finished")


if __name__ == "__main__":
    test_llama3_backend()

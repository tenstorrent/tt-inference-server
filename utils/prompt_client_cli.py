# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import logging
import argparse

import numpy as np
from transformers import AutoTokenizer

from utils.prompt_configs import PromptConfig, BatchConfig, EnvironmentConfig
from utils.prompt_client import PromptClient
from utils.batch_processor import BatchProcessor
from utils.prompt_generation import generate_prompts, generate_images

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_client_args(parser):
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (default: streaming enabled)",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"),
        help="Model name vLLM API server is using.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--num_full_iterations",
        type=int,
        default=1,
        help="Number of full iterations over prompts.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=32,
        help="Batch size for concurrent requests.",
    )
    parser.add_argument(
        "--input_seq_len",
        type=int,
        default=-1,
        help="Length parameter of the input sequence when using random prompts.",
    )
    parser.add_argument(
        "--output_seq_len",
        type=int,
        default=2048,
        help="Make completions all the same pre-defined maximum length for testing.",
    )
    parser.add_argument(
        "--inter_batch_delay",
        type=int,
        default=0,
        help="Seconds of delay between batches.",
    )
    parser.add_argument(
        "--vary_max_concurrent",
        action="store_true",
        help="Randomize normally the batch size for each batch of prompts.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=-1,
        help="Maximum length of generated prompts.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="fixed",
        choices=["fixed", "uniform", "normal"],
        help="Distribution method for selecting random prompt lengths.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        help="The name of the dataset to generate prompts from, or 'random' for random generation.",
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="The model tokenizer to use for vocabulary, truncation, and templating.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Provided jinja2 template to apply to the generated prompts.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the generated prompts in JSONL format.",
    )
    parser.add_argument(
        "--print_prompts",
        action="store_true",
        default=False,
        help="Print generated prompts.",
    )
    parser.add_argument(
        "--skip_trace_precapture",
        action="store_true",
        default=False,
        help="Skips trace precapture phase, use to speed up execution if trace captures have already completed.",
    )
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include randomly generated images with prompts",
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of generated images",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Height of generated images",
    )
    parser.add_argument(
        "--use_chat_api",
        action="store_true",
        help="Use the chat completions API format",
    )
    return parser


def main():
    # set numpy seed for reproducibility
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser = add_client_args(parser)
    args = parser.parse_args()

    assert args.max_prompt_length != -1 or args.input_seq_len != -1, (
        "Either --max_prompt_length or --input_seq_len must be provided."
    )
    if args.max_prompt_length == -1:
        assert args.input_seq_len > 0
        args.max_prompt_length = args.input_seq_len
    elif args.input_seq_len == -1:
        assert args.max_prompt_length > 0
        args.input_seq_len = args.max_prompt_length

    # Create configs from arguments
    prompt_config = PromptConfig(
        input_seq_len=args.input_seq_len,
        max_prompt_length=args.max_prompt_length,
        num_prompts=args.num_prompts,
        distribution=args.distribution,
        dataset=args.dataset,
        tokenizer_model=args.tokenizer_model or args.vllm_model,
        template=args.template,
        save_path=args.save_path,
        print_prompts=args.print_prompts,
        include_images=args.include_images,
        images_per_prompt=args.images_per_prompt,
        image_width=args.image_width,
        image_height=args.image_height,
        use_chat_api=args.use_chat_api,
    )

    output_seq_lens = [args.output_seq_len] * args.num_prompts
    batch_config = BatchConfig(
        max_concurrent=args.max_concurrent,
        output_seq_lens=output_seq_lens,
        num_full_iterations=args.num_full_iterations,
        vary_max_concurrent=args.vary_max_concurrent,
        inter_batch_delay=args.inter_batch_delay,
        stream=not args.no_stream,
        use_chat_api=args.use_chat_api,
    )

    env_config = EnvironmentConfig()

    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained(prompt_config.tokenizer_model)
    prompt_client = PromptClient(env_config)
    batch_processor = BatchProcessor(prompt_client, batch_config)

    # Generate prompts
    prompts, input_seq_lengths = generate_prompts(prompt_config)
    images = generate_images(prompt_config)
    assert len(images) == len(prompts)

    if not args.skip_trace_precapture:
        # pre-capture traces to not include 1st run trace capture time
        image_resolutions = []
        if images[0]:
            image_resolutions = [
                (prompt_config.image_width, prompt_config.image_height)
            ]
        prompt_client.capture_traces(
            context_lens=[(args.input_seq_len, args.output_seq_len)],
            image_resolutions=image_resolutions,
        )

    # Process batches
    logger.info(
        f"Starting batch processing with max_concurrent={batch_config.max_concurrent}"
    )
    responses = batch_processor.process_batch(
        prompts=prompts,
        images=images,
        input_seq_lengths=input_seq_lengths,
        tokenizer=tokenizer,
    )

    logger.info(f"Completed processing {len(responses)} responses")

    # Calculate and log summary statistics
    if responses:
        mean_tpot = np.mean([r["tpot_ms"] for r in responses])
        mean_ttft = np.mean([r["ttft_ms"] for r in responses])
        logger.info(f"Mean TTFT: {mean_ttft:.4f}")
        logger.info(f"Mean TPOT: {mean_tpot:.4f}")
        mean_tps = 1.0 / max(mean_tpot, 1e-6)
        logger.info(f"Mean User TPS: {mean_tps:.4f}")


if __name__ == "__main__":
    main()

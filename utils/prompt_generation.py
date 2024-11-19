# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from pathlib import Path
import logging
import argparse
import json
from pathlib import Path

import torch
from jinja2 import Template
from datasets import load_dataset
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set the random seed for reproducibility
torch.manual_seed(42)


def load_alpaca_eval_dataset_samples(n_samples):
    # Load alpaca_eval dataset with specified number of samples
    alpaca_ds = load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        split=f"eval[:{n_samples}]",
    )
    return alpaca_ds["instruction"]


def tokenize_encode(prompt, tokenizer, max_length, tokenizer_model=None):
    return tokenizer.encode(
        prompt, add_special_tokens=False, truncation=True, max_length=max_length
    )


def tokenize_decode(encoded_prompt, tokenizer, tokenizer_model=None):
    return tokenizer.decode(encoded_prompt)


# Define a function to generate random prompts using a model's vocabulary
def generate_random_prompts(
    num_prompts: int,
    max_length: int,
    distribution: str = "uniform",
    tokenizer_model=None,
):
    """
    Generate random prompts using the model's vocabulary.

    Args:
        num_prompts (int): Number of prompts to generate.
        max_length (int, optional): Maximum length of the generated prompt.
        distribution (str, optional): The distribution method for selecting random prompt lengths ('uniform' or 'max_length').
        tokenizer_model (str, optional): The model to use for generating random tokens.

    Returns:
        List[str]: A list of generated prompts.
    """
    if tokenizer_model is None:
        raise ValueError("Model must be provided when using 'random' as the dataset.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    vocab_size = tokenizer.vocab_size

    # Determine the length of each prompt based on the distribution
    if distribution == "uniform":
        prompt_lengths = torch.randint(1, max_length, (num_prompts,)).tolist()
    elif distribution == "max_length":
        prompt_lengths = [max_length] * num_prompts
    else:
        raise ValueError(
            f"Invalid distribution method: '{distribution}'. Use 'uniform' or 'max_length'."
        )

    # Generate random tokens for all prompts
    token_ids_list = [
        torch.randint(0, vocab_size, (length,)).tolist() for length in prompt_lengths
    ]
    prompts = [
        tokenize_decode(token_ids, tokenizer=tokenizer) for token_ids in token_ids_list
    ]
    return prompts


# Define a function to generate prompts using a specified task and template
def generate_task_prompts(task_name: str, num_prompts: int, max_length: int):
    """
    Generate prompts using lm-evaluation-harness from the specified task.

    Args:
        task_name (str): The name of the task (dataset) to generate prompts from.
        num_prompts (int): Number of prompts to generate.
        max_length (int, optional): Maximum length of the generated prompt.

    Returns:
        List[str]: A list of generated prompts.
    """
    from lm_eval.tasks import get_task_dict

    # Load the specified task
    tasks = get_task_dict([task_name])
    task = tasks.get(task_name)

    if task is None:
        raise ValueError(
            f"Task '{task_name}' not found. Make sure the task name is correct."
        )

    # Get the dataset for the task
    dataset = task.dataset("validation")
    prompts = list(dataset)[:num_prompts]

    # Get the list of available templates for the specified task
    templates = task.templates()

    # Apply templates to the generated prompts
    templated_prompts = []
    for prompt in prompts:
        data_truncated = (
            {"text": prompt[:max_length]}
            if max_length is not None
            else {"text": prompt}
        )
        template = templates[torch.randint(0, len(templates), (1,)).item()]
        templated_prompt = template.apply(data_truncated)
        templated_prompts.append(
            templated_prompt[:max_length]
            if max_length is not None
            else templated_prompt
        )
    prompts = templated_prompts

    return prompts


def apply_jinja_template(prompts, template_path):
    """
    Apply a jinja2 template to the generated prompts.

    Args:
        prompts (List[str]): A list of generated prompts.
        template (str): The jinja2 template to apply to the prompts.

    Returns:
        List[str]: A list of templated prompts.
    """
    assert os.path.exists(template_path), f"Template file '{template_path}' not found."
    with open(template_path, "r") as file:
        template_str = file.read()
    templated_prompts = []
    template = Template(template_str)
    # render single prompt history
    templated_prompts = [
        template.render(chat_history=[{"role": "user", "content": p}]) for p in prompts
    ]

    return templated_prompts


def process_prompts(prompts, max_length, template, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    use_hf_template = True
    if use_hf_template:
        if template is not None:
            assert os.path.exists(
                template_path
            ), f"Template file '{template_path}' not found."
            with open(template_path, "r") as file:
                template_str = file.read()
            template = template_str

        # use automatic template in tokenizer
        tokenizer.apply_chat_template([{"role": "user", "content": prompts[0]}])
        templated_trunc_encoding = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                truncate=True,
                max_length=max_length,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        templated_trunc_prompts = [
            tokenize_decode(p, tokenizer=tokenizer) for p in templated_trunc_encoding
        ]
        breakpoint()
    else:
        # manually apply template to prompts
        logging.info(f"Applying template '{template}' to the generated prompts.")
        assert max_length > 64, "Max length must be greater than 64 for templating."
        template_file = Path(template).resolve()
        templated_prompts = apply_jinja_template(prompts, template_file)
        # truncate prompts to max_length
        prompt_encodings = [
            tokenize_encode(p, tokenizer=tokenizer, max_length=max_length)
            for p in prompts
        ]
        prompt_lens = [len(p) for p in prompt_encodings]
        templated_prompt_encodings = [
            tokenize_encode(p, tokenizer=tokenizer, max_length=max_length)
            for p in templated_prompts
        ]
        templated_prompt_lengths = [len(p) for p in templated_prompt_encodings]
        trunc_lens = [
            max_length - max(l - max_length, 0) for l in templated_prompt_lengths
        ]
        trunc_encodings = [p[:l] for p, l in zip(prompt_encodings, trunc_lens)]
        trunc_prompts = [tokenizer.decode(p) for p in trunc_encodings]
        # re-apply template to truncated prompts
        templated_trunc_prompts = apply_jinja_template(trunc_prompts, template_file)
        post_templatie_truncation_lens = [
            len(tokenize_decode(p, tokenizer=tokenizer))
            for p in templated_trunc_prompts
        ]
        logging.info(
            f"templated_trunc_prompts lengths: {post_templatie_truncation_lens}"
        )

    processed_prompts = templated_trunc_prompts
    breakpoint()
    return processed_prompts


# Main function to handle prompt generation and templating
def generate_prompts(args):
    logging.info(f"generate_prompts args={args}")
    if args.input_seq_len != -1:
        args.max_length = args.input_seq_len

    if args.max_length is not None:
        # determine true max_length
        max_length = args.max_length

    if args.dataset.lower() == "random":
        logger.info("Generating random prompts...")
        prompts = generate_random_prompts(
            args.num_prompts, max_length, args.distribution, args.tokenizer_model
        )
    elif args.dataset is not None:
        logger.info(f"Generating prompts from the '{args.dataset}' dataset...")
        if args.dataset == "alpaca_eval":
            prompts = load_alpaca_eval_dataset_samples(args.n_samples)
        else:
            from lm_eval.tasks import get_task_dict

            prompts = generate_task_prompts(args.task, args.num_prompts, max_length)
    else:
        raise ValueError("Dataset must be provided.")

    prompts = process_prompts(prompts, max_length, args.template, args.tokenizer_model)

    print_prompts = not args.save_path
    # Save prompts to a JSONL file if a save path is provided
    if args.save_path:
        file_path = Path(args.save_path).resolve()
        try:
            with open(file_path, "w") as f:
                for prompt in prompts:
                    json.dump({"prompt": prompt}, f)
                    f.write("\n")
            logger.info(f"Prompts saved to {file_path}")
        except Exception as e:
            logger.info(f"Error saving prompts to {file_path}: {e}")
            print_prompts = True

    if print_prompts:
        logger.info("Generated Prompts:")
        for idx, prompt in enumerate(prompts, 1):
            print(f"{idx}: {prompt}")

    return prompts


def add_prompt_gen_args(parser):
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="The model tokenizer to use for vocabulary, truncation, and templating.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="random",
        help="The name of the dataset to generate prompts from, or 'random' for random token generation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum length of generated prompts.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The task name to apply templates if dataset is 'random'.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="max_length",
        choices=[
            "max_length",
            "uniform",
        ],
        help="Distribution method for selecting random prompt lengths ('uniform' or 'max_length').",
    )
    parser.add_argument(
        "--num_prompts", type=int, default=3, help="Number of prompts to generate."
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
        default="generated_prompts.jsonl",
        help="Path to save the generated prompts in JSONL format.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts.")
    parser = add_prompt_gen_args(parser)
    args = parser.parse_args()
    try:
        generate_prompts(args)
    except ValueError as e:
        print(e)

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


def load_alpaca_eval_dataset_samples(num_prompts):
    # Load alpaca_eval dataset with specified number of samples
    alpaca_ds = load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        split=f"eval[:{num_prompts}]",
    )
    return alpaca_ds["instruction"]


def tokenize_encode(prompt, tokenizer, max_length, tokenizer_model):
    if tokenizer_model == "meta-llama/Llama-3.1-70B-Instruct":
        return tokenizer.encode(
            prompt, add_special_tokens=False, truncation=True, max_length=max_length
        )
    else:
        raise ValueError(f"Unsupported tokenizer model: '{tokenizer_model}'.")


def tokenize_decode(encoded_prompt, tokenizer, tokenizer_model):
    if tokenizer_model == "meta-llama/Llama-3.1-70B-Instruct":
        return tokenizer.decode(encoded_prompt)
    else:
        raise ValueError(f"Unsupported tokenizer model: '{tokenizer_model}'.")


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
        tokenize_decode(token_ids, tokenizer=tokenizer, tokenizer_model=tokenizer_model)
        for token_ids in token_ids_list
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


def template_prompt(prompt, tokenizer, template_file, tokenize=True):
    if template_file is None and len(tokenizer.chat_template) > 0:
        # use default chat template in tokenizer
        # NOTE: apply_chat_template does NOT truncate the prompt with the arguements:
        # truncate=True,
        # max_length=max_length,
        templated_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=tokenize,
            )
            for p in prompts
        ]
        breakpoint()
    elif template_file is not None:
        # assert max_length > 64, "Max length must be greater than 64 for templating."
        template_path = Path(template).resolve()
        templated_prompts = apply_jinja_template(prompts, template_path)
        if tokenize:
            templated_prompts = [
                tokenize_encode(
                    tp,
                    tokenizer=tokenizer,
                    max_length=None,
                    tokenizer_model=tokenizer_model,
                )
                for tp in templated_prompts
            ]
    else:
        raise ValueError("Template must be provided for templating prompts.")

    return templated_prompts


def truncate_template_prompt(prompt, tokenizer, template, max_length):
    # get original prompt token encoding lengths
    prompt_encodings = [
        tokenize_encode(
            p, tokenizer=tokenizer, max_length=None, tokenizer_model=tokenizer_model
        )
        for p in prompts
    ]
    prompt_lens = [len(p) for p in prompt_encodings]

    # Apply template to the prompts and get encoded length of templated prompts
    templated_prompt_encodings = template_prompt(
        prompts, template, tokenizer, tokenize=True
    )
    templated_prompt_lengths = [len(p) for p in templated_prompt_encodings]

    # calculate truncation lengths: difference in length between templated and max_length
    trunc_lens = [
        max_length - max(tpl - max_length, 0) for tpl in templated_prompt_lengths
    ]
    # truncate encoded prompts before templating to max_length
    trunc_encodings = [pe[:tl] for pe, tl in zip(prompt_encodings, trunc_lens)]
    trunc_prompts = [tokenizer.decode(te) for te in trunc_encodings]
    # re-apply template to truncated prompts
    # finally, template the prompts and return prompt text with templating
    templated_trunc_prompts = template_prompt(
        trunc_prompts, template, tokenizer, tokenize=False
    )

    # get length of templated prompts


def process_prompts(prompts, max_length, template, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if template:
        # manually apply template to prompts
        logging.info(f"Applying template to the generated prompts: {template}")
        truncated_prompts = truncate_template_prompt(
            prompt, tokenizer, template, max_length
        )
    else:
        logging.info(f"No template applied to generated prompts.")
        truncated_prompts = [
            tokenize_encode(
                p,
                tokenizer=tokenizer,
                max_length=max_length,
                tokenizer_model=tokenizer_model,
            )
            for p in prompts
        ]
    breakpoint()

    processed_prompts = truncated_prompts
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
            prompts = load_alpaca_eval_dataset_samples(args.num_prompts)
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
        default=128,
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

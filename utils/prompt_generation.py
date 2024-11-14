# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC


import torch
import argparse
import json
from pathlib import Path

# Set the random seed for reproducibility
torch.manual_seed(42)


# Define a function to generate random prompts using a model's vocabulary
def generate_random_prompts(
    num_prompts: int, max_length: int, distribution: str = "uniform", model=None
):
    """
    Generate random prompts using the model's vocabulary.

    Args:
        num_prompts (int): Number of prompts to generate.
        max_length (int, optional): Maximum length of the generated prompt.
        distribution (str, optional): The distribution method for selecting random prompt lengths ('uniform' or 'max_length').
        model (str, optional): The model to use for generating random tokens.

    Returns:
        List[str]: A list of generated prompts.
    """
    if model is None:
        raise ValueError("Model must be provided when using 'random' as the dataset.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
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
    prompts = [tokenizer.decode(token_ids) for token_ids in token_ids_list]

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

    return prompts


# Main function to handle prompt generation and templating
def main(args):
    if args.max_length is not None:
        # determine true max_length
        max_length = args.max_length

    if args.dataset.lower() == "random":
        print("Generating random prompts...")
        prompts = generate_random_prompts(
            args.num_prompts, max_length, args.distribution, args.model
        )
    elif args.dataset is not None:
        print(f"Generating prompts from the '{args.dataset}' dataset...")
        from lm_eval.tasks import get_task_dict

        prompts = generate_task_prompts(args.task, args.num_prompts, max_length)
    else:
        raise ValueError("Dataset must be provided.")

    # If a task name is provided and dataset is 'random', apply task templates to the prompts
    if args.task is not None:
        from lm_eval.tasks import get_task_dict

        tasks = get_task_dict([args.task])
        task = tasks.get(args.task)

        if task is None:
            raise ValueError(
                f"Task '{args.task}' not found. Make sure the task name is correct."
            )

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

    print_prompts = not args.save_path
    # Save prompts to a JSONL file if a save path is provided
    if args.save_path:
        file_path = Path(args.save_path).resolve()
        try:
            with open(file_path, "w") as f:
                for prompt in prompts:
                    json.dump({"prompt": prompt}, f)
                    f.write("\n")
            print(f"Prompts saved to {file_path}")
        except Exception as e:
            print(f"Error saving prompts to {file_path}: {e}")
            print_prompts = True

    if print_prompts:
        print("Generated Prompts:")
        for idx, prompt in enumerate(prompts, 1):
            print(f"{idx}: {prompt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts using lm-evaluation-harness."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The model tokenizer to use for vocabulary, truncation, and templating.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to generate prompts from, or 'random' for random token generation.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
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
        default="uniform",
        choices=["uniform", "max_length"],
        help="Distribution method for selecting random prompt lengths ('uniform' or 'max_length').",
    )
    parser.add_argument(
        "--num_prompts", type=int, default=10, help="Number of prompts to generate."
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

    args = parser.parse_args()
    try:
        main(args)
    except ValueError as e:
        print(e)

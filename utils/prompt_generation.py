# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
from pathlib import Path
import logging
import json
import io
import base64
from datetime import date

import torch
import numpy as np
from PIL import Image
from jinja2 import Template
from datasets import load_dataset
from transformers import AutoTokenizer

from utils.prompt_configs import PromptConfig


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# set torch seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def generate_images(prompt_config: PromptConfig):
    images = []

    for i in range(prompt_config.num_prompts):
        if prompt_config.include_images:
            imgs = [
                generate_random_images(
                    width=prompt_config.image_width,
                    height=prompt_config.image_height,
                )
                for i in range(prompt_config.images_per_prompt)
            ]
        else:
            imgs = []
        images.append(imgs)

    return images


def generate_random_images(
    width=256, height=256, base64_encoded=True, img_format="PNG"
):
    """
    Generate a random RGB image and return it as a base64 string.

    Args:
        width (int): Width of the image
        height (int): Height of the image

    Returns:
        str: Base64 encoded image data
    """
    assert width > 0
    assert height > 0
    # Generate random RGB data
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Create PIL Image
    image = Image.fromarray(data, "RGB")

    buffered = io.BytesIO()
    image.save(buffered, format=img_format)
    if base64_encoded:
        img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img_data = buffered.getvalue()

    return img_data


def load_alpaca_eval_dataset_samples(num_prompts):
    # Load alpaca_eval dataset with specified number of samples
    alpaca_ds = load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        split=f"eval[:{num_prompts}]",
        trust_remote_code=True,
    )
    return alpaca_ds["instruction"]


def tokenize_encode(prompt, tokenizer, max_length, tokenizer_model):
    return tokenizer.encode(
        prompt, add_special_tokens=False, truncation=True, max_length=max_length
    )


def tokenize_decode(encoded_prompt, tokenizer, tokenizer_model):
    return tokenizer.decode(encoded_prompt)


# Define a function to generate random prompts using a model's vocabulary
def generate_random_prompts(
    num_prompts: int,
    max_length: int,
    input_seq_len: int,
    distribution: str,
    tokenizer_model=None,
    text_content=True,
    image_content=False,
):
    """
    Generate random prompts using the model's vocabulary.

    Args:
        num_prompts (int): Number of prompts to generate.
        max_length (int, optional): Maximum length of the generated prompt.
        input_seq_len (int, optional): Length parameter of the input sequence.
        distribution (str, optional): The distribution method for selecting random prompt lengths ('fixed', 'uniform', or 'normal).
        tokenizer_model (str, optional): The model to use for generating random tokens.

    Returns:
        List[str]: A list of generated prompts.
    """
    if tokenizer_model is None:
        raise ValueError("Model must be provided when using 'random' as the dataset.")

    kwargs = {}
    if tokenizer_model.startswith("mistralai"):
        kwargs["use_fast"] = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **kwargs)
    vocab_size = tokenizer.vocab_size

    # Determine the length of each prompt based on the distribution
    if distribution == "fixed":
        prompt_lengths = [input_seq_len] * num_prompts
    elif distribution == "uniform":
        prompt_lengths = torch.randint(1, input_seq_len, (num_prompts,)).tolist()
    elif distribution == "normal":
        prompt_lengths = (
            torch.normal(mean=input_seq_len, std=input_seq_len / 4, size=(num_prompts,))
            .clamp(1, max_length)
            .round()
            .to(torch.int32)
            .tolist()
        )
    else:
        raise ValueError(
            f"Invalid distribution method: '{distribution}'. Must be 'fixed', 'uniform', or 'normal'."
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
        template.render(
            messages=[{"role": "user", "content": p}],
            bos_token="<|begin_of_text|>",
            add_generation_prompt=True,
            date_string=date.today().isoformat(),
        )
        for p in prompts
    ]

    return templated_prompts


def template_prompt(prompts, tokenizer, template, tokenize, tokenizer_model):
    if template == "chat_template" and len(tokenizer.chat_template) > 0:
        # use default chat template in tokenizer
        # NOTE: apply_chat_template does NOT truncate the prompt with the arguements:
        # truncate=True,
        # max_length=max_length,
        date_str = date.today().isoformat()
        templated_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=tokenize,
                date_string=date_str,
            )
            for p in prompts
        ]
    elif template is not None:
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


def truncate_template_prompt(prompts, tokenizer, template, max_length, tokenizer_model):
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
        prompts,
        template=template,
        tokenizer=tokenizer,
        tokenize=True,
        tokenizer_model=tokenizer_model,
    )
    templated_prompt_lengths = [len(p) for p in templated_prompt_encodings]

    # calculate truncation lengths: difference in length between templated and max_length
    trunc_diffs = [max(tpl - max_length, 0) for tpl in templated_prompt_lengths]
    trunc_lens = [max(pl - td, 0) for pl, td in zip(prompt_lens, trunc_diffs)]
    # truncate encoded prompts before templating to max_length
    trunc_encodings = [pe[:tl] for pe, tl in zip(prompt_encodings, trunc_lens)]
    trunc_prompts = [tokenizer.decode(te) for te in trunc_encodings]
    # re-apply template to truncated prompts
    # finally, template the prompts and return prompt text with templating
    templated_trunc_prompts = template_prompt(
        trunc_prompts,
        template=template,
        tokenizer=tokenizer,
        tokenize=False,
        tokenizer_model=tokenizer_model,
    )
    return templated_trunc_prompts


def process_prompts(prompts, max_length, template, tokenizer_model):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if template:
        # manually apply template to prompts
        logging.info(f"Applying template to the generated prompts: {template}")
        truncated_prompts = truncate_template_prompt(
            prompts, tokenizer, template, max_length, tokenizer_model
        )
    else:
        logging.info("No template applied to generated prompts.")
        truncated_prompts = [
            tokenizer.decode(
                tokenize_encode(
                    p,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    tokenizer_model=tokenizer_model,
                )
            )
            for p in prompts
        ]

    processed_prompts = truncated_prompts

    processed_lengths = [
        len(
            tokenize_encode(
                p, tokenizer=tokenizer, max_length=None, tokenizer_model=tokenizer_model
            )
        )
        for p in processed_prompts
    ]
    return processed_prompts, processed_lengths


# Main function to handle prompt generation and templating
def generate_prompts(prompt_config: PromptConfig):
    logging.info(f"generate_prompts args={prompt_config}")
    # vLLM appears to add extra token on receipt of prompt
    # TODO: verify if this is bos token or something else
    prompt_config.max_prompt_length = prompt_config.max_prompt_length - 1
    if prompt_config.input_seq_len == -1:
        prompt_config.input_seq_len = prompt_config.max_prompt_length
    else:
        prompt_config.input_seq_len = prompt_config.input_seq_len - 1

    if prompt_config.dataset.lower() == "random":
        # default case
        logger.info("Generating random prompts...")
        # -1 is for the extra token added by vLLM
        assert prompt_config.input_seq_len > -1, (
            "input_seq_len must be set for random prompts."
        )
        assert prompt_config.max_prompt_length > -1, (
            "max_length must be set for random prompts."
        )
        prompts = generate_random_prompts(
            prompt_config.num_prompts,
            prompt_config.max_prompt_length,
            prompt_config.input_seq_len,
            prompt_config.distribution,
            prompt_config.tokenizer_model,
        )
    elif prompt_config.dataset is not None:
        assert prompt_config.max_prompt_length > -1, (
            "max_length must be set for datasets prompts."
        )
        logger.info(f"Generating prompts from the '{prompt_config.dataset}' dataset...")
        if prompt_config.dataset == "alpaca_eval":
            prompts = load_alpaca_eval_dataset_samples(prompt_config.num_prompts)
    else:
        raise ValueError("Dataset must be provided.")

    prompts, prompt_lengths = process_prompts(
        prompts,
        prompt_config.max_prompt_length,
        prompt_config.template,
        prompt_config.tokenizer_model,
    )
    # Add 1 to prompt lengths to account for the extra token added by vLLM
    prompt_lengths = [pl + 1 for pl in prompt_lengths]

    print_prompts = (prompt_config.num_prompts < 5) and prompt_config.print_prompts
    # Save prompts to a JSONL file if a save path is provided
    if prompt_config.save_path:
        file_path = Path(prompt_config.save_path).resolve()
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
            print(f"prompt {idx}:\n{prompt}")

    return prompts, prompt_lengths

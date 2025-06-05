# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import pytest
from argparse import Namespace
import json
import tempfile
import os

from utils.prompt_generation import generate_prompts


@pytest.fixture
def tokenizer_model():
    return "meta-llama/Llama-3.1-70B-Instruct"


@pytest.fixture
def temp_save_path():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp:
        temp_path = temp.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def template_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".j2", delete=False) as temp:
        temp.write(
            """{% for message in messages %}{% if message.role == 'user' %}User: {{ message.content }}{% endif %}{% endfor %}"""
        )
        temp_path = temp.name
    yield temp_path
    os.unlink(temp_path)


def test_random_prompts_fixed_distribution(tokenizer_model):
    args = Namespace(
        num_prompts=3,
        input_seq_len=10,
        max_prompt_length=10,
        dataset="random",
        distribution="fixed",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 3
    assert all(isinstance(p, str) for p in prompts)
    assert all(pl <= 10 for pl in lengths)


def test_random_prompts_uniform_distribution(tokenizer_model):
    args = Namespace(
        num_prompts=3,
        input_seq_len=20,
        max_prompt_length=20,
        dataset="random",
        distribution="uniform",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 3
    assert all(isinstance(p, str) for p in prompts)
    assert all(pl <= 20 for pl in lengths)


def test_random_prompts_normal_distribution(tokenizer_model):
    args = Namespace(
        num_prompts=3,
        input_seq_len=15,
        max_prompt_length=15,
        dataset="random",
        distribution="normal",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 3
    assert all(isinstance(p, str) for p in prompts)
    assert all(pl <= 15 for pl in lengths)


def test_alpaca_eval_dataset(tokenizer_model):
    args = Namespace(
        num_prompts=2,
        input_seq_len=-1,
        max_prompt_length=50,
        dataset="alpaca_eval",
        distribution="fixed",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 2
    assert all(isinstance(p, str) for p in prompts)
    assert all(pl <= 51 for pl in lengths)  # 50 + 1 for vLLM token


def test_with_template(tokenizer_model, template_path):
    args = Namespace(
        num_prompts=2,
        input_seq_len=20,
        max_prompt_length=30,
        dataset="random",
        distribution="fixed",
        template=template_path,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 2
    assert all(isinstance(p, str) for p in prompts)
    assert all(pl <= 31 for pl in lengths)  # 30 + 1 for vLLM token
    assert all("User:" in p for p in prompts)


def test_save_to_file(tokenizer_model, temp_save_path):
    args = Namespace(
        num_prompts=2,
        input_seq_len=10,
        max_prompt_length=10,
        dataset="random",
        distribution="fixed",
        template=None,
        save_path=temp_save_path,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)

    # Verify file contents
    with open(temp_save_path, "r") as f:
        saved_prompts = [json.loads(line)["prompt"] for line in f]
    assert saved_prompts == prompts


@pytest.mark.parametrize(
    "invalid_args",
    [
        # Missing tokenizer model for random dataset
        pytest.param(
            Namespace(
                num_prompts=2,
                input_seq_len=10,
                max_prompt_length=10,
                dataset="random",
                distribution="fixed",
                template=None,
                save_path=None,
                tokenizer_model=None,
                task=None,
            ),
            id="missing_tokenizer",
        ),
        # Invalid distribution
        pytest.param(
            Namespace(
                num_prompts=2,
                input_seq_len=10,
                max_prompt_length=10,
                dataset="random",
                distribution="invalid",
                template=None,
                save_path=None,
                tokenizer_model="meta-llama/Llama-3.1-70B-Instruct",
                task=None,
            ),
            id="invalid_distribution",
        ),
    ],
)
def test_invalid_arguments(invalid_args):
    with pytest.raises(ValueError):
        generate_prompts(invalid_args)


def test_max_length_default_to_input_seq_len(tokenizer_model):
    args = Namespace(
        num_prompts=2,
        input_seq_len=15,
        max_prompt_length=15,
        dataset="random",
        distribution="fixed",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 2
    assert all(pl <= 15 for pl in lengths)


def test_very_short_sequence(tokenizer_model):
    args = Namespace(
        num_prompts=2,
        input_seq_len=1,
        max_prompt_length=1,
        dataset="random",
        distribution="fixed",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 2
    assert all(pl <= 2 for pl in lengths)  # 1 + 1 for vLLM token


def test_zero_prompts(tokenizer_model):
    args = Namespace(
        num_prompts=0,
        input_seq_len=10,
        max_prompt_length=10,
        dataset="random",
        distribution="fixed",
        template=None,
        save_path=None,
        tokenizer_model=tokenizer_model,
        task=None,
    )
    prompts, lengths = generate_prompts(args)
    assert len(prompts) == 0
    assert len(lengths) == 0

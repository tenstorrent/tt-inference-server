# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# The same strings as tt-train/tests/python/experimental/test_sft_templates.py: the
# prompts ttml trains on, which inference has to send byte for byte.
from utils.dataset_loaders.alpaca.alpaca_utils import alpaca_prompt

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nTranslate to French.\n\n"
    "### Input:\nGood morning\n\n"
    "### Response:\n"
)

ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\nGive three tips for staying healthy.\n\n"
    "### Response:\n"
)


def test_alpaca_prompt_with_input():
    assert alpaca_prompt("Translate to French.", "Good morning") == ALPACA_WITH_INPUT


def test_alpaca_prompt_without_input():
    assert alpaca_prompt("Give three tips for staying healthy.") == ALPACA_NO_INPUT
    assert (
        alpaca_prompt("Give three tips for staying healthy.", " \n") == ALPACA_NO_INPUT
    )

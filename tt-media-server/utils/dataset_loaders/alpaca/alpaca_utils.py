# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# The prompt adapters trained by ttml on Alpaca or custom datasets saw, byte for byte
# (tt-train/sources/ttml/ttml/experimental/datasets/utils/sft/templates/alpaca.py).
PROMPT_INTRO = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request."
)

PROMPT_INTRO_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return (
            f"{PROMPT_INTRO}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        f"{PROMPT_INTRO_NO_INPUT}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    )


DATASET_PATH = "tatsu-lab/alpaca"
TRAIN_SPLIT_RATIO = 0.98
SEED = 23

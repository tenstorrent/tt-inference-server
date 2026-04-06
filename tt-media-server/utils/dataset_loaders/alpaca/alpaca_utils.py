# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from inspect import cleandoc
from string import Template

PROMPT_INTRO = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

PROMPT_TEMPLATE = Template(
    cleandoc(
        f"""
        {PROMPT_INTRO}

        ### Instruction:
        $instruction

        ### Input:
        $input

        ### Response:
        """
    )
)

PROMPT_TEMPLATE_NO_INPUT = Template(
    cleandoc(
        f"""
        {PROMPT_INTRO}

        ### Instruction:
        $instruction

        ### Response:
        """
    )
)

DATASET_PATH = "tatsu-lab/alpaca"
TRAIN_SPLIT_RATIO = 0.98
SEED = 23

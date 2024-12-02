# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
from typing import List, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace

from utils.prompt_generation import generate_prompts


class DataReader:
    """
    Reads sample data from a dataset file using an iterator.
    """
    def __init__(self) -> None:
        # Create custom args
        self.args = SimpleNamespace(
            tokenizer_model="meta-llama/Llama-3.1-70B-Instruct",
            dataset="random",
            max_prompt_length=128,
            input_seq_len=128,
            num_prompts=32,
            distribution="fixed",
            template=None,
            save_path=None,
        )

        # Generate prompts
        self.prompts, self.prompt_lengths = generate_prompts(self.args)

        # Initialize data iterator
        self.data = iter(self.prompts)

    def __iter__(self):
        """Allow DataReader to be used as an iterator."""
        return self

    def __next__(self) -> Union[str, List[str]]:
        """Return the next prompt from the dataset. Reset iterator when exhausted."""
        try:
            return next(self.data)
        except StopIteration:
            # Reset the iterator if all data has been consumed
            self.data = iter(self.prompts)
            return next(self.data)

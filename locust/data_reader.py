# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import random
from typing import List, Union

from datasets import load_dataset


class DataReader:
    """
    Reads sample data from a dataset file using an iterator. Data can be optionally shuffled once.
    """
    def __init__(self, with_shuffle: bool = False) -> None:
        # Load dataset and extract prompts
        self.content = load_dataset("fka/awesome-chatgpt-prompts")["train"]["prompt"]

        # Optionally shuffle content
        if with_shuffle:
            random.shuffle(self.content)

        # Initialize data iterator
        self.data = iter(self.content)

    def __iter__(self):
        """Allow DataReader to be used as an iterator."""
        return self

    def __next__(self) -> Union[str, List[str]]:
        """Return the next prompt from the dataset. Reset iterator when exhausted."""
        try:
            return next(self.data)
        except StopIteration:
            # Reset the iterator if all data has been consumed
            self.data = iter(self.content)
            return next(self.data)

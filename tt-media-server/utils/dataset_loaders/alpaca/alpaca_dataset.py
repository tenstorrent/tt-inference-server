# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from utils.dataset_loaders.alpaca.alpaca_utils import (
    DATASET_PATH,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_NO_INPUT,
    SEED,
    TRAIN_SPLIT_RATIO,
)
from utils.dataset_loaders.base_dataset import BaseDataset


class AlpacaDataset(BaseDataset):
    # Alpaca only has a train split; we create validation from it.
    # Shared across instances to avoid reloading.
    _shared_dataset = None

    def __init__(
        self,
        model_name: str,
        max_sequence_length: int,
        split: str = "train",
        collate_fn=None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self.split = split
        self.collate_fn = collate_fn
        self.max_length = max_sequence_length

        self._prepare_dataset()

    def _tokenize_function(self, example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text.strip():
            prompt = PROMPT_TEMPLATE.substitute(
                instruction=instruction, input=input_text
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_INPUT.substitute(instruction=instruction)

        full_text = prompt + output

        encoding = self.tokenizer(
            full_text, truncation=False, padding=False, return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(
            prompt, truncation=False, padding=False, return_tensors="pt"
        )
        prompt_input_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_len = prompt_input_ids.size(0)
        labels[:prompt_len] = -100

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["full_text"] = full_text
        example["len"] = input_ids.size(0)

        return example

    def _prepare_dataset(self):
        if AlpacaDataset._shared_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize_function)
            filtered_dataset = tokenized_dataset.filter(
                lambda x: x["len"] <= self.max_length
            )
            filtered_dataset = filtered_dataset.remove_columns(
                [
                    col
                    for col in filtered_dataset.column_names
                    if col not in self.required_columns
                ]
            )
            AlpacaDataset._shared_dataset = filtered_dataset.shuffle(seed=SEED)

        full_dataset = AlpacaDataset._shared_dataset
        n = len(full_dataset)
        train_end = int(TRAIN_SPLIT_RATIO * n)

        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_end, n))
        else:
            raise ValueError(
                f"Invalid split '{self.split}' for AlpacaDataset. "
                f"Only 'train' and 'validation' are supported."
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def get_dataloader(self, batch_size: int) -> DataLoader:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
        )

        if self.collate_fn is not None:

            def total_collate_fn(batch):
                return self.collate_fn(data_collator(batch))
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch

from blacksmith.datasets.torch.sst2.sst2_utils import (
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
    LBL2VALUE,
    DATASET_BENCHMARK,
    DATASET_NAME,
)
from blacksmith.tools.templates.configs import TrainingConfig
from blacksmith.datasets.torch.torch_dataset import BaseDataset


class SSTDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig
            split: Dataset split to use ("train", "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    def _tokenize_function(self, example):
        prompt = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
        full_text = prompt + response

        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
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
        raw_dataset = load_dataset(DATASET_BENCHMARK, DATASET_NAME, split=self.split)

        tokenized_dataset = raw_dataset.map(self._tokenize_function)
        self.full_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.config.max_length)
        self.dataset = self.full_dataset.remove_columns(
            [col for col in self.full_dataset.column_names if col not in self.required_columns]
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

    def get_dataloader(self) -> DataLoader:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding="max_length", max_length=self.config.max_length
        )

        if self.collate_fn is not None:
            total_collate_fn = lambda batch: self.collate_fn(data_collator(batch))
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )

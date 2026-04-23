# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from utils.dataset_loaders.base_dataset import BaseDataset
from utils.dataset_loaders.sst2.sst2_utils import (
    DATASET_BENCHMARK,
    DATASET_NAME,
    LBL2VALUE,
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
)


class SSTDataset(BaseDataset):
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
        user_content = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        # full_text = full_text.rstrip("\n") + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text, truncation=False, padding=False, return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(
            prompt, truncation=False, padding=False, return_tensors="pt",
            add_special_tokens=False,
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
        raw_dataset = load_dataset(DATASET_BENCHMARK, DATASET_NAME, split=self.split)

        tokenized_dataset = raw_dataset.map(self._tokenize_function, load_from_cache_file=False)
        self.full_dataset = tokenized_dataset.filter(
            lambda example: example["len"] <= self.max_length, load_from_cache_file=False
        )
        self.dataset = self.full_dataset.remove_columns(
            [
                col
                for col in self.full_dataset.column_names
                if col not in self.required_columns
            ]
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
            tokenizer=self.tokenizer, padding="max_length", max_length=self.max_length
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

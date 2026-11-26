# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""OpenAI MRCR (openai/mrcr), 8 needles, prompts up to 128K tokens.

Each MRCR prompt is a whole multi-turn conversation (a JSON list of chat
messages). lm-eval's chat path sends one user message per doc, so the doc text
is that JSON behind CHAT_MESSAGES_SENTINEL, and the eval launcher's
lm_eval_request_overrides.py (--expand-chat-messages) swaps it back into the
request's ``messages``. Binning and grading follow the dataset card: o200k_base
tokens of prompt + answer, and difflib's SequenceMatcher ratio on responses
that start with the required random prefix.
"""

import json
import random
from difflib import SequenceMatcher
from itertools import zip_longest

import datasets

# Must match llm_module/lm_eval_request_overrides.py.
CHAT_MESSAGES_SENTINEL = "__TT_CHAT_MESSAGES_JSON__"

# Dataset-card bins (prompt + answer tokens): [4096, 8192], (8192, 16384], ...
BIN_UPPER_BOUNDS = [8192, 16384, 32768, 65536, 131072]
MAX_TOKENS = BIN_UPPER_BOUNDS[-1]
# o200k_base averages well under 5 chars per token on this text; skipping the
# longer prompts by size avoids tokenizing the 256K-1M bins at all.
MAX_CHARS_PREFILTER = MAX_TOKENS * 5
SHUFFLE_SEED = 1234


def _n_tokens(enc, doc) -> int:
    messages = json.loads(doc["prompt"])
    return sum(len(enc.encode(m["content"])) for m in messages) + len(
        enc.encode(doc["answer"])
    )


def _bin_index(n_tokens: int) -> int:
    for i, upper in enumerate(BIN_UPPER_BOUNDS):
        if n_tokens <= upper:
            return i
    return len(BIN_UPPER_BOUNDS)


def process_docs(dataset):
    """Keep the 8-needle samples in the bins up to 128K tokens (100 per bin) and
    interleave the bins round-robin, so a --limit prefix is balanced across
    context lengths (the card's "128K" is the average over these bins)."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    bins = [[] for _ in BIN_UPPER_BOUNDS]
    for row in dataset:
        if row["n_needles"] != 8 or row["n_chars"] > MAX_CHARS_PREFILTER:
            continue
        idx = _bin_index(_n_tokens(enc, row))
        if idx < len(bins):
            bins[idx].append(dict(row, n_tokens_bin=BIN_UPPER_BOUNDS[idx]))
    rng = random.Random(SHUFFLE_SEED)
    for rows in bins:
        rng.shuffle(rows)
    ordered = [row for group in zip_longest(*bins) for row in group if row]
    return datasets.Dataset.from_list(ordered)


def doc_to_text(doc) -> str:
    return CHAT_MESSAGES_SENTINEL + doc["prompt"]


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def process_results(doc, results):
    return {
        "mrcr_score": grade(results[0], doc["answer"], doc["random_string_to_prepend"])
    }

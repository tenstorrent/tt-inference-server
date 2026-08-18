# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Phase 1 golden capture: reference embeddings from the Python BGE runner.

Runs a fixed prompt set through BGELargeENRunner and writes a JSON file used
later to validate the C++ server (same prompts must yield the same vectors).

Each prompt is embedded twice: alone (batch of 1) and inside a full batch.
Batch composition changes padding, which shifts bfloat8_b numerics slightly,
so both references are stored to avoid chasing phantom mismatches later.

Usage (tt-metal worktree venv active):

    TT_METAL_HOME=/localdev/jzivanovic/tt-metal-65718bb \
    PYTHONPATH="$TT_METAL_HOME:$PWD" \
    python scripts/capture_embedding_golden.py
"""

import os
import sys

# Must happen before any tt-media-server import (Settings reads env at import).
os.environ.setdefault("MODEL", "bge-large-en-v1.5")
os.environ.setdefault("DEVICE", "n150")
os.environ.setdefault("DEVICE_IDS", "(0)")

import asyncio  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

from domain.text_embedding_request import TextEmbeddingRequest  # noqa: E402
from tt_model_runners.embedding_runner import BGELargeENRunner  # noqa: E402

MODEL_ID = "BAAI/bge-large-en-v1.5"
MAX_MODEL_LEN = 384
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "goldens", "bge_large_en_v1_5_n150.json"
)

# The tt-metal BGE demo logs its internal device-vs-CPU check during warmup;
# recorded as provenance for the golden vectors.
_warmup_pcc: list[float] = []


def _install_pcc_sink() -> None:
    try:
        from loguru import logger as loguru_logger

        def sink(message):
            m = re.search(r"PCC=([0-9.eE+-]+)", str(message))
            if m:
                _warmup_pcc.append(float(m.group(1)))

        loguru_logger.add(sink, level="INFO")
    except Exception:
        pass  # provenance only; capture must not fail because of it


def _repeat_to_token_count(hf_tokenizer, target: int) -> str:
    """Build a prompt whose untruncated token count is exactly `target`."""
    n = target  # first guess; BERT adds [CLS]/[SEP]
    while True:
        text = " ".join(["hello"] * n)
        count = len(hf_tokenizer(text, truncation=False)["input_ids"])
        if count == target:
            return text
        n += target - count


def build_prompts(hf_tokenizer) -> list[dict]:
    prompts = [
        {"id": "single_token", "text": "cat"},
        {
            "id": "short_sentence",
            "text": "The quick brown fox jumps over the lazy dog.",
        },
        {
            "id": "exact_384",
            "text": _repeat_to_token_count(hf_tokenizer, MAX_MODEL_LEN),
        },
        {"id": "over_limit_600", "text": " ".join(["hello"] * 600)},
        {
            "id": "non_ascii",
            "text": "Beograd je glavni grad Srbije. Летње вече на Дунаву. 東京は日本の首都です。",
        },
        {"id": "question", "text": "What is the capital of France?"},
        {
            "id": "passage",
            "text": "Paris is the capital and most populous city of France.",
        },
        {"id": "numbers", "text": "In 2024, revenue grew by 17.5% to $3.2 billion."},
        {
            "id": "code_like",
            "text": "def embed(text): return model.forward(tokenize(text))",
        },
        {
            "id": "negation",
            "text": "The delivery did not arrive on time and the customer was unhappy.",
        },
    ]
    for p in prompts:
        p["untruncated_tokens"] = len(
            hf_tokenizer(p["text"], truncation=False)["input_ids"]
        )
    return prompts


def embed(runner, texts: list[str]) -> list:
    responses = runner.run(
        [TextEmbeddingRequest(model=MODEL_ID, input=t) for t in texts]
    )
    return [(list(r.embedding), int(r.total_tokens)) for r in responses]


def main() -> int:
    _install_pcc_sink()

    runner = BGELargeENRunner("0")
    hf_tokenizer = runner.tokenizer.tokenizer  # underlying HF AutoTokenizer
    prompts = build_prompts(hf_tokenizer)

    print(f"[golden] {len(prompts)} prompts prepared", flush=True)
    runner.set_device()
    if not asyncio.run(runner.warmup()):
        print("[golden] warmup FAILED")
        return 1

    batch_size = min(8, int(getattr(runner, "max_num_seqs", 8)))

    # Pass 1: each prompt alone.
    for p in prompts:
        [(vec, tokens)] = embed(runner, [p["text"]])
        p["embedding_single"] = vec
        p["token_count"] = tokens

    # Pass 2: full batches, order preserved.
    for start in range(0, len(prompts), batch_size):
        group = prompts[start : start + batch_size]
        results = embed(runner, [p["text"] for p in group])
        for p, (vec, _tokens) in zip(group, results):
            p["embedding_batched"] = vec
            p["batch_group"] = start // batch_size

    # Determinism: same input alone twice must match exactly.
    [(vec_again, _)] = embed(runner, [prompts[0]["text"]])
    max_diff = max(
        abs(a - b) for a, b in zip(prompts[0]["embedding_single"], vec_again)
    )

    runner.close_device()

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    commit = subprocess.run(
        ["git", "-C", tt_metal_home, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    golden = {
        "metadata": {
            "model": MODEL_ID,
            "device": "n150",
            "tt_metal_commit": commit,
            "max_model_len": MAX_MODEL_LEN,
            "batch_size": batch_size,
            "embedding_dim": len(prompts[0]["embedding_single"]),
            "warmup_pcc": _warmup_pcc[0] if _warmup_pcc else None,
            "determinism_max_abs_diff": max_diff,
            "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "prompts": prompts,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(golden, f)

    print(f"[golden] wrote {OUTPUT_PATH}")
    print(
        f"[golden] dim={golden['metadata']['embedding_dim']} "
        f"warmup_pcc={golden['metadata']['warmup_pcc']} "
        f"determinism_max_abs_diff={max_diff}"
    )
    for p in prompts:
        print(
            f"[golden]   {p['id']}: untruncated={p['untruncated_tokens']} "
            f"served_tokens={p['token_count']}"
        )
    print("[golden] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

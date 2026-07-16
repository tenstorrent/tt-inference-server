# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Generate the in-tree mooncake JSONL trace for the ``highcache_50k`` preset.

The output JSONL is consumed by AIPerf via
``--custom-dataset-type mooncake-trace --input-file <out>``. Each line is a
single request with the schema documented at
https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md

This models the customer's trillion-scale traffic shape:

    50K shared (cacheable) prefix + 5K new ISL + 500 OSL.

Every session re-sends an identical ~50K-token system prefix (the SHARED_ROOT
hash blocks), then appends ~5K tokens of per-request unique input. Because the
50K root is byte-identical across all sessions, AIPerf's radix grouping marks
those blocks as cache-shared, so once the cache is warm the per-session KV-cache
hit-rate is ~= 50000 / (50000 + 5000) = ~90.9% (>= the customer's 90% target).

Why a trace (vs the synthetic ``shared_system`` scenario): the trace path goes
through AIPerf's prefix-synthesis pipeline (Use Case 3/4 in the AIPerf docs),
which lets the same shape be scaled with the ``--synthesis-*`` multipliers and
fed through ``--goodput`` for SLA compliance, while still exercising a realistic
radix-tree reuse pattern rather than a single shared system message.

Security note: this script uses ``random.Random(SEED)`` intentionally and is
not security-sensitive. The output is a fixture file committed to the repo for
byte-stable reproducibility; switching to ``secrets``/``os.urandom`` would
produce a different trace on every regeneration and defeat the purpose of the
file. No value produced here is used as an identifier, token, nonce, credential,
or in any access-control decision -- they are benchmark-input integers (input
length, output length, block hashes, inter-arrival deltas). The ``# nosec
B311`` markers below acknowledge this.

Usage:
    python tt-inference-server-v2/llm_module/prefix_cache/sample_traces/generate_customer_mooncake.py
"""

import json
import random  # nosec B311  # noqa: S311  -- see module docstring
from pathlib import Path

SEED = 50_000
BLOCK_SIZE = 512

# Customer shape. We keep both ISL components as exact multiples of BLOCK_SIZE so
# that ``len(hash_ids) == ceil(input_length / BLOCK_SIZE)`` holds exactly (AIPerf
# 0.5 rejects traces where the hash-id count is incompatible with input_length).
SHARED_PREFIX_TOKENS = 50_176  # 98 blocks * 512 ≈ 50K shared (cacheable) prefix
UNIQUE_TOKENS = 5_120  # 10 blocks * 512 ≈ 5K new per-request input
OUTPUT_TOKENS = 500  # 500 OSL

NUM_SESSIONS = 32  # one concurrency wave (one SC16 decode unit)
TURNS_PER_SESSION = 8  # 32 * 8 = 256 requests -> usable TTFT percentiles

SHARED_PREFIX_BLOCKS = SHARED_PREFIX_TOKENS // BLOCK_SIZE  # 98
UNIQUE_BLOCKS = UNIQUE_TOKENS // BLOCK_SIZE  # 10
INPUT_LENGTH = SHARED_PREFIX_TOKENS + UNIQUE_TOKENS  # 55,296 (108 blocks)

OUT_PATH = Path(__file__).with_name("customer_mooncake.jsonl")

# Stable, non-colliding hash ids for the single shared 50K root. Every session
# reuses these exact blocks, which is what drives the >= 90% steady-state reuse.
SHARED_ROOT = list(range(1, SHARED_PREFIX_BLOCKS + 1))


def main() -> None:
    rng = random.Random(SEED)  # nosec B311  # noqa: S311  -- deterministic fixture

    requests = []
    timestamp_ms = 0
    next_unique_block = 1_000_000
    for session in range(NUM_SESSIONS):
        for turn in range(TURNS_PER_SESSION):
            # Shared 50K root (cache-shared across all sessions) followed by a
            # per-request unique tail so every request still hashes to a
            # distinct leaf while reusing the 98-block root.
            unique = list(range(next_unique_block, next_unique_block + UNIQUE_BLOCKS))
            next_unique_block += UNIQUE_BLOCKS
            hash_ids = SHARED_ROOT + unique
            assert len(hash_ids) * BLOCK_SIZE >= INPUT_LENGTH, (
                f"trace generator bug: input_length={INPUT_LENGTH} "
                f"hash_ids={len(hash_ids)} block_size={BLOCK_SIZE}"
            )

            # Poisson-ish inter-arrival (~120 ms mean) for a non-trivial
            # schedule across the 256 requests.
            timestamp_ms += int(rng.expovariate(1.0 / 120.0))  # nosec B311  # noqa: S311

            requests.append(
                {
                    "input_length": INPUT_LENGTH,
                    "output_length": OUTPUT_TOKENS,
                    "timestamp": timestamp_ms,
                    "hash_ids": hash_ids,
                    "session_id": f"cust-session-{session}",
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(
        f"Wrote {len(requests)} requests to {OUT_PATH} "
        f"(input_length={INPUT_LENGTH}, shared_prefix_blocks={SHARED_PREFIX_BLOCKS}, "
        f"steady-state reuse ~= {SHARED_PREFIX_TOKENS / INPUT_LENGTH * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()

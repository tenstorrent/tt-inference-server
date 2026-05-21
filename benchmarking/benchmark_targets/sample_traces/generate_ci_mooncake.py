# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Generate the in-tree mooncake JSONL trace used by the CI prefix-cache preset.

The output JSONL is consumed by AIPerf via
``--custom-dataset-type mooncake-trace --input-file <out>``. Each line is a
single request with the schema documented at
https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md

We synthesize a small, deterministic workload with three controlled prefix
groups so the CI run produces non-zero cache hits (the analyze-trace stats
expect ~0.5+ reuse ratio).

Usage:
    python benchmarking/benchmark_targets/sample_traces/generate_ci_mooncake.py
"""

import json
import random
from pathlib import Path

SEED = 9472
NUM_REQUESTS = 64
NUM_PREFIX_GROUPS = 4
BLOCK_SIZE = 512
OUT_PATH = Path(__file__).with_name("ci_mooncake.jsonl")


def main() -> None:
    rng = random.Random(SEED)
    # Each prefix group gets a unique [hash_id, hash_id, ...] root path.
    # AIPerf groups requests with identical hash_ids prefixes as cache-shared.
    groups = []
    for g in range(NUM_PREFIX_GROUPS):
        prefix_blocks = 1 + (g % 3)  # 1, 2 or 3 shared blocks
        # Stable, non-colliding ids per group (10*group + offset).
        groups.append(
            [10 * (g + 1) + i for i in range(prefix_blocks)]
        )

    isl_choices_short = (256, 384, 512)
    isl_choices_long = (1024, 1536, 2048)
    osl_choices = (64, 96, 128, 192)

    requests = []
    timestamp_ms = 0
    for i in range(NUM_REQUESTS):
        group_idx = rng.choices(
            range(NUM_PREFIX_GROUPS),
            # Skewed distribution so a few groups dominate (~ Zipf-like).
            weights=[8, 4, 2, 1],
            k=1,
        )[0]
        prefix_hashes = list(groups[group_idx])
        # Add a per-request unique suffix block so the trace exhibits both
        # shared roots and unique leaves.
        unique_block_id = 1000 + i
        hash_ids = prefix_hashes + [unique_block_id]

        # Mix short and long contexts so the CI preset covers both regimes.
        if rng.random() < 0.7:
            isl = rng.choice(isl_choices_short)
        else:
            isl = rng.choice(isl_choices_long)
        osl = rng.choice(osl_choices)

        # Poisson-ish inter-arrival (~25 ms mean) for a non-trivial schedule.
        timestamp_ms += int(rng.expovariate(1.0 / 25.0))

        requests.append(
            {
                "input_length": isl,
                "output_length": osl,
                "timestamp": timestamp_ms,
                "hash_ids": hash_ids,
                "session_id": f"ci-session-{group_idx}-{i // 8}",
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Wrote {len(requests)} requests to {OUT_PATH}")


if __name__ == "__main__":
    main()

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

Security note: this script uses ``random.Random(SEED)`` intentionally and is
not security-sensitive. The output is a fixture file committed to the repo
for byte-stable CI reproducibility; switching to ``secrets``/``os.urandom``
would produce a different trace on every regeneration and defeat the entire
purpose of the file. No value produced here is used as an identifier, token,
nonce, credential, or in any access-control decision -- they are
benchmark-input integers (input length, output length, block hashes,
inter-arrival deltas). The ``# nosec B311`` markers below acknowledge this.

Usage:
    python benchmarking/benchmark_targets/sample_traces/generate_ci_mooncake.py
"""

import json
import math
import random  # nosec B311  # noqa: S311  -- see module docstring
from pathlib import Path

SEED = 9472
NUM_REQUESTS = 64
NUM_PREFIX_GROUPS = 4
BLOCK_SIZE = 512
OUT_PATH = Path(__file__).with_name("ci_mooncake.jsonl")

# AIPerf 0.5 validates that
#     len(hash_ids) * BLOCK_SIZE >= input_length > (len(hash_ids) - 1) * BLOCK_SIZE
# i.e. len(hash_ids) == ceil(input_length / BLOCK_SIZE). The previous
# generator emitted a fixed `len(prefix_blocks)+1` regardless of ISL,
# which made every 1024+-token request fail with
#   ConfigurationError('Input length: 1536, Hash IDs: [..], Block size: 512 are not compatible.')
# Below we always size hash_ids = ceil(input_length / BLOCK_SIZE), keeping
# the group's shared prefix at the front (so AIPerf's radix-tree groups it
# with siblings) and padding the tail with per-request unique blocks.
#
# Why we only allow ISLs that are multiples of BLOCK_SIZE: AIPerf treats
# the trailing block as "fills exactly input_length - (len-1)*B tokens".
# We keep the math trivially valid by always picking ISL ∈ {B, 2B, 3B, 4B}
# so the trace's intent ("requests of length L share their first K
# blocks") is preserved without depending on AIPerf's fractional handling
# of the last block.


def main() -> None:
    rng = random.Random(SEED)  # nosec B311  # noqa: S311  -- deterministic CI fixture
    # Each prefix group gets a unique root path of shared blocks. AIPerf
    # groups requests with identical hash_ids prefixes as cache-shared,
    # so this controls the theoretical reuse ceiling per group.
    groups = []
    for g in range(NUM_PREFIX_GROUPS):
        # 1..3 shared blocks per group, deterministic
        prefix_blocks = 1 + (g % 3)
        # Stable, non-colliding ids per group (10*group + offset).
        groups.append([10 * (g + 1) + i for i in range(prefix_blocks)])

    # ISLs are exact multiples of BLOCK_SIZE so len(hash_ids) is unambiguous.
    isl_choices_short = (BLOCK_SIZE, 2 * BLOCK_SIZE)            #  512, 1024
    isl_choices_long = (3 * BLOCK_SIZE, 4 * BLOCK_SIZE)         # 1536, 2048
    osl_choices = (64, 96, 128, 192)

    requests = []
    timestamp_ms = 0
    next_unique_block = 10_000
    for i in range(NUM_REQUESTS):
        # All rng.* calls below are non-cryptographic by design; see module
        # docstring for the security rationale.
        group_idx = rng.choices(  # nosec B311  # noqa: S311
            range(NUM_PREFIX_GROUPS),
            # Skewed (Zipf-ish) distribution: a few groups dominate, the
            # rest are tail traffic. Produces non-trivial cache pressure.
            weights=[8, 4, 2, 1],
            k=1,
        )[0]
        prefix_hashes = list(groups[group_idx])

        # Mix short and long contexts so the CI preset exercises both
        # regimes (small ISL = cheap-to-recompute, long ISL = where
        # prefix caching actually pays off).
        if rng.random() < 0.7:  # nosec B311  # noqa: S311
            isl = rng.choice(isl_choices_short)  # nosec B311  # noqa: S311
        else:
            isl = rng.choice(isl_choices_long)  # nosec B311  # noqa: S311
        osl = rng.choice(osl_choices)  # nosec B311  # noqa: S311

        # Required: len(hash_ids) == ceil(isl / BLOCK_SIZE).
        num_blocks = math.ceil(isl / BLOCK_SIZE)
        # Keep the shared root, then pad with unique blocks so every
        # request still hashes to a distinct leaf.
        shared = prefix_hashes[: max(1, min(len(prefix_hashes), num_blocks - 1))]
        unique_tail_count = num_blocks - len(shared)
        unique = list(range(next_unique_block, next_unique_block + unique_tail_count))
        next_unique_block += unique_tail_count
        hash_ids = shared + unique
        assert len(hash_ids) * BLOCK_SIZE >= isl, (
            f"trace generator bug: isl={isl} hash_ids={hash_ids} "
            f"block_size={BLOCK_SIZE}"
        )

        # Poisson-ish inter-arrival (~25 ms mean) for a non-trivial schedule.
        timestamp_ms += int(rng.expovariate(1.0 / 25.0))  # nosec B311  # noqa: S311

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

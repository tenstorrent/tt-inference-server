# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Resolve opt-in benchmark protocol metadata consistently for setup and runs."""


def uses_fixed_workload(metadata: dict) -> bool:
    protocol = metadata.get("benchmark_protocol")
    if protocol is None:
        # Compatibility with catalogues written before the named protocol.
        return bool(metadata.get("benchmark_token_timing", False))
    if protocol not in {"standard", "fixed_workload"}:
        raise ValueError(f"Unknown benchmark protocol: {protocol!r}")
    return protocol == "fixed_workload"


def fixed_workload_repetitions(metadata: dict) -> int:
    count = metadata.get("benchmark_repetitions", 3)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("benchmark_repetitions must be a positive integer")
    return count

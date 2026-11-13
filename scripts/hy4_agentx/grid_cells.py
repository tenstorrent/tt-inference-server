#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Grid A (TTFT, OSL 128 ...) and Grid B (decode, OSL 4096) cell lists of the frozen benchmark suite (tt-bench-harness), CPU only, stdlib only.

usage: grid_cells.py pairs A|B [--maxisl N]          comma list "ISLxOSL,..." for TT_BENCH_ISL_OSL_PAIRS (cells above --maxisl are dropped)
       grid_cells.py table A|B <B> [--maxisl N]      one line per cell: isl osl conc num_prompts
       grid_cells.py cost <B> [--maxisl N] [--scale S] [--itl-ms MS]   rough wall-clock estimate (see BENCH_SUITE.md)
The near-max ISL is max_ctx - 4352 (1,044,224 for 1M): the server counts one extra prompt token and OSL 4096 needs room (same ISL in both grids).
Concurrencies are 1 and B (TT_BENCH_CONCURRENCIES=1,B); prompts per cell = stock get_num_prompts of reference_config/benchmarking (tests assert the two agree).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_model import HarnessModel  # noqa: E402

A_BASE = [(128, 128), (128, 1024), (1024, 128), (2048, 128), (4096, 128), (8192, 128), (8192, 1024), (10000, 1024), (16384, 128), (32768, 128),
          (65536, 128), (131072, 128), (262144, 128), (524288, 128)]
B_BASE_ISL = [128, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
NEAR_MAX_MARGIN = 4352


def get_num_prompts(input_len: int, output_len: int, max_concurrency: int) -> int:
    """Copy of reference_config.benchmarking.benchmark_config.get_num_prompts (min_num_prompts=0); a test keeps the two equal."""
    if output_len > 1024 or input_len > 16384:
        return 1 * max_concurrency
    if input_len > 4096:
        return 2 * max_concurrency
    if (128 < output_len <= 1024) or (128 < input_len <= 4096):
        return 4 * max_concurrency
    if output_len <= 128:
        return 8 * max_concurrency
    raise ValueError(output_len)


def grid_pairs(grid: str, max_ctx: int, maxisl: int = 0) -> List[Tuple[int, int]]:
    top = max_ctx - NEAR_MAX_MARGIN
    if grid == "A":
        pairs = A_BASE + [(top, 128)]
    elif grid == "B":
        pairs = [(i, 4096) for i in B_BASE_ISL] + [(top, 4096)]
    else:
        raise ValueError("grid must be A or B")
    return [p for p in pairs if p[0] + p[1] <= max_ctx and (not maxisl or p[0] <= maxisl)]


def cells(grid: str, b: int, max_ctx: int, maxisl: int = 0) -> List[Tuple[int, int, int, int]]:
    out = []
    for isl, osl in grid_pairs(grid, max_ctx, maxisl):
        for c in (1, b):
            out.append((isl, osl, c, get_num_prompts(isl, osl, c)))
    return out


# DeepSeek-V4.1-Flash release image 5, B=8, Grid A concurrency 1: ISL -> TTFT seconds (dsv4-logs/program/image5_release/RESULTS.md). The only measured reference we have.
DSV4_TTFT_C1_S = [(128, 0.37), (1024, 0.91), (4096, 2.15), (8192, 3.28), (16384, 5.5), (32768, 10.1), (65536, 19.3), (131072, 37.9), (262144, 76.3),
                  (524288, 152.3), (1044224, 309.0)]


def ttft_c1_dsv4(isl: int) -> float:
    pts = DSV4_TTFT_C1_S
    if isl <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if isl <= x1:
            return y0 + (y1 - y0) * (isl - x0) / (x1 - x0)
    return pts[-1][1] * isl / pts[-1][0]


def estimate_seconds(b: int, max_ctx: int, maxisl: int, scale: float = 3.0, itl_ms: float = 25.0, compile_s: float = 1800.0, cell_overhead_s: float = 30.0) -> dict:
    """ROUGH wall clock of the two grids at B (UNVERIFIED model, replace by the first measured grid): per cell, waves = ceil(n/c) sequential waves; a wave costs
    TTFT(isl, c) + osl * itl. TTFT(c=1) = DSV4 table * ``scale`` (Hy4 has 78 layers vs 40: scale ~2 is the layer ratio, 3-4 adds MoE/attention cost);
    TTFT(c=B) = TTFT(c=1) * m with m = 1.15 for isl >= 32768 (users sit on different mesh rows; DSV4 B=8: 525 s vs 309 s with TWO users per row) else 1.5.
    The first cell of each grid pays ``compile_s``; every cell ``cell_overhead_s``. Returns seconds per grid and the total."""
    res = {}
    for g in ("A", "B"):
        t = compile_s
        for isl, osl, c, n in cells(g, b, max_ctx, maxisl):
            waves = -(-n // c)
            m = 1.0 if c == 1 else (1.15 if isl >= 32768 else 1.5)
            t += cell_overhead_s + waves * (ttft_c1_dsv4(isl) * scale * m + osl * itl_ms / 1000.0)
        res[g] = t
    res["total"] = res["A"] + res["B"]
    return res


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    m = HarnessModel.load()
    maxisl = 0
    scale, itl = 3.0, 25.0
    for i, a in enumerate(argv):
        if a == "--maxisl":
            maxisl = int(argv[i + 1])
        if a == "--scale":
            scale = float(argv[i + 1])
        if a == "--itl-ms":
            itl = float(argv[i + 1])
    cmd = argv[0]
    if cmd == "pairs":
        print(",".join(f"{i}x{o}" for i, o in grid_pairs(argv[1], m.max_ctx, maxisl)))
    elif cmd == "table":
        for row in cells(argv[1], m.check_batch(int(argv[2])), m.max_ctx, maxisl):
            print(*row)
    elif cmd == "cost":
        r = estimate_seconds(m.check_batch(int(argv[1])), m.max_ctx, maxisl, scale, itl)
        print({k: round(v / 3600, 2) for k, v in r.items()}, "hours")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

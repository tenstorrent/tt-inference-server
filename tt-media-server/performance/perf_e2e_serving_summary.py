# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Mean/median/P99 latency block for perf_sp_runner_mock_e2e reports (reads E2E_RUNS_TSV)."""

import csv
import math
import os
import statistics
import sys


def read_rows(tsv_path):
    with open(tsv_path, newline="") as f:
        return list(csv.DictReader(f))


def floats(key, rows):
    out = []
    for row in rows:
        v = (row.get(key) or "").strip()
        if v and v != "n/a":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def p99_sorted(sorted_ms):
    if not sorted_ms:
        return float("nan")
    n = len(sorted_ms)
    idx = min(n - 1, max(0, int(math.ceil(0.99 * n) - 1)))
    return sorted_ms[idx]


def triplet_ms(vals_s):
    if not vals_s:
        return ("n/a", "n/a", "n/a")
    ms = sorted(v * 1000.0 for v in vals_s)
    mean = statistics.mean(ms)
    med = statistics.median(ms)
    p9 = p99_sorted(ms)
    return (f"{mean:.2f}", f"{med:.2f}", f"{p9:.2f}")


def main():
    tsv = os.environ.get("E2E_RUNS_TSV", "")
    nf = int(os.environ.get("TT_MOCK_OUTPUT_NUM_FRAMES", "81"))

    if not tsv or not os.path.isfile(tsv):
        print("")
        return 0

    rows = read_rows(tsv)
    n_ok = len(rows)
    wall_vals = floats("wall_s", rows)
    total_wall = sum(wall_vals) if wall_vals else 0.0

    client_ttf = floats("client_ttf_prog_s", rows)
    enc_ttft = floats("mock_ttft_first_frame_s", rows)
    enc_after = floats("mock_encode_after_first_s", rows)

    tput_per_frame_s = []
    if nf > 1:
        for v in enc_after:
            tput_per_frame_s.append(v / float(nf - 1))

    ct = triplet_ms(client_ttf)
    et = triplet_ms(enc_ttft)
    tp = triplet_ms(tput_per_frame_s)
    itl = tp

    req_per_s = n_ok / total_wall if total_wall > 0 else 0.0
    mean_wall = f"{total_wall / n_ok:.2f} s" if n_ok else "n/a"

    lines = [
        "",
        "## Latency summary (serving-style)",
        "",
        "### Formulas (how each metric is defined)",
        "",
        "- **Client TTFT (per job, seconds)** — `client_ttf_prog_s` in TSV: "
        "`time.time() - E2E_T0_FLOAT` at the first status GET that returns `in_progress` "
        "(`E2E_T0_FLOAT` = wall time immediately before the create-job POST). "
        "Table below: each sample × 1000 → **ms**; **Mean** = arithmetic mean; **Median** = middle of sorted ms; "
        "**P99** = sorted_ms[ min(n−1, ceil(0.99×n) − 1) ] on the ms list (n = number of valid samples).",
        "",
        "- **Encoder export wall (column still: Encoder TTFT, seconds)** — `mock_ttft_first_frame_s` "
        "copies `[VIDEO_DELIVERY]` `ttft_to_first_frame_appended_s`. For **batch** MP4 export "
        "(`encoder_incremental=False`, e.g. FFmpeg pipe), this equals **full** `export_to_mp4` "
        "duration, not “time to first output frame.” Name is legacy from incremental imageio.",
        "",
        "- **Encoder TPOT-like (synthetic, seconds)** — `mock_encode_after_first_s / (num_frames - 1)` "
        "where `mock_encode_after_first_s` = `encode_after_first_frame_s`. For **batch** encode this "
        "field is the **whole** ffmpeg (or one-shot) phase, not “after frame 1”; the ratio is only a "
        f"rough **average ms per frame** if work were uniform (`num_frames` = **{nf}**). "
        "Same ms / mean / median / P99.",
        "",
        "- **ITL (est.)** — Same synthetic value as **TPOT-like** here; not real inter-frame latency.",
        "",
        "- **Request throughput** — `successful_requests / sum(wall_s)` over TSV `wall_s` (integer client "
        "wall per job; sequential runs, so this is a coarse lower bound, not parallel RPS).",
        "",
        "TSV columns: `client_ttf_prog_s`, `mock_ttft_first_frame_s`, `mock_encode_after_first_s`.",
        "",
        "| Latency | Mean | Median | P99 | Unit |",
        "|---------|-----:|-------:|----:|------|",
        f"| Client TTFT | {ct[0]} | {ct[1]} | {ct[2]} | ms |",
        f"| Encoder TTFT | {et[0]} | {et[1]} | {et[2]} | ms |",
        f"| Encoder TPOT-like | {tp[0]} | {tp[1]} | {tp[2]} | ms |",
        f"| Inter-frame est. | {itl[0]} | {itl[1]} | {itl[2]} | ms |",
        "",
        "```",
        "=" * 80,
        "  Video mock e2e - benchmark summary (sequential jobs)",
        "=" * 80,
        "",
        "Formulas (see markdown section above for full text)",
        "  Client TTFT_s     = time to first GET with status=in_progress (from POST t0)",
        "  Encoder TTFT_s    = ttft field (batch=full export wall; legacy name)",
        "  TPOT-like_s       = encode_after / (num_frames - 1); batch=synthetic avg",
        "  ITL est._s        = same as TPOT-like_s (not real streaming ITL)",
        f"  num_frames        = {nf}",
        "  P99 on ms list    = sorted_ms[ceil(0.99*n)-1] clamped to [0, n-1]",
        "  req/s             = n_jobs / sum(wall_s)",
        "",
        "General",
        f"  Successful requests:           {n_ok}",
        "  Failed requests:                 0",
        "  Maximum request concurrency:     1",
        f"  Total client wall (sum):        {total_wall:.0f} s",
        f"  Mean client wall / job:         {mean_wall}",
        f"  Request throughput:             {req_per_s:.2f} req/s",
        "",
        "TTFT - client API",
        f"  Mean TTFT (ms):                  {ct[0]}",
        f"  Median TTFT (ms):                {ct[1]}",
        f"  P99 TTFT (ms):                   {ct[2]}",
        "",
        "Encoder TTFT",
        f"  Mean TTFT (ms):                  {et[0]}",
        f"  Median TTFT (ms):                {et[1]}",
        f"  P99 TTFT (ms):                   {et[2]}",
        "",
        "TPOT analog",
        f"  Mean TPOT (ms):                  {tp[0]}",
        f"  Median TPOT (ms):                {tp[1]}",
        f"  P99 TPOT (ms):                   {tp[2]}",
        "",
        "ITL analog",
        f"  Mean ITL (ms):                   {itl[0]}",
        f"  Median ITL (ms):                 {itl[1]}",
        f"  P99 ITL (ms):                    {itl[2]}",
        "",
        f"  num_frames divisor: {nf}",
        "=" * 80,
        "```",
        "",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

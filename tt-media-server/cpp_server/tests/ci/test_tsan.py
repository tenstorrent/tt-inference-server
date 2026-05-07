# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""ThreadSanitizer race-detection test for the cpp_server.

Replaces the runtime portion of test-gate.yml's `cpp-tsan` job — building the
server with `--tsan` still happens in YAML (it's a build-cache concern, not a
test). This test:

  1. Runs ctest under TSan suppressions (race reports go to ctest_tsan.log).
  2. Starts the TSan-instrumented server with log_path pointed at the test's
     artifacts dir, so TSan writes any race reports there as `tsan-server.<pid>`.
  3. Drives a concurrent multi-turn workload via guidellm.
  4. Counts "WARNING: ThreadSanitizer" lines across the ctest log and every
     tsan-server.<pid> file. Fails if the total (after suppressions) is non-zero.

Add new TSan-related cases by defining additional tests in this file — they
inherit the same suppressions and race-counting helpers.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import pytest

from _step_summary import write_step_summary

SUPPRESSIONS_FILENAME = "tsan_suppressions.txt"


def _tsan_options(suppressions: Path, *, log_path: str | None = None) -> str:
    parts = [
        f"suppressions={suppressions}",
        "halt_on_error=0",
        "second_deadlock_stack=1",
        "history_size=7",
    ]
    if log_path is not None:
        parts.append(f"log_path={log_path}")
    return ":".join(parts)


def _count_race_warnings(paths: Iterable[Path]) -> list[tuple[str, int]]:
    """Count `grep -c "WARNING: ThreadSanitizer"` per path."""
    counts: list[tuple[str, int]] = []
    for path in paths:
        if not path.exists():
            counts.append((path.name, 0))
            continue
        text = path.read_text(errors="replace")
        n = sum(1 for line in text.splitlines() if "WARNING: ThreadSanitizer" in line)
        counts.append((path.name, n))
    return counts


def _emit_summary(counts: list[tuple[str, int]], total: int) -> None:
    rows = "\n".join(f"| `{name}` | {n} |" for name, n in counts)
    write_step_summary(
        "## 🧵 ThreadSanitizer Results\n\n"
        "| Source | Race reports |\n"
        "|--------|------:|\n"
        f"{rows}\n"
        f"| **Total (after suppressions)** | **{total}** |\n\n"
    )


def test_no_data_races_after_ctest_and_workload(
    cpp_server,
    cpp_server_dir: Path,
    guidellm_bench,
    artifacts_dir: Path,
):
    suppressions = cpp_server_dir / SUPPRESSIONS_FILENAME
    assert suppressions.exists(), f"suppressions file missing: {suppressions}"

    build_dir = cpp_server_dir / "build"
    if not build_dir.exists():
        pytest.skip(
            f"build dir missing: {build_dir} "
            "(run `cpp_server/build.sh --tsan --blaze` first)"
        )

    # 1. Unit tests under TSan. Exit code is informational; the race count is
    #    what gates this test.
    ctest_log = artifacts_dir / "ctest_tsan.log"
    ctest_env = os.environ.copy()
    ctest_env["TSAN_OPTIONS"] = _tsan_options(suppressions)
    with open(ctest_log, "wb") as out:
        subprocess.run(
            ["ctest", "--output-on-failure", "-j4"],
            cwd=str(build_dir),
            env=ctest_env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )

    # 2. Server under TSan. Race reports land in artifacts_dir/tsan-server.<pid>
    #    so the per-test artifact upload picks them up automatically.
    tsan_log_prefix = artifacts_dir / "tsan-server"
    server = cpp_server(
        "tsan_server",
        port=8001,
        cwd=build_dir,
        env={"TSAN_OPTIONS": _tsan_options(suppressions, log_path=str(tsan_log_prefix))},
        timeout=180.0,
    )

    # 3. Concurrent multi-turn workload (8 users × 8 turns).
    guidellm_bench(
        label="cpp-tsan",
        target=server.base_url,
        output_subdir="tsan_ci",
        log_filename="tsan_bench.log",
        rate=8,
        max_requests=64,
        data="prefix_tokens=512,prompt_tokens=512,output_tokens=128,turns=8",
        extra_env={"TSAN_OPTIONS": _tsan_options(suppressions)},
    )

    # 4. Aggregate race counts from ctest output + every tsan-server.<pid>.
    tsan_server_files = sorted(artifacts_dir.glob("tsan-server.*"))
    counts = _count_race_warnings([ctest_log, *tsan_server_files])
    total = sum(n for _, n in counts)

    _emit_summary(counts, total)

    assert total == 0, (
        f"ThreadSanitizer detected {total} data race(s) after suppressions "
        f"(per-source: {counts}); race reports under {artifacts_dir}"
    )

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Codeforces Elo on open-r1/codeforces (verifiable-prompts, test split).

The test split holds 422 problems from late-2024 / early-2025 contests, each
with open-r1's ready-made C++17 prompt. A solution is judged like open-r1's
pass_fail mode: it must pass every official test plus open-r1's generated
tests, compared token-wise or with the problem's generated checker (a Python
program printing 1 for an accepted output). Judging compiles with the host's
g++ and runs the binary on this host under time and memory limits, so the
EvalTask must set allow_code_execution.

``codeforces_elo`` is the maximum-likelihood rating R under Codeforces'
problem-rating model, where a contestant rated R solves a problem rated r with
probability 1 / (1 + 10 ** ((r - R) / 400)) (a problem's rating is the
contestant rating with a 50% solve chance). Google does not publish how the
Gemma 4 card's Codeforces ELO is computed, so this is the closest public
protocol, not a reproduction of it.
"""

import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from itertools import zip_longest

import datasets

LANGUAGE = "cpp"
# The CI runner is not a Codeforces judge machine: allow some headroom over
# the problem's time limit. Override with CODEFORCES_TIME_LIMIT_MULTIPLIER.
TIME_LIMIT_MULTIPLIER = float(os.environ.get("CODEFORCES_TIME_LIMIT_MULTIPLIER", "2.0"))
COMPILE_TIMEOUT_SEC = 120
CHECKER_TIMEOUT_SEC = 60
# Address-space limits count reserved but untouched memory, which Codeforces'
# peak-memory measurement does not; double the limit to compensate.
MEMORY_LIMIT_FACTOR = 2
# Rating buckets (lower bounds) for the round-robin order of process_docs.
RATING_BUCKETS = (1200, 1600, 2000, 2400)
SHUFFLE_SEED = 1234
ELO_RANGE = (0.0, 4000.0)

_CODE_BLOCK = re.compile(r"```([\w+#-]*)[ \t]*\n(.*?)```", re.DOTALL)
_CPP_TAGS = {"cpp", "c++", "cc", "cxx", "c"}


def process_docs(dataset):
    """Rated, non-interactive C++ prompts, interleaved round-robin across
    rating buckets so a --limit prefix spans the difficulty range."""
    buckets = [[] for _ in range(len(RATING_BUCKETS) + 1)]
    for row in dataset:
        if (
            row["language"] != LANGUAGE
            or row["rating"] is None
            or row["interaction_format"]
            or not row["executable"]
        ):
            continue
        buckets[sum(row["rating"] >= b for b in RATING_BUCKETS)].append(row)
    rng = random.Random(SHUFFLE_SEED)
    for rows in buckets:
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
    ordered = [row for group in zip_longest(*buckets) for row in group if row]
    return datasets.Dataset.from_list(ordered)


def extract_code(response: str):
    """The last C++ code block (or the last code block of any language)."""
    blocks = _CODE_BLOCK.findall(response or "")
    if not blocks:
        return None
    cpp = [code for tag, code in blocks if tag.lower() in _CPP_TAGS]
    return (cpp or [code for _, code in blocks])[-1]


@lru_cache(maxsize=8)
def _contest_generated_tests(contest_id: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id="open-r1/codeforces",
            filename=f"generated_tests/test_cases_{int(contest_id):04d}.parquet",
            repo_type="dataset",
        )
    except EntryNotFoundError:
        return {}
    df = pd.read_parquet(path).sort_values("test_i")
    return {
        problem_id: group[["input", "output"]].to_dict("records")
        for problem_id, group in df.groupby("problem_id")
    }


def _tests(doc):
    generated = []
    if doc["generated_tests"]:
        generated = _contest_generated_tests(doc["contest_id"]).get(doc["id"], [])
    return list(doc["official_tests"] or []) + generated


def _limit_resources(memory_bytes: int, cpu_seconds: int):
    # Best effort: an exception here would abort the whole scoring pass, and the
    # wall-clock timeout in judge() still bounds every run. Codeforces runs with
    # a stack as large as the memory limit.
    limits = (
        (resource.RLIMIT_AS, (memory_bytes, memory_bytes)),
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
        (resource.RLIMIT_STACK, (memory_bytes, memory_bytes)),
    )

    def _apply():
        for limit, value in limits:
            try:
                resource.setrlimit(limit, value)
            except (ValueError, OSError):
                pass

    return _apply


def _outputs_match(workdir, test_input, expected, actual, checker) -> bool:
    if not checker:
        return actual.split() == expected.split()
    for name, text in (
        ("input.txt", test_input),
        ("correct_output.txt", expected),
        ("solution_output.txt", actual),
    ):
        with open(os.path.join(workdir, name), "w") as f:
            f.write(text)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "checker.py",
                "input.txt",
                "correct_output.txt",
                "solution_output.txt",
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=CHECKER_TIMEOUT_SEC,
        )
        verdict = float(result.stdout.split()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return False
    # Checkers print 0-1 (or 0-100); only a full score accepts the output.
    return abs(verdict - 1.0) < 1e-6 or abs(verdict - 100.0) < 1e-6


def judge(doc, response: str) -> bool:
    code = extract_code(response)
    if code is None:
        return False
    tests = _tests(doc)
    if not tests:
        return False
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("codeforces_elo needs g++ on the eval host")

    with tempfile.TemporaryDirectory(prefix="codeforces_") as workdir:
        with open(os.path.join(workdir, "main.cpp"), "w") as f:
            f.write(code)
        try:
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-pipe", "-o", "main", "main.cpp"],
                cwd=workdir,
                capture_output=True,
                timeout=COMPILE_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return False
        if build.returncode != 0:
            return False
        if doc["generated_checker"]:
            with open(os.path.join(workdir, "checker.py"), "w") as f:
                f.write(doc["generated_checker"])

        time_limit = float(doc["time_limit"] or 1.0) * TIME_LIMIT_MULTIPLIER
        memory_bytes = int(float(doc["memory_limit"] or 256.0) * MEMORY_LIMIT_FACTOR)
        memory_bytes *= 1024 * 1024
        limits = _limit_resources(memory_bytes, int(time_limit) + 1)
        file_mode = doc["input_mode"] == "file"
        for test in tests:
            if file_mode:
                with open(os.path.join(workdir, "input.txt"), "w") as f:
                    f.write(test["input"])
            try:
                run = subprocess.run(
                    [os.path.join(workdir, "main")],
                    cwd=workdir,
                    input=None if file_mode else test["input"],
                    capture_output=True,
                    text=True,
                    timeout=time_limit + 1.0,
                    preexec_fn=limits,
                )
            except subprocess.TimeoutExpired:
                return False
            if run.returncode != 0:
                return False
            actual = run.stdout
            if file_mode:
                output_path = os.path.join(workdir, "output.txt")
                if not os.path.exists(output_path):
                    return False
                with open(output_path) as f:
                    actual = f.read()
            if not _outputs_match(
                workdir,
                test["input"],
                test["output"],
                actual,
                doc["generated_checker"],
            ):
                return False
    return True


def process_results(doc, results):
    solved = float(judge(doc, results[0]))
    return {
        "pass_at_1": solved,
        "codeforces_elo": (float(doc["rating"]), solved),
    }


def _solve_probability(contestant: float, problem: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((problem - contestant) / 400.0))


def elo_aggregation(items) -> float:
    """Maximum-likelihood rating: the R at which the expected number of solved
    problems equals the observed number (clamped to ELO_RANGE)."""
    items = [(float(rating), float(solved)) for rating, solved in items]
    solved = sum(s for _, s in items)
    lo, hi = ELO_RANGE
    if solved <= 0:
        return lo
    if solved >= len(items):
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        expected = sum(_solve_probability(mid, rating) for rating, _ in items)
        if expected < solved:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

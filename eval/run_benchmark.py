#!/usr/bin/env python3
"""Synthetic benchmark runner.

Usage:
    python3 eval/run_benchmark.py --model MODEL --provider PROVIDER [--api-key KEY]

Example:
    python3 eval/run_benchmark.py --model anthropic/claude-sonnet-4-6 --provider litellm
"""
import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_DIR = os.path.join(_EVAL_DIR, "fixture")

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _EVAL_DIR)

from orchestrator import agent as agent_mod
from orchestrator.personas import IMPLEMENTER, CORRECTNESS_REVIEWER
import orchestrator.tools as T
from benchmark_tasks import IMPLEMENTER_TASKS, REVIEWER_TASKS

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run the synthetic benchmark.")
parser.add_argument("--model", required=True)
parser.add_argument("--provider", required=True)
parser.add_argument("--api-key", dest="api_key", default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Persona builders
# ---------------------------------------------------------------------------

def _make_implementer(model: str, provider: str) -> dict:
    p = copy.deepcopy(IMPLEMENTER)
    p["model"] = model
    p["provider"] = provider
    if provider == "tt-console":
        p["max_tokens"] = 131072
    return p


def _make_reviewer(model: str, provider: str) -> dict:
    p = copy.deepcopy(CORRECTNESS_REVIEWER)
    p["model"] = model
    p["provider"] = provider
    if provider == "tt-console":
        p["max_tokens"] = 131072
    # Tell the reviewer the diff is self-contained so it doesn't loop on tool calls.
    p["system"] = (
        p["system"]
        + "\n\nFor this evaluation task the diff is provided directly in the message."
        " Review it as-is; you do not need to fetch additional files."
    )
    return p


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _setup_temp_fixture() -> tuple[str, str]:
    """Copy eval/fixture/ into a fresh temp dir, init a git repo, return (tmp_dir, initial_hash)."""
    tmp = tempfile.mkdtemp(prefix="benchmark_")
    for item in os.listdir(_FIXTURE_DIR):
        if item == "__pycache__":
            continue
        src = os.path.join(_FIXTURE_DIR, item)
        dst = os.path.join(tmp, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    for cmd in [
        ["git", "init"],
        ["git", "config", "user.email", "benchmark@test.local"],
        ["git", "config", "user.name", "Benchmark"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "initial fixture"],
    ]:
        subprocess.run(cmd, cwd=tmp, capture_output=True, check=True)
    initial_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp, capture_output=True, text=True, check=True
    ).stdout.strip()
    return tmp, initial_hash


def _run_tests(test_command: str, cwd: str) -> bool:
    r = subprocess.run(
        test_command, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=60
    )
    return r.returncode == 0


def _changed_files(cwd: str, initial_hash: str) -> list[str]:
    """List files changed between initial_hash and current working tree (including staged)."""
    r = subprocess.run(
        ["git", "diff", initial_hash, "--name-only"],
        cwd=cwd, capture_output=True, text=True
    )
    return [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]


def _check_minimal(fixture_file: str, cwd: str, initial_hash: str) -> bool:
    changed = _changed_files(cwd, initial_hash)
    test_file = f"tests/test_{fixture_file.replace('.py', '')}.py"
    allowed = {fixture_file, test_file}
    return all(f in allowed for f in changed)


def _check_debris(cwd: str) -> bool:
    r = subprocess.run(["git", "status", "--short"], cwd=cwd, capture_output=True, text=True)
    untracked = [l for l in r.stdout.strip().split("\n") if l.startswith("??")]
    return len(untracked) == 0


def _has_termination_token(text: str) -> bool:
    return "IMPLEMENTATION_COMPLETE" in text


def _has_think_leakage(text: str) -> bool:
    return bool(re.search(r"<think\b", text, re.IGNORECASE))


def _extract_verdict(text: str) -> str | None:
    if re.search(r"\bAPPROVED\b", text):
        return "APPROVED"
    if re.search(r"\bOBJECTION\b", text):
        return "OBJECTION"
    return None


# ---------------------------------------------------------------------------
# Implementer benchmark
# ---------------------------------------------------------------------------

print("=== Running implementer benchmark ===")
print()

impl_persona = _make_implementer(args.model, args.provider)
impl_results = []
all_outputs: list[str] = []

for task in IMPLEMENTER_TASKS:
    print(f"--- {task['label']} ---")
    tmp_dir, initial_hash = _setup_temp_fixture()
    try:
        prompt = (
            f"You are working in the directory: {tmp_dir}\n\n"
            f"{task['prompt']}\n\n"
            "Do NOT create a PR or push to any remote. "
            "Fix the code in-place, verify with the test command shown above, "
            "then emit IMPLEMENTATION_COMPLETE."
        )
        try:
            text, _ = agent_mod.run(
                impl_persona,
                [{"role": "user", "content": prompt}],
                cwd=tmp_dir,
                max_tool_rounds=15,
                verbose=True,
                api_key=args.api_key,
            )
        except agent_mod.MaxToolRoundsError as exc:
            print(f"  MaxToolRoundsError: {exc}")
            text = ""

        all_outputs.append(text)

        correct = _run_tests(task["test_command"], tmp_dir)
        minimal = _check_minimal(task["fixture_file"], tmp_dir, initial_hash)
        token = _has_termination_token(text)
        no_debris = _check_debris(tmp_dir)
        task_pass = correct and minimal and token and no_debris

        impl_results.append({
            "label": task["label"],
            "correct": correct,
            "minimal": minimal,
            "token": token,
            "no_debris": no_debris,
            "pass": task_pass,
        })
        print()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Reviewer benchmark
# ---------------------------------------------------------------------------

print("=== Running reviewer benchmark ===")
print()

reviewer_persona = _make_reviewer(args.model, args.provider)
reviewer_results = []

for task in REVIEWER_TASKS:
    print(f"--- {task['label']} ---")
    prompt = (
        "Review the following diff and vote.\n\n"
        f"```diff\n{task['diff']}\n```\n\n"
        "Reason through the change, then end with exactly one of:\n"
        "  APPROVED\n"
        "  OBJECTION: <concise list of concerns>"
    )
    tmp_dir = tempfile.mkdtemp(prefix="reviewer_")
    try:
        for cmd in [["git", "init"], ["git", "config", "user.email", "x@x.x"], ["git", "config", "user.name", "X"]]:
            subprocess.run(cmd, cwd=tmp_dir, capture_output=True)
        try:
            text, _ = agent_mod.run(
                reviewer_persona,
                [{"role": "user", "content": prompt}],
                cwd=tmp_dir,
                max_tool_rounds=5,
                verbose=True,
                api_key=args.api_key,
            )
        except agent_mod.MaxToolRoundsError as exc:
            print(f"  MaxToolRoundsError: {exc}")
            text = ""

        all_outputs.append(text)
        verdict = _extract_verdict(text)
        expected = task["expected_verdict"]

        reviewer_results.append({
            "label": task["label"],
            "expected": expected,
            "got": verdict or "NONE",
            "pass": verdict == expected,
        })
        print()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

SEP = "=" * 60
print(SEP)
print()

# --- Implementer ---
print("=== Implementer Benchmark ===")
impl_pass_count = sum(1 for r in impl_results if r["pass"])
for r in impl_results:
    status = "PASS" if r["pass"] else "FAIL"
    details = (
        f"correct={'yes' if r['correct'] else 'no'} "
        f"minimal={'yes' if r['minimal'] else 'no'} "
        f"token={'yes' if r['token'] else 'no'} "
        f"debris={'no' if r['no_debris'] else 'yes'}"
    )
    print(f"{r['label'] + ':':<35} {status} ({details})")

impl_threshold = 4
impl_bench_pass = impl_pass_count >= impl_threshold
print(
    f"Score: {impl_pass_count}/{len(impl_results)}  "
    f"{'PASS' if impl_bench_pass else 'FAIL'} "
    f"(threshold: {impl_threshold}/{len(impl_results)})"
)

# --- Reviewer ---
print()
print("=== Reviewer Benchmark ===")

total_bugs = sum(1 for t in REVIEWER_TASKS if t["expected_verdict"] == "OBJECTION")
tp = sum(1 for r in reviewer_results if r["expected"] == "OBJECTION" and r["got"] == "OBJECTION")
fp = sum(1 for r in reviewer_results if r["expected"] == "APPROVED" and r["got"] == "OBJECTION")

for r in reviewer_results:
    status = "PASS" if r["pass"] else "FAIL"
    print(f"{r['label'] + ':':<35} {status} (got {r['got']}, expected {r['expected']})")

recall_denom = total_bugs if total_bugs > 0 else 1
recall = tp / recall_denom
precision_denom = tp + fp if (tp + fp) > 0 else 1
reviewer_bench_pass = recall >= 0.9 and fp <= 1
print(
    f"Recall: {tp}/{total_bugs}  "
    f"Precision: {tp}/{tp + fp}  "
    f"{'PASS' if reviewer_bench_pass else 'FAIL'} "
    f"(threshold: recall>=90%, FP<=1)"
)

# --- Role Coherence ---
print()
print("=== Role Coherence ===")

think_leaks = sum(1 for t in all_outputs if _has_think_leakage(t))
# Implementer outputs are the first len(IMPLEMENTER_TASKS) entries
impl_token_count = sum(
    1 for t in all_outputs[:len(IMPLEMENTER_TASKS)] if _has_termination_token(t)
)
coherence_think_pass = think_leaks == 0
coherence_token_pass = impl_token_count == len(IMPLEMENTER_TASKS)
coherence_pass = coherence_think_pass and coherence_token_pass

print(
    f"{'<think> tag leakage:':<35} "
    f"{'PASS' if coherence_think_pass else 'FAIL'} ({think_leaks} occurrences)"
)
print(
    f"{'Termination token:':<35} "
    f"{'PASS' if coherence_token_pass else 'FAIL'} "
    f"({impl_token_count}/{len(IMPLEMENTER_TASKS)} tasks)"
)
print(f"{'Overall coherence:':<35} {'PASS' if coherence_pass else 'FAIL'}")

# --- Overall ---
print()
overall_pass = impl_bench_pass and reviewer_bench_pass and coherence_pass
print(f"=== OVERALL: {'PASS' if overall_pass else 'FAIL'} ===")
print()

sys.exit(0 if overall_pass else 1)

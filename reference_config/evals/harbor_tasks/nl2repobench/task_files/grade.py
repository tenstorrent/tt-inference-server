#!/usr/bin/env python3
"""NL2Repo-Bench grader: the official post-processing, run in the separate verifier.

Mirrors multimodal-art-projection/NL2RepoBench openhands/post_processor.py @781a1da:
  1. remove_package_files from the agent workspace (any depth)
  2. remove_test_files (test_files.json entries, relative to the workspace)
  3. official image /workspace, with the cleaned agent workspace copied over it
  4. run test_commands.json in /workspace and count passed tests
  5. reward = min(passed / test_case_count, 1)

with fixes for reward hacks in that pipeline (cf. PrimeIntellect-ai/prime-envs#929):
  - installed copies of the project outside /workspace are deleted at image
    build and before every pytest command (drop_preinstalled.py); otherwise a
    failed import of the agent's package falls back to the real library
  - agent conftest.py / sitecustomize.py / usercustomize.py are stripped, and
    top-level agent modules that shadow anything importable in the verifier
    (pytest, stdlib, ...) are stripped; PYTHONPATH=/workspace puts them first
  - the reference's test files and conftests win over agent files
  - only pytest's final summary line counts (upstream sums every "N passed" in
    the log); a clean summary with a non-zero pytest exit counts as fabricated
  - bare `pytest` commands name the hidden test paths (gen_tasks.py)
The upstream count is still recorded as "upstream_passed" for comparison.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AGENT_WS = Path("/workspace")
REF_WS = Path("/opt/nl2repo_ref")
STAGE = Path("/tmp/nl2repo_agent_ws")
META = Path("/tests/nl2repo")
OUT = Path("/logs/verifier")

PACKAGE_FILES = {
    "setup.py", "pyproject.toml", "setup.cfg", "requirements.txt",
    "requirements-dev.txt", "requirements-test.txt", "tox.ini", "pytest.ini",
    "poetry.lock", "Pipfile", "Pipfile.lock", "environment.yml",
    "conda-env.yaml", "manifest.in", "MANIFEST.in",
}
AGENT_HOOK_FILES = {"conftest.py", "sitecustomize.py", "usercustomize.py"}
TEST_DIRS = {"tests", "test", "testing"}
_MODULE_SUFFIXES = {".py", ".pyc", ".so", ".pyd"}
COMMAND_TIMEOUT_SEC = int(os.environ.get("NL2REPO_COMMAND_TIMEOUT_SEC", "1800"))


def _merge_tree(src: Path, dst: Path) -> None:
    """copytree(dirs_exist_ok=True) for Python 3.7 verifier images."""
    for root, dirs, files in os.walk(str(src)):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in dirs:
            s = Path(root) / name
            if s.is_symlink():
                d = dst / rel / name
                if d.is_symlink() or d.is_file():
                    d.unlink()
                elif d.is_dir():
                    shutil.rmtree(str(d))
                os.symlink(os.readlink(str(s)), str(d))
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            if d.is_dir() and not d.is_symlink():
                shutil.rmtree(str(d))
            elif d.exists() or d.is_symlink():
                d.unlink()
            if s.is_symlink():
                os.symlink(os.readlink(str(s)), str(d))
            else:
                shutil.copy2(str(s), str(d))


def _is_test_path(rel: Path) -> bool:
    name = rel.name
    return (
        name == "conftest.py"
        or (name.endswith(".py") and (name.startswith("test") or name.endswith("_test.py")))
        or bool(TEST_DIRS & set(rel.parts[:-1]))
    )


def _shadowed_names(stage: Path, keep: set) -> list:
    """Top-level agent entries named like a module importable without /workspace.

    `keep` (the project's own module names) is never stripped: the verifier
    image has any preinstalled copy of the project removed at build time.
    """
    saved = sys.path[:]
    sys.path[:] = [
        p for p in sys.path[1:]
        if p and not p.startswith(("/workspace", "/tests", str(stage)))
    ]
    try:
        hits = []
        for entry in sorted(stage.iterdir()):
            if not (entry.is_dir() or entry.suffix in _MODULE_SUFFIXES):
                continue
            name = entry.name.split(".")[0]
            if not name.isidentifier() or name in keep:
                continue
            try:
                found = importlib.util.find_spec(name) is not None
            except Exception:
                found = False
            if found:
                hits.append(entry)
        return hits
    finally:
        sys.path[:] = saved


# Final summary, default or -q: "==== 1 failed, 5 passed in 0.12s (0:00:01) ===="
# (pytest < 5: "in 0.12 seconds").
_SUMMARY_RE = re.compile(
    r"^(?:=+ )?((?:\d+ [a-z]+(?:, )?)+|no tests ran) in [\d.]+ ?s(?:econds)?"
    r"(?: \([^)]*\))?(?: =+)?$"
)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_pytest(output: str, exit_code: int) -> dict:
    summaries = [
        line.strip() for line in _ANSI_RE.sub("", output).split("\n")
        if _SUMMARY_RE.match(line.strip())
    ]
    counts = {}
    if summaries:
        for n, kind in re.findall(r"(\d+) ([a-z]+)", summaries[-1]):
            kind = {"errors": "error", "warnings": "warning"}.get(kind, kind)
            counts[kind] = counts.get(kind, 0) + int(n)
    passed = counts.get("passed", 0)
    failed, errors = counts.get("failed", 0), counts.get("error", 0)
    # pytest exits 0 exactly when nothing failed or errored, so a clean summary
    # from a failing run was printed by something other than pytest.
    fabricated = passed > 0 and not failed and not errors and exit_code != 0
    return {
        "summary": summaries[-1] if summaries else None,
        "summary_lines": len(summaries),
        "passed": 0 if fabricated else passed,
        "failed": failed, "errors": errors, "fabricated": fabricated,
    }


def upstream_count(output: str) -> int:
    """post_processor.py: first "(\\d+) passed" match on every line."""
    passed = 0
    for line in output.split("\n"):
        m = re.search(r"(\d+) passed", line)
        if m:
            passed += int(m.group(1))
    return passed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_commands = json.loads((META / "test_commands.json").read_text())
    test_files = json.loads((META / "test_files.json").read_text())
    test_case_count = int((META / "test_case_count.txt").read_text().strip())

    # The agent workspace artifact re-materializes at /workspace; move it aside.
    if STAGE.exists():
        shutil.rmtree(str(STAGE))
    if AGENT_WS.exists():
        shutil.move(str(AGENT_WS), str(STAGE))
    else:
        STAGE.mkdir(parents=True)

    removed = []
    for root, _dirs, files in os.walk(STAGE):
        for name in files:
            if name in PACKAGE_FILES or name in AGENT_HOOK_FILES:
                path = Path(root) / name
                path.unlink()
                removed.append(str(path.relative_to(STAGE)))
    for rel in test_files:
        path = STAGE / rel
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(str(path))
            removed.append(rel + "/")
        elif path.exists() or path.is_symlink():
            path.unlink()
            removed.append(rel)

    keep_path = META / "target_modules.json"
    keep = set(json.loads(keep_path.read_text())) if keep_path.exists() else set()
    for entry in _shadowed_names(STAGE, keep):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(str(entry))
            removed.append(entry.name + "/ (shadows an installed module)")
        else:
            entry.unlink()
            removed.append(entry.name + " (shadows an installed module)")

    # COPY workspace /workspace onto the official image's /workspace, then put
    # the reference's test files back over anything the agent wrote there.
    shutil.copytree(str(REF_WS), str(AGENT_WS), symlinks=True)
    _merge_tree(STAGE, AGENT_WS)
    restored = []
    for root, _dirs, files in os.walk(str(REF_WS)):
        for name in files:
            src = Path(root) / name
            rel = src.relative_to(REF_WS)
            dst = AGENT_WS / rel
            if not _is_test_path(rel) or src.is_symlink():
                continue
            if dst.is_symlink() or not dst.is_file() or dst.read_bytes() != src.read_bytes():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(str(dst))
                elif dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                restored.append(str(rel))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"/workspace:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "/workspace"
    results = []
    passed = failed = errors = upstream_passed = 0
    for command in test_commands:
        dropped = ""
        if "pytest" in command.lower() and keep:
            # Test commands may install the real project as a dependency.
            proc = subprocess.run(
                ["bash", "-lc", "python /tests/drop_preinstalled.py " + " ".join(sorted(keep))],
                cwd=AGENT_WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            dropped = proc.stdout.decode("utf-8", errors="replace")
        try:
            proc = subprocess.run(
                ["bash", "-lc", command], cwd=AGENT_WS, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=COMMAND_TIMEOUT_SEC,
            )
            output = proc.stdout.decode("utf-8", errors="replace")
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or b"").decode("utf-8", errors="replace")
            output += f"\n[grade.py] command timed out after {COMMAND_TIMEOUT_SEC}s"
            exit_code = -1
        result = {"command": command, "exit_code": exit_code}
        if dropped:
            result["dropped_installed_target"] = dropped.strip().splitlines()
        (OUT / f"command_{len(results) + 1}.log").write_text(dropped + output)
        if "pytest" in command.lower():
            result.update(parse_pytest(output, exit_code))
            result["upstream_passed"] = upstream_count(output)
            passed += result["passed"]
            failed += result["failed"]
            errors += result["errors"]
            upstream_passed += result["upstream_passed"]
        results.append(result)

    reward = min(passed / test_case_count, 1) if test_case_count > 0 else 0.0
    upstream_reward = min(upstream_passed / test_case_count, 1) if test_case_count > 0 else 0.0
    (OUT / "reward.txt").write_text(f"{reward}\n")
    (OUT / "nl2repo_result.json").write_text(json.dumps({
        "passed": passed, "failed": failed, "errors": errors,
        "test_case_count": test_case_count, "reward": reward,
        "upstream_passed": upstream_passed, "upstream_reward": upstream_reward,
        "removed_from_agent_workspace": removed,
        "restored_reference_test_files": restored, "commands": results,
    }, indent=2))
    print(f"passed={passed} failed={failed} errors={errors} total={test_case_count} reward={reward:.4f} upstream_reward={upstream_reward:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

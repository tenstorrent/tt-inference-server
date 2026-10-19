#!/usr/bin/env python3
"""NL2Repo-Bench grader: the official post-processing, run in the separate verifier.

Mirrors multimodal-art-projection/NL2RepoBench openhands/post_processor.py @781a1da:
  1. remove_package_files from the agent workspace (any depth)
  2. remove_test_files (test_files.json entries, relative to the workspace)
  3. official image /workspace, with the cleaned agent workspace copied over it
  4. run test_commands.json in /workspace; for commands containing "pytest" sum
     every "(\\d+) passed" match per output line
  5. reward = min(passed / test_case_count, 1)
"""

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
COMMAND_TIMEOUT_SEC = int(os.environ.get("NL2REPO_COMMAND_TIMEOUT_SEC", "1800"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_commands = json.loads((META / "test_commands.json").read_text())
    test_files = json.loads((META / "test_files.json").read_text())
    test_case_count = int((META / "test_case_count.txt").read_text().strip())

    # The agent workspace artifact re-materializes at /workspace; move it aside.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    if AGENT_WS.exists():
        shutil.move(str(AGENT_WS), str(STAGE))
    else:
        STAGE.mkdir(parents=True)

    removed = []
    for root, _dirs, files in os.walk(STAGE):
        for name in files:
            if name in PACKAGE_FILES:
                path = Path(root) / name
                path.unlink()
                removed.append(str(path.relative_to(STAGE)))
    for rel in test_files:
        path = STAGE / rel
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            removed.append(rel + "/")
        elif path.exists() or path.is_symlink():
            path.unlink()
            removed.append(rel)

    # COPY workspace /workspace onto the official image's /workspace.
    shutil.copytree(REF_WS, AGENT_WS, symlinks=True)
    shutil.copytree(STAGE, AGENT_WS, symlinks=True, dirs_exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"/workspace:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "/workspace"
    results = []
    passed = failed = errors = 0
    for command in test_commands:
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
        results.append({"command": command, "exit_code": exit_code})
        (OUT / f"command_{len(results)}.log").write_text(output)
        if "pytest" in command.lower():
            for line in output.split("\n"):
                m = re.search(r"(\d+) passed", line)
                if m:
                    passed += int(m.group(1))
                m = re.search(r"(\d+) failed", line)
                if m:
                    failed += int(m.group(1))
                m = re.search(r"(\d+) error", line)
                if m:
                    errors += int(m.group(1))

    reward = min(passed / test_case_count, 1) if test_case_count > 0 else 0.0
    (OUT / "reward.txt").write_text(f"{reward}\n")
    (OUT / "nl2repo_result.json").write_text(json.dumps({
        "passed": passed, "failed": failed, "errors": errors,
        "test_case_count": test_case_count, "reward": reward,
        "removed_from_agent_workspace": removed, "commands": results,
    }, indent=2))
    print(f"passed={passed} failed={failed} errors={errors} total={test_case_count} reward={reward:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

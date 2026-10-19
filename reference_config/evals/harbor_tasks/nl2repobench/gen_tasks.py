#!/usr/bin/env python3
"""Generate Harbor tasks for NL2Repo-Bench from the upstream repo's test_files/.

usage: gen_tasks.py <upstream NL2RepoBench dir> <out dir> [task ...]

Per task (see NOTES.md for the design):
  environment/  agent image: python:3.12 + /workspace/start.md + repo-fetch guard
  tests/        separate verifier image: official image, reference moved aside
  instruction.md  the official prompt, verbatim
Target names for the guard are read from the official image, which must be
pullable (docker as the current user).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "task_files"
IMAGE_REPO = "ghcr.io/multimodal-art-projection/nl2repobench"
UPSTREAM_COMMIT = "781a1da"
AGENT_IMAGE = "python:3.12-bookworm"

# openhands/openhands_app.py @781a1da
OFFICIAL_PROMPT = (
    "According to the start.md in the workspace, implement the entire project as per "
    "the requirements specified in the document, ensuring that the final product can be "
    "directly run in the current directory. The running requirements should comply with "
    "the <API Usage Guide> section of the document. Please complete this task step by step."
)

# Lists the target's own names as seen from the official image: the distribution
# name, and top-level modules the tests import that are not installed (= the
# package under test, which the image deliberately lacks).
_PROBE = r'''
import ast, importlib.util, json, pathlib, re, sys
ws = pathlib.Path("/workspace")
names = set()
for f in ("pyproject.toml", "setup.cfg", "setup.py"):
    p = ws / f
    if p.exists():
        t = p.read_text(errors="replace")
        for m in re.finditer(r"""(?m)^\s*name\s*[=:]\s*["']?([A-Za-z0-9_.\-]+)""", t):
            names.add(m.group(1))
        for m in re.finditer(r"""name\s*=\s*["']([A-Za-z0-9_.\-]+)["']""", t):
            names.add(m.group(1))
local = {p.stem for p in ws.rglob("*.py")} | {p.name for p in ws.rglob("*") if p.is_dir()}
mods = set()
for p in ws.rglob("*.py"):
    try:
        tree = ast.parse(p.read_text(errors="replace"))
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
for m in sorted(mods):
    if m in local or m in sys.stdlib_module_names:
        continue
    try:
        found = importlib.util.find_spec(m) is not None
    except Exception:
        found = False
    if not found:
        names.add(m)
print(json.dumps({"names": sorted(names), "workspace": sorted(p.name for p in ws.iterdir())}))
'''


def image_for(task: str) -> str:
    return f"{IMAGE_REPO}/{task.lower()}:1.0"


def probe_image(task: str) -> dict:
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image_for(task), "-c", _PROBE],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def toml_str(s: str) -> str:
    return json.dumps(s)


def write_task(src: Path, out_root: Path) -> dict:
    task = src.name
    probe = probe_image(task)
    targets = sorted({task, *probe["names"]} - {"tests", "test", "src", "setup"})
    dst = out_root / task
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "environment").mkdir(parents=True)
    (dst / "tests" / "nl2repo").mkdir(parents=True)

    shutil.copy(src / "start.md", dst / "environment" / "start.md")
    shutil.copy(TEMPLATE / "guard.py", dst / "environment" / "nl2repo_guard.py")
    (dst / "environment" / "nl2repo-guard.json").write_text(
        json.dumps({"task": task, "targets": targets}, indent=2) + "\n"
    )
    (dst / "environment" / "managed-settings.json").write_text(json.dumps({
        "hooks": {"PreToolUse": [{
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": "python3 /opt/nl2repo/nl2repo_guard.py"}],
        }]},
    }, indent=2) + "\n")
    (dst / "environment" / "Dockerfile").write_text(f"""FROM {AGENT_IMAGE}
RUN mkdir -p /opt/nl2repo /etc/claude-code /workspace
COPY nl2repo_guard.py /opt/nl2repo/nl2repo_guard.py
COPY nl2repo-guard.json /etc/nl2repo-guard.json
COPY managed-settings.json /etc/claude-code/managed-settings.json
COPY start.md /workspace/start.md
WORKDIR /workspace
""")

    for f in ("test_commands.json", "test_files.json", "test_case_count.txt"):
        shutil.copy(src / f, dst / "tests" / "nl2repo" / f)
    shutil.copy(TEMPLATE / "grade.py", dst / "tests" / "grade.py")
    (dst / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -u\nmkdir -p /logs/verifier\n"
        "python /tests/grade.py || echo 0 > /logs/verifier/reward.txt\n"
    )
    (dst / "tests" / "Dockerfile").write_text(f"""FROM {image_for(task)}
# The official image's /workspace holds the upstream tests and packaging; the
# agent never sees it. grade.py lays it under the agent's cleaned workspace.
RUN mv /workspace /opt/nl2repo_ref && mkdir /workspace
COPY nl2repo /tests/nl2repo
COPY grade.py /tests/grade.py
COPY --chmod=755 test.sh /tests/test.sh
WORKDIR /workspace
""")

    (dst / "instruction.md").write_text(OFFICIAL_PROMPT + "\n")
    count = int((src / "test_case_count.txt").read_text().strip())
    (dst / "task.toml").write_text(f"""schema_version = "1.3"

artifacts = ["/workspace"]

[task]
name = {toml_str("nl2repobench/" + task.lower())}
description = {toml_str(f"NL2Repo-Bench {task} (upstream {UPSTREAM_COMMIT}, {count} test cases)")}
authors = []
keywords = ["nl2repo"]

[metadata]
category = "repository-generation"
upstream_task = {toml_str(task)}

[agent]
timeout_sec = 14400.0

[verifier]
timeout_sec = 3600.0
environment_mode = "separate"

[verifier.environment]
network_mode = "public"
build_timeout_sec = 3600.0
cpus = 2
memory_mb = 8192
storage_mb = 20480

[environment]
network_mode = "public"
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 8192
storage_mb = 20480
""")
    return {"task": task, "targets": targets, "test_case_count": count,
            "ref_workspace": probe["workspace"]}


def main() -> int:
    upstream, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    wanted = set(sys.argv[3:])
    out_root.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in (upstream / "test_files").iterdir() if p.is_dir())
    if wanted:
        srcs = [p for p in srcs if p.name in wanted]
        missing = wanted - {p.name for p in srcs}
        if missing:
            raise SystemExit(f"unknown tasks: {sorted(missing)}")
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for src in srcs:
        manifest[src.name] = write_task(src, out_root)
        print(json.dumps(manifest[src.name]))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

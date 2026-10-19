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
import shlex
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

# Reports, from the official image, the distribution name (packaging metadata)
# and the top-level modules the tests import that are not installed. The latter
# include the package under test but also deps that `pip install -e .` adds at
# grade time, so write_task keeps only modules named like the project.
_PROBE = r'''
import ast, importlib.util, json, pathlib, re, sys
ws = pathlib.Path("/workspace")
dist = set()
def section(text, header):
    m = re.search(r"(?ms)^\[" + re.escape(header) + r"\]\s*$(.*?)(?=^\[|\Z)", text)
    return m.group(1) if m else ""
for f, headers in (("pyproject.toml", ("project", "tool.poetry", "tool.flit.metadata")), ("setup.cfg", ("metadata",))):
    p = ws / f
    if p.exists():
        t = p.read_text(errors="replace")
        for h in headers:
            m = re.search(r"(?m)^\s*(?:name|module|dist-name)\s*[=:]\s*[\"']?([A-Za-z0-9_.\-]+)", section(t, h))
            if m:
                dist.add(m.group(1))
p = ws / "setup.py"
if p.exists():
    m = re.search(r"""\bname\s*=\s*["']([A-Za-z0-9_.\-]+)["']""", p.read_text(errors="replace"))
    if m:
        dist.add(m.group(1))
local = {q.stem for q in ws.rglob("*.py")} | {q.name for q in ws.rglob("*") if q.is_dir()}
mods = set()
for q in ws.rglob("*.py"):
    try:
        tree = ast.parse(q.read_text(errors="replace"))
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
missing = []
for m in sorted(mods):
    if m in local or m in getattr(sys, "stdlib_module_names", ()):
        continue
    try:
        found = importlib.util.find_spec(m) is not None
    except Exception:
        found = False
    if not found:
        missing.append(m)
print(json.dumps({"dist": sorted(dist), "missing_modules": missing,
                  "workspace": sorted(q.name for q in ws.iterdir()),
                  "python": "%d.%d.%d" % sys.version_info[:3]}))
'''


def _name_tokens(name: str) -> set:
    """'ydata-profiling' -> {'ydata', 'profiling'}; 'jinja2' -> {'jinja'}."""
    return {t for t in re.split(r"[^a-z]+", name.lower()) if len(t) >= 4}


def target_names(task: str, probe: dict) -> list:
    """Names whose install/fetch the guard blocks: the project itself."""
    names = {task, *probe["dist"]}
    tokens = set().union(*(_name_tokens(n) for n in names))
    # A module named after only part of the project ("flask" for Flask-RESTful)
    # is a dependency the agent may legitimately install, not the project.
    names |= {
        m for m in probe["missing_modules"]
        if _name_tokens(m) & tokens and not _name_tokens(m) < tokens
    }
    return sorted(names)


def image_for(task: str) -> str:
    return f"{IMAGE_REPO}/{task.lower()}:1.0"


def probe_image(task: str) -> dict:
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image_for(task), "-c", _PROBE],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


# Upstream runs a bare `pytest` for these, which also collects tests the agent
# wrote (reward 1.0 with no implementation, cf. PrimeIntellect-ai/prime-envs#929).
# The paths are the hidden tests in the official image; pytz keeps them at the
# workspace root, not at test_files.json's src/pytz/tests.
EXPLICIT_TEST_PATHS = {
    "flask-restful": ["tests"],
    "more-Itertools": ["tests/test_more.py", "tests/test_recipes.py"],
    "parse": ["tests"],
    "python-pytest-cases": ["tests"],
    "pytz": ["test_docs.py", "test_lazy.py", "test_tzinfo.py"],
    "uiautomator": ["test"],
}
_LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def _is_bare_pytest(command: str) -> bool:
    argv = shlex.split(command)
    if argv[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        argv = argv[2:]
    return argv[:1] == ["pytest"] and all(a.startswith("-") for a in argv[1:])


def pin_test_paths(task: str, commands: list) -> list:
    bare = [c for c in commands if _is_bare_pytest(c)]
    if bool(bare) != (task in EXPLICIT_TEST_PATHS):
        raise SystemExit(f"{task}: bare pytest commands {bare} vs EXPLICIT_TEST_PATHS")
    return [
        " ".join([c, *EXPLICIT_TEST_PATHS[task]]) if _is_bare_pytest(c) else c
        for c in commands
    ]


_INSTALLED_PROBE = r'''
import importlib.util, json, sys
sys.path[:] = [p for p in sys.path[1:] if p and not p.startswith("/workspace")]
hits = []
for n in sys.argv[1:]:
    try:
        if importlib.util.find_spec(n) is not None:
            hits.append(n)
    except Exception:
        pass
print(json.dumps(hits))
'''


def target_modules(targets: list) -> list:
    """Importable spellings of the target names: 'pytorch-grad-cam' -> 'pytorch_grad_cam'."""
    names = set()
    for t in targets:
        for n in (t, t.lower()):
            n = re.sub(r"[-.]", "_", n)
            if n.isidentifier():
                names.add(n)
    return sorted(names)


def preinstalled_targets(task: str, modules: list) -> list:
    """Target modules the official image already has installed (the real library)."""
    out = subprocess.run(
        ["docker", "run", "--rm", "-w", "/", "--entrypoint", "python", image_for(task),
         "-c", _INSTALLED_PROBE, *modules],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def read_spec(path: Path) -> bytes:
    """start.md is stored in Git LFS; a clone without git-lfs has ~130-byte pointers."""
    data = path.read_bytes()
    if data.startswith(_LFS_POINTER) or len(data) < 1024:
        raise SystemExit(f"{path} is not a spec ({len(data)} bytes): run `git lfs pull`")
    data.decode("utf-8")
    return data


def toml_str(s: str) -> str:
    return json.dumps(s)


def write_task(src: Path, out_root: Path) -> dict:
    task = src.name
    probe = probe_image(task)
    targets = target_names(task, probe)
    modules = target_modules(targets)
    preinstalled = preinstalled_targets(task, modules)
    dst = out_root / task
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "environment").mkdir(parents=True)
    (dst / "tests" / "nl2repo").mkdir(parents=True)

    (dst / "environment" / "start.md").write_bytes(read_spec(src / "start.md"))
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

    for f in ("test_files.json", "test_case_count.txt"):
        shutil.copy(src / f, dst / "tests" / "nl2repo" / f)
    upstream_commands = json.loads((src / "test_commands.json").read_text())
    commands = pin_test_paths(task, upstream_commands)
    (dst / "tests" / "nl2repo" / "test_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (dst / "tests" / "nl2repo" / "test_commands.upstream.json").write_text(
        json.dumps(upstream_commands, indent=2) + "\n"
    )
    (dst / "tests" / "nl2repo" / "target_modules.json").write_text(json.dumps(modules) + "\n")
    shutil.copy(TEMPLATE / "grade.py", dst / "tests" / "grade.py")
    shutil.copy(TEMPLATE / "drop_preinstalled.py", dst / "tests" / "drop_preinstalled.py")
    drop = f"RUN python /tests/drop_preinstalled.py {' '.join(preinstalled)}\n" if preinstalled else ""
    (dst / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -u\nmkdir -p /logs/verifier\n"
        "python /tests/grade.py || echo 0 > /logs/verifier/reward.txt\n"
    )
    (dst / "tests" / "Dockerfile").write_text(f"""FROM {image_for(task)}
# The official image's /workspace holds the upstream tests and packaging; the
# agent never sees it. grade.py lays it under the agent's cleaned workspace.
# pyautogui's image runs as a non-root user; the grader needs root.
USER root
RUN mv /workspace /opt/nl2repo_ref && mkdir /workspace
COPY drop_preinstalled.py /tests/drop_preinstalled.py
{drop}COPY nl2repo /tests/nl2repo
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
            "ref_workspace": probe["workspace"], "verifier_python": probe["python"],
            "probe": probe, "test_commands": commands,
            "preinstalled_targets": preinstalled}


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

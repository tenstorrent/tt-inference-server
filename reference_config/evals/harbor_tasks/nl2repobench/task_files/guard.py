#!/usr/bin/env python3
"""Claude Code PreToolUse hook: block Bash commands that fetch the target repository.

Reproduces the Qwen3.8 card's NL2Repo restriction ("we disable Bash commands that
attempt to access the specific repository, such as pip download, pip install, and
git clone"). Qwen did not publish its exact rules; these are ours:

- any ``pip download`` (never needed to build a repo from a spec)
- any ``git clone`` (the target or a fork of it)
- an install command (pip/uv/poetry/pipx/conda/easy_install) naming the target
  distribution, import package or repository, or a URL/VCS ref containing it
- curl/wget/http fetches of a URL that names the target

Exit code 2 blocks the call and feeds stderr back to the model.
"""

import json
import re
import shlex
import sys

CONFIG_PATH = "/etc/nl2repo-guard.json"

_INSTALL_RE = re.compile(
    r"\b(?:pip3?|python3?(?:\.\d+)?\s+-m\s+pip|uv\s+pip|uv|poetry|pipx|conda|mamba|easy_install)"
    r"\s+(?:install|add)\b"
)
_PIP_DOWNLOAD_RE = re.compile(r"\b(?:pip3?|python3?(?:\.\d+)?\s+-m\s+pip|uv\s+pip)\s+download\b")
_GIT_CLONE_RE = re.compile(r"\bgit\b(?:\s+-\S+(?:\s+\S+)?)*\s+clone\b")
_FETCH_RE = re.compile(r"\b(?:curl|wget|aria2c|httpie|http)\b|urllib|requests\.get|urlretrieve")
_URL_RE = re.compile(r"(?:https?|git|ssh)://\S+|git@\S+|\S+\.(?:whl|tar\.gz|zip|tgz)\b")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _requirement_name(token: str) -> str:
    """'Foo_Bar[extra]>=1.0' -> 'foo-bar'."""
    return _norm(re.split(r"[\[<>=!~;@ ]", token, maxsplit=1)[0])


def _segments(command: str) -> list[str]:
    return [s for s in re.split(r"&&|\|\||;|\||\n", command) if s.strip()]


def violation(command: str, targets: set[str]) -> str | None:
    lowered = command.lower()
    if _PIP_DOWNLOAD_RE.search(lowered):
        return "pip download is disabled for this task."
    if _GIT_CLONE_RE.search(lowered):
        return "git clone is disabled for this task."
    for segment in _segments(command):
        seg_lower = segment.lower()
        urls = [_norm(u) for u in _URL_RE.findall(seg_lower)]
        if any(t in u for u in urls for t in targets):
            if _INSTALL_RE.search(seg_lower) or _FETCH_RE.search(seg_lower):
                return "Fetching the target project's source is disabled for this task."
        if not _INSTALL_RE.search(seg_lower):
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        for token in tokens:
            if token.startswith("-"):
                continue
            if _requirement_name(token) in targets:
                return (
                    f"Installing the target project ({token}) is disabled for this "
                    "task; implement it from start.md instead."
                )
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if event.get("tool_name") != "Bash":
        return 0
    command = (event.get("tool_input") or {}).get("command") or ""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        targets = {_norm(t) for t in json.load(f)["targets"] if t.strip()}
    reason = violation(command, targets)
    if reason:
        print(f"Blocked: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

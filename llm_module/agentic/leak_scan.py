# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Flag agent commands that reach for a coding task's reference solution.

On repository tasks (SWE-bench Pro) a model can copy the upstream fix instead
of writing one: by reading the repository's future git history, or by fetching
the fix from a code host. ``swebench_pro`` closes both paths; this scan is the
check that they stayed closed, and it reads every trial's ATIF trajectory
(``agent/trajectory.json``, written for mini-swe-agent and Claude Code alike).

Two kinds of hit:

- ``network``: a fetch command (curl, wget, git clone/fetch/ls-remote, an HTTP
  client in Python, ...) aimed at a code host, or a package-manager fetch of
  the task's own project. ``fetched`` is ``True`` when the command's output
  shows remote content (an HTTP 2xx/3xx status, a finished clone, or a long
  output without a failure marker), so a blocked attempt reads ``False``.
- ``history``: a git command that walks refs beyond the checked-out branch
  (``git log --all``, ``reflog``, ``fsck``, ``branch --contains``, ...) or
  reads a commit by hash. Harmless on a sanitized image, which has nothing
  beyond the base commit; on any other image it can reveal the fix.

The scan flags attempts as well as successes, so a flagged trial is something
to read, not an automatic failure. Results go to ``leak_scan.json`` in the job
directory, and a summary into the job's ``result.json`` for the report.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

LEAK_SCAN_FILE = "leak_scan.json"

_CODE_HOST = re.compile(
    r"\b(?:github\.com|raw\.githubusercontent\.com|api\.github\.com|"
    r"codeload\.github\.com|objects\.githubusercontent\.com|gitlab\.com|"
    r"bitbucket\.org)\b",
    re.I,
)
_FETCH = re.compile(
    r"\bgit\s+(?:clone|fetch|pull|ls-remote|remote\s+add|submodule)\b|\bcurl\b|"
    r"\bwget\b|\burllib\b|\burlopen\b|\brequests\.(?:get|post)\b|\bhttpx\b|"
    r"\bgh\s+(?:api|pr|repo)\b",
    re.I,
)
_PACKAGE_FETCH = (
    r"\b(?:pip3?\s+(?:download|install)|(?:npm|pnpm)\s+(?:view|pack|info|show|"
    r"install|i|add)|yarn\s+(?:info|add)|go\s+(?:get|install|mod\s+download|"
    r"list\s+-m))\b[^\n;&|]*"
)
_HISTORY = re.compile(
    r"\bgit\s+(?:log|rev-list|shortlog)\b[^\n;&|]*\s--(?:all|branches|remotes|tags|"
    r"reflog)\b|\bgit\s+(?:reflog|fsck|cherry-pick)\b|"
    r"\bgit\s+(?:branch|tag)\b[^\n;&|]*\s(?:-a\b|--all\b|--contains\b)|"
    r"\bgit\s+(?:show|checkout|diff|cat-file|ls-tree|archive)\b[^\n;&|]*"
    r"\b[0-9a-f]{7,40}\b",
    re.I,
)
_FAILED = re.compile(
    r"Could not resolve|Name or service not known|Temporary failure in name "
    r"resolution|Connection refused|Connection reset|timed out|Network is "
    r"unreachable|Failed to connect|fatal: unable to access|no such host|"
    r"ENOTFOUND|EAI_AGAIN|ECONNREFUSED|ETIMEDOUT|HTTP/[\d.]+ [45]\d\d|"
    r"curl: \(\d+\)|http_code[=: ]+0{3}",
    re.I,
)
_SUCCEEDED = re.compile(
    r"HTTP/[\d.]+ [23]\d\d|\b200 OK\b|Receiving objects: +100%|"
    r"\b[0-9a-f]{40}\s+(?:HEAD|refs/)|\"sha\": \"[0-9a-f]{40}\"",
)
# Tools that only read or write local files; their arguments are not commands.
_FILE_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Grep", "Glob"}


def scan_job(job_dir: Path, task_dirs: Optional[Path] = None) -> dict[str, Any]:
    """Scan every trial under a Harbor job directory.

    *task_dirs* is the local dataset the job ran; a task's
    ``tests/config.json`` ``repo`` (``owner/name``) lets the scan also catch a
    package-manager fetch of the project itself.
    """
    flagged = []
    n_scanned = 0
    for result_file in sorted(Path(job_dir).glob("*/result.json")):
        trial_dir = result_file.parent
        trajectory = trial_dir / "agent" / "trajectory.json"
        if not trajectory.is_file():
            continue
        result = _load_json(result_file) or {}
        task_name = str(result.get("task_name") or "").split("/")[-1]
        n_scanned += 1
        hits = scan_trajectory(
            _load_json(trajectory) or {}, _project(task_dirs, task_name)
        )
        if hits:
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            flagged.append(
                {
                    "trial": trial_dir.name,
                    "task_name": task_name,
                    "reward": rewards.get("reward"),
                    "hits": hits,
                }
            )
    return {"n_trials_scanned": n_scanned, "flagged": flagged}


def scan_trajectory(
    trajectory: dict[str, Any], project: Optional[str] = None
) -> list[dict[str, Any]]:
    own_package = _own_package_pattern(project)
    hits = []
    for step in trajectory.get("steps") or []:
        for command, output in _commands(step):
            kind = None
            if _FETCH.search(command) and _CODE_HOST.search(command):
                kind = "network"
            elif own_package is not None and own_package.search(command):
                kind = "network"
            elif _HISTORY.search(command):
                kind = "history"
            if kind is None:
                continue
            hit = {
                "kind": kind,
                "step": step.get("step_id"),
                "command": " ".join(command.split())[:300],
            }
            if kind == "network":
                hit["fetched"] = _fetched(output)
            hits.append(hit)
    return hits


def write_leak_scan(
    job_dir: Path, result_file: Path, task_dirs: Optional[Path] = None
) -> dict[str, Any]:
    """Scan *job_dir*, write ``leak_scan.json`` and summarize into *result_file*."""
    report = scan_job(job_dir, task_dirs)
    with (Path(job_dir) / LEAK_SCAN_FILE).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    def trials(kind: str, *, fetched: bool = False) -> list[str]:
        return [
            t["trial"]
            for t in report["flagged"]
            if any(
                h["kind"] == kind and (not fetched or h.get("fetched"))
                for h in t["hits"]
            )
        ]

    summary = {
        "n_trials_scanned": report["n_trials_scanned"],
        "n_flagged": len(report["flagged"]),
        "network_attempts": trials("network"),
        "network_fetched": trials("network", fetched=True),
        "history_lookups": trials("history"),
    }
    for t in report["flagged"]:
        logger.warning(
            "[leak-scan] %s (reward=%s): %s",
            t["trial"],
            t["reward"],
            "; ".join(f"{h['kind']}: {h['command'][:120]}" for h in t["hits"][:3]),
        )
    logger.info(
        "[leak-scan] %d of %d trials flagged (%d fetched from a code host); "
        "details in %s",
        summary["n_flagged"],
        summary["n_trials_scanned"],
        len(summary["network_fetched"]),
        Path(job_dir) / LEAK_SCAN_FILE,
    )

    result = _load_json(result_file)
    if isinstance(result, dict):
        result["leak_scan"] = summary
        with Path(result_file).open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return summary


def _commands(step: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """(command text, its output) for each tool call in an ATIF step."""
    calls = step.get("tool_calls") or []
    results = (step.get("observation") or {}).get("results") or []
    by_id = {r.get("source_call_id"): r.get("content") for r in results}
    for i, call in enumerate(calls):
        name = call.get("function_name") or ""
        if name in _FILE_TOOLS:
            continue
        args = call.get("arguments") or {}
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str):
            command = json.dumps(args)
        output = by_id.get(call.get("tool_call_id"))
        if output is None and len(calls) == len(results):
            # mini-swe-agent results carry no call id; they come in call order.
            output = results[i].get("content")
        yield command, output if isinstance(output, str) else json.dumps(output)


def _fetched(output: str) -> bool:
    text = output[:20000]
    if _SUCCEEDED.search(text) or (
        "Cloning into" in text and not re.search(r"fatal:|error:", text)
    ):
        return True
    if _FAILED.search(text):
        return False
    return len(" ".join(text.split())) >= 200


def _own_package_pattern(project: Optional[str]) -> Optional[re.Pattern[str]]:
    if not project or "/" not in project:
        return None
    owner, name = project.split("/", 1)
    return re.compile(
        _PACKAGE_FETCH
        + rf"(?<![\w.-])(?:{re.escape(name)}|github\.com/{re.escape(owner)}/"
        + rf"{re.escape(name)})(?![\w-])",
        re.I,
    )


def _project(task_dirs: Optional[Path], task_name: str) -> Optional[str]:
    if task_dirs is None or not task_name:
        return None
    config = _load_json(Path(task_dirs) / task_name / "tests" / "config.json")
    repo = config.get("repo") if isinstance(config, dict) else None
    return repo if isinstance(repo, str) else None


def _load_json(path: Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

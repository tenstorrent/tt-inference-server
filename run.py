#!/usr/bin/env python3
"""
Usage:
  python run.py [--api-key KEY] [--mode {pr,groom}] [--run-dir DIR]
                [--max-tool-rounds N]
                [--implementer-model MODEL] [--implementer-provider PROVIDER]
                [--implementer-max-tokens N]
                [--reviewer-model MODEL] [--reviewer-provider PROVIDER]
                <repo_path> "<task description>"

Examples:
  python run.py ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --mode groom /repo "triage open issues"
  python run.py --run-dir /tmp/myrun ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --api-key sk-my-key ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --max-tool-rounds 20 ~/code/myrepo "fix this one-line typo"
  python run.py --max-tool-rounds 200 ~/code/myrepo "refactor the entire auth module"
  python run.py --implementer-model moonshotai/Kimi-K2 --implementer-provider tt-console \\
                ~/code/myrepo "fix issue #42"
  python run.py --reviewer-model anthropic/claude-sonnet-4-6 --reviewer-provider litellm \\
                ~/code/myrepo "fix issue #42"

Modes:
  pr    (default) Implement a code change, debate with reviewers, and open a
        GitHub pull request when consensus is reached.
  groom Read open issues, apply labels / comments / close duplicates via the
        gh CLI, then debate the decisions with product and technical reviewers.

API key resolution order (highest priority first):
  1. --api-key CLI argument
  2. TT_CHAT_API_KEY environment variable
  3. Key file at /workspace/global/.litellm.key

Run directory (--run-dir or ORCHESTRATOR_RUN_DIR env var):
  When set, run.py writes status.json to this directory so that external
  pollers (nohup+polling pattern) can track progress without blocking on
  the process. The file contains {"status": "running"|"done"|"failed",
  "pr_url": "...", "error": "..."}.

Max tool rounds (--max-tool-rounds):
  Hard cap on tool-call iterations per agent call.  The default (100) is
  intentionally generous because the cap is a safety rail, not a cost-control
  mechanism.  Use a lower value for simple single-file tasks and a higher
  value for large multi-file refactors.

Model/provider overrides (--implementer-* / --reviewer-*):
  Override only the model, provider, or max_tokens for the implementer or all
  reviewers.  All other persona settings (system prompt, etc.) are preserved.
  When --implementer-provider tt-console is given without --implementer-max-tokens,
  max_tokens defaults to 32768 (matching the benchmark runner convention).

Security note: passing the key via --api-key will expose it in the process
listing (e.g. `ps aux`) and in shell history. Prefer the TT_CHAT_API_KEY
environment variable or the key file for non-interactive / CI use.
"""

import sys, os, argparse, json, re, subprocess
# Use abspath so this works regardless of how the script is invoked
# (bare filename, relative path, or absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import orchestrate, orchestrate_groom, DEFAULT_MAX_TOOL_ROUNDS
from orchestrator.config import validate_provider_keys
from orchestrator.personas import ALL_PERSONAS, GROOM_REVIEWERS, GROOMER

_TT_CONSOLE_DEFAULT_MAX_TOKENS = 32768


def write_status(run_dir, status, **kwargs):
    if not run_dir:
        return
    data = {"status": status, **kwargs}
    path = os.path.join(run_dir, "status.json")
    with open(path, "w") as f:
        json.dump(data, f)


def parse_issue_number(task: str) -> int | None:
    """Return the first GitHub issue number found in *task*, or None.

    Matches the pattern ``#<digits>`` anywhere in the task string, e.g.::

        "Fix issue #18: orchestrator should self-assign"  ->  18
        "Fix issue #3: ..."                               ->  3
        "triage open issues"                              ->  None

    Only the first match is used — each run handles a single issue.

    >>> parse_issue_number("Fix issue #18: self-assign on start")
    18
    >>> parse_issue_number("triage open issues") is None
    True
    """
    m = re.search(r"#(\d+)", task)
    return int(m.group(1)) if m else None


def assign_issue_if_present(task: str, repo_path: str) -> None:
    """Self-assign the GitHub issue referenced in *task*, if any.

    Parses the first ``#<N>`` token from *task* and runs::

        gh issue edit <N> --add-assignee @me

    using ``shell=False`` (direct ``execve``) so no shell metacharacter
    quoting is needed and injection via a crafted task string is not
    possible.  The command is executed inside *repo_path* so that ``gh``
    can infer the repository from the git remote.

    If no issue number is found in *task*, this function is a no-op.

    Failures are printed as warnings but **do not abort the run** — the
    self-assignment is a best-effort visibility hint, not a correctness
    gate.  A failed assignment means the project board does not show the
    issue as in-progress, but the implementation work continues normally.
    """
    number = parse_issue_number(task)
    if number is None:
        return

    print(f"[run] self-assigning issue #{number} to @me", flush=True)
    try:
        result = subprocess.run(
            ["gh", "issue", "edit", str(number), "--add-assignee", "@me"],
            shell=False,          # no shell — argv tokens are passed verbatim
            capture_output=True,
            text=True,
            timeout=30,
            cwd=repo_path,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            print(f"[run] issue #{number} assigned: {output or 'ok'}", flush=True)
        else:
            print(
                f"[run] WARNING: could not assign issue #{number} "
                f"(exit {result.returncode}): {output}",
                flush=True,
            )
    except Exception as exc:
        print(f"[run] WARNING: assign_issue_if_present failed: {exc}", flush=True)


def build_implementer_override(args) -> dict:
    override = {}
    if args.implementer_model:
        override["model"] = args.implementer_model
    if args.implementer_provider:
        override["provider"] = args.implementer_provider
    if args.implementer_max_tokens is not None:
        override["max_tokens"] = args.implementer_max_tokens
    elif args.implementer_provider == "tt-console":
        # tt-console has a tighter context window; mirror the benchmark default.
        override["max_tokens"] = _TT_CONSOLE_DEFAULT_MAX_TOKENS
    return override


def build_reviewer_override(args) -> dict:
    override = {}
    if args.reviewer_model:
        override["model"] = args.reviewer_model
    if args.reviewer_provider:
        override["provider"] = args.reviewer_provider
    return override


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Multi-agent orchestrator: implement a task, debate, and open a PR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help=(
            "LiteLLM API key to use for all model calls. "
            "Falls back to the TT_CHAT_API_KEY environment variable, "
            "then the key file at /workspace/global/.litellm.key. "
            "WARNING: this value will be visible in process listings "
            "(e.g. `ps aux`) and in shell history; prefer the environment "
            "variable or key file for non-interactive / CI use."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["pr", "groom"],
        default="pr",
        help=(
            "Orchestration mode. "
            "'pr' (default): implement a code change and open a pull request. "
            "'groom': triage open issues (label, comment, close duplicates)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        metavar="DIR",
        default=None,
        help=(
            "Directory to write status.json into during the run. "
            "Enables the nohup+polling pattern: start the process detached and "
            "poll status.json until status != 'running'. "
            "Falls back to the ORCHESTRATOR_RUN_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--max-tool-rounds",
        metavar="N",
        type=int,
        default=DEFAULT_MAX_TOOL_ROUNDS,
        help=(
            f"Hard cap on tool-call iterations per agent call (default: {DEFAULT_MAX_TOOL_ROUNDS}). "
            "Lower this for simple single-file tasks; raise it for large "
            "multi-file refactors.  The cap is a safety rail — cost is better "
            "controlled at the token/dollar level."
        ),
    )
    parser.add_argument(
        "--implementer-model",
        metavar="MODEL",
        default=None,
        dest="implementer_model",
        help="Override the implementer persona's model (e.g. moonshotai/Kimi-K2).",
    )
    parser.add_argument(
        "--implementer-provider",
        metavar="PROVIDER",
        default=None,
        dest="implementer_provider",
        help="Override the implementer persona's provider (e.g. tt-console, litellm).",
    )
    parser.add_argument(
        "--implementer-max-tokens",
        metavar="N",
        type=int,
        default=None,
        dest="implementer_max_tokens",
        help=(
            "Override max_tokens for the implementer. "
            f"Defaults to {_TT_CONSOLE_DEFAULT_MAX_TOKENS} when --implementer-provider tt-console "
            "is set and this flag is omitted."
        ),
    )
    parser.add_argument(
        "--reviewer-model",
        metavar="MODEL",
        default=None,
        dest="reviewer_model",
        help="Override the model for all reviewer personas.",
    )
    parser.add_argument(
        "--reviewer-provider",
        metavar="PROVIDER",
        default=None,
        dest="reviewer_provider",
        help="Override the provider for all reviewer personas (e.g. tt-console, litellm).",
    )
    parser.add_argument(
        "repo_path",
        help="Path to the target git repository.",
    )
    parser.add_argument(
        "task",
        help="Natural-language description of the task to implement.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.run_dir or os.environ.get("ORCHESTRATOR_RUN_DIR")
    if run_dir:
        run_dir = os.path.abspath(run_dir)
        os.makedirs(run_dir, exist_ok=True)

    repo_path = os.path.abspath(args.repo_path)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"ERROR: {repo_path} is not a git repo")
        write_status(run_dir, "failed", error="not a git repo")
        sys.exit(1)

    write_status(run_dir, "running")

    implementer_override = build_implementer_override(args)
    reviewer_override = build_reviewer_override(args)

    # Collect the effective provider set (persona defaults + any CLI overrides)
    # so we can validate keys before any agent work begins.
    if args.mode == "groom":
        personas_in_use = [GROOMER] + GROOM_REVIEWERS
    else:
        from orchestrator.personas import IMPLEMENTER, REVIEWERS
        import copy
        effective_implementer = {**IMPLEMENTER, **implementer_override}
        effective_reviewers = [{**r, **reviewer_override} for r in REVIEWERS]
        personas_in_use = [effective_implementer] + effective_reviewers

    required_providers = {p.get("provider", "litellm") for p in personas_in_use}
    try:
        validate_provider_keys(required_providers)
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        write_status(run_dir, "failed", error=str(exc))
        sys.exit(1)

    # Poison the push URL so the implementer agent cannot push directly.
    # The orchestrator restores it immediately before its own git push.
    _poison = subprocess.run(
        ["git", "remote", "set-url", "--push", "origin", "DISABLED"],
        cwd=repo_path,
        capture_output=True,
    )
    if _poison.returncode == 0:
        print("[run] remote push URL set to DISABLED", flush=True)
    else:
        # No 'origin' remote (e.g. offline dev env) — push is already impossible.
        print("[run] WARNING: could not poison push URL (no origin remote?); "
              "push will remain as-is", flush=True)

    # Self-assign the issue before any agent work begins so the project board
    # shows this issue as in-progress.  This is best-effort; failures are
    # printed as warnings and do not abort the run.
    assign_issue_if_present(args.task, repo_path)

    try:
        if args.mode == "groom":
            success = orchestrate_groom(
                args.task,
                repo_path,
                max_debate_rounds=3,
                max_tool_rounds=args.max_tool_rounds,
                verbose=True,
                api_key=args.api_key,
            )
        else:
            success = orchestrate(
                args.task,
                repo_path,
                max_debate_rounds=3,
                max_tool_rounds=args.max_tool_rounds,
                verbose=True,
                api_key=args.api_key,
                implementer_override=implementer_override or None,
                reviewer_override=reviewer_override or None,
            )
    except Exception as e:
        write_status(run_dir, "failed", error=str(e))
        raise

    if success:
        # Extract PR URL from stdout if orchestrate() printed one
        write_status(run_dir, "done")
    else:
        write_status(run_dir, "failed", error="orchestrator returned failure")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

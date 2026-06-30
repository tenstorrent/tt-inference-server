#!/usr/bin/env python3
"""
Usage:
  python run.py [--api-key KEY] [--mode {pr,groom}] [--run-dir DIR]
                [--max-tool-rounds N] <repo_path> "<task description>"

Examples:
  python run.py ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --mode groom /repo "triage open issues"
  python run.py --run-dir /tmp/myrun ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --api-key sk-my-key ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --max-tool-rounds 20 ~/code/myrepo "fix this one-line typo"
  python run.py --max-tool-rounds 200 ~/code/myrepo "refactor the entire auth module"

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

Security note: passing the key via --api-key will expose it in the process
listing (e.g. `ps aux`) and in shell history. Prefer the TT_CHAT_API_KEY
environment variable or the key file for non-interactive / CI use.
"""

import sys, os, argparse, json
# Use abspath so this works regardless of how the script is invoked
# (bare filename, relative path, or absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import orchestrate, orchestrate_groom, DEFAULT_MAX_TOOL_ROUNDS


def write_status(run_dir, status, **kwargs):
    if not run_dir:
        return
    data = {"status": status, **kwargs}
    path = os.path.join(run_dir, "status.json")
    with open(path, "w") as f:
        json.dump(data, f)


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

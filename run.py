#!/usr/bin/env python3
"""
Usage:
  python run.py [--api-key KEY] [--mode {pr,groom}] <repo_path> "<task description>"

Examples:
  python run.py ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --mode groom /repo "triage open issues"
  python run.py --api-key sk-my-key ~/code/myrepo "add rate limiting to the /login endpoint"

Modes:
  pr    (default) Implement a code change, debate with reviewers, and open a
        GitHub pull request when consensus is reached.
  groom Read open issues, apply labels / comments / close duplicates via the
        gh CLI, then debate the decisions with product and technical reviewers.

API key resolution order (highest priority first):
  1. --api-key CLI argument
  2. TT_CHAT_API_KEY environment variable
  3. Key file at /workspace/global/.litellm.key

Security note: passing the key via --api-key will expose it in the process
listing (e.g. `ps aux`) and in shell history. Prefer the TT_CHAT_API_KEY
environment variable or the key file for non-interactive / CI use.
"""

import sys, os, argparse
# Use abspath so this works regardless of how the script is invoked
# (bare filename, relative path, or absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import orchestrate, orchestrate_groom


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

    repo_path = os.path.abspath(args.repo_path)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"ERROR: {repo_path} is not a git repo")
        sys.exit(1)

    if args.mode == "groom":
        success = orchestrate_groom(
            args.task,
            repo_path,
            max_debate_rounds=3,
            verbose=True,
            api_key=args.api_key,
        )
    else:
        # Default: pr mode
        success = orchestrate(
            args.task,
            repo_path,
            max_debate_rounds=3,
            verbose=True,
            api_key=args.api_key,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

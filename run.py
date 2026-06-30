#!/usr/bin/env python3
"""
Usage:
  python run.py [--api-key KEY] <repo_path> "<task description>"

Examples:
  python run.py ~/code/myrepo "add rate limiting to the /login endpoint"
  python run.py --api-key sk-my-key ~/code/myrepo "add rate limiting to the /login endpoint"

API key resolution order (highest priority first):
  1. --api-key CLI argument
  2. TT_CHAT_API_KEY environment variable
  3. Key file at /workspace/global/.litellm.key

Security note: passing the key via --api-key will expose it in the process
listing (e.g. `ps aux`) and in shell history. Prefer the TT_CHAT_API_KEY
environment variable or the key file for non-interactive / CI use.
"""

import sys, os, argparse, subprocess
# Use abspath so this works regardless of how the script is invoked
# (bare filename, relative path, or absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import orchestrate

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
        "repo_path",
        help="Path to the target git repository.",
    )
    parser.add_argument(
        "task",
        help="Natural-language description of the task to implement.",
    )
    return parser


# Timeout (seconds) applied to every git subprocess.  Network operations such
# as `git fetch` can stall indefinitely when a remote is unreachable; a hard
# ceiling lets the runner surface the failure quickly rather than hanging the
# whole CI job.  60 s matches the upper end of what a fetch of a single branch
# tip should ever need on a slow link.
_GIT_TIMEOUT = 60


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a git command in *cwd* and return the CompletedProcess.

    Always enforces a ``_GIT_TIMEOUT``-second wall-clock limit so that a slow
    or unreachable remote cannot hang the runner indefinitely.
    """
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )


def sync_to_latest_main(repo_path: str) -> None:
    """Fetch origin/main and reset the current branch to its tip.

    **What this does — and why it is intentionally destructive:**

    1. ``git fetch origin main`` — downloads the latest commit(s) for the
       ``main`` branch from the remote and updates the ``origin/main``
       remote-tracking ref.  Only ``main`` is contacted; other branches are
       untouched, so the fetch is fast.

    2. ``git reset --hard origin/main`` — moves *three* things atomically:

       * The **current branch pointer** (e.g. ``refs/heads/main``) to the
         fetched commit.  Any local commits that were not yet pushed are
         abandoned (they remain reachable via the reflog for 30 days but are
         no longer part of the branch history).
       * The **index** (staging area) is replaced with the tree at that commit.
       * The **working tree** is updated to match; any uncommitted local
         modifications are discarded without warning.

    This destructive behaviour is intentional: the runner operates on a
    fresh, disposable clone where there should never be local commits or
    uncommitted edits worth preserving.  The goal is to guarantee that the
    implementer agent always branches off the true latest HEAD of ``main``,
    eliminating the merge conflicts described in issue #5.

    **No-remote fast-path:**

    If the repository has no remote named ``origin`` (e.g. a local-only repo
    used during development or in tests), the sync step is skipped with a
    warning so the caller is not broken.

    Raises:
        subprocess.TimeoutExpired: propagated if a git command exceeds
            ``_GIT_TIMEOUT`` seconds, so the runner fails loudly instead of
            hanging indefinitely.
        SystemExit: if the fetch or reset fails for any reason other than a
            missing ``origin`` remote.
    """
    # Check whether an 'origin' remote exists at all.
    result = _git(["remote", "get-url", "origin"], cwd=repo_path)
    if result.returncode != 0:
        print(
            "WARNING: no 'origin' remote found — skipping fetch/reset to "
            "origin/main. Working from the current local HEAD.",
            file=sys.stderr,
        )
        return

    # Fetch only the main branch tip from origin.
    #
    # Note: passing an explicit refspec ("main") to `git fetch` makes
    # --prune a no-op — git only prunes stale remote-tracking refs when the
    # full configured refspec set is used (i.e. no explicit branch argument).
    # Pruning stale refs is not a goal here, so --prune is intentionally
    # absent; we fetch only what we need and nothing more.
    print("Fetching latest origin/main …", flush=True)
    result = _git(["fetch", "origin", "main"], cwd=repo_path)
    if result.returncode != 0:
        print(
            f"ERROR: 'git fetch origin main' failed:\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Hard-reset the current branch, index, and working tree to the fetched
    # tip.  See the docstring above for the full scope of what is discarded.
    print("Resetting to origin/main …", flush=True)
    result = _git(["reset", "--hard", "origin/main"], cwd=repo_path)
    if result.returncode != 0:
        print(
            f"ERROR: 'git reset --hard origin/main' failed:\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Log the resulting HEAD so operators can confirm which commit the
    # implementer will branch from.
    head = _git(["log", "--oneline", "-1"], cwd=repo_path)
    print(f"Repo is now at: {head.stdout.strip()}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"ERROR: {repo_path} is not a git repo")
        sys.exit(1)

    # Ensure the implementer always starts from the true latest HEAD of main.
    # Fixes issue #5: stale clones caused merge conflicts when origin/main had
    # moved forward since the bootstrap clone was taken.
    sync_to_latest_main(repo_path)

    success = orchestrate(
        args.task,
        repo_path,
        max_debate_rounds=3,
        verbose=True,
        api_key=args.api_key,  # None -> falls back to env-var / key file
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

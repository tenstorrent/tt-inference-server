#!/usr/bin/env python3
"""
Usage:
  python run.py <repo_path> "<task description>"

Example:
  python run.py ~/code/myrepo "add rate limiting to the /login endpoint"
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator import orchestrate

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    repo_path = os.path.abspath(sys.argv[1])
    task = sys.argv[2]

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"ERROR: {repo_path} is not a git repo")
        sys.exit(1)

    success = orchestrate(task, repo_path, max_debate_rounds=3, verbose=True)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

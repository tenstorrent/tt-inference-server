#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Prepare `RELEASE_BRANCH` for a release.

This helper:
- resets or creates `RELEASE_BRANCH` from `--base-branch`
- optionally applies Nightly Models CI updates to `workflows/model_spec.py`
- or regenerates pre-release outputs from manual `model_spec.py` edits
- optionally commits the generated outputs and force-pushes `RELEASE_BRANCH`
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from dispatch_release_models_ci import dispatch_release_workflow
    from release_dispatch_inputs import (
        prune_release_models_ci_config,
        resolve_release_workflow_refs,
        validate_release_models_ci_config,
    )
    from release_paths import get_versioned_release_logs_dir
except ImportError:
    from scripts.release.dispatch_release_models_ci import dispatch_release_workflow
    from scripts.release.release_dispatch_inputs import (
        prune_release_models_ci_config,
        resolve_release_workflow_refs,
        validate_release_models_ci_config,
    )
    from scripts.release.release_paths import get_versioned_release_logs_dir


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_SPEC_PATH = Path("workflows/model_spec.py")
MODELS_CI_CONFIG_PATH = Path(".github/workflows/models-ci-config.json")
PRE_RELEASE_DIFF_JSON = "pre_release_models_diff.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a release branch and generate pre-release artifacts"
    )
    parser.add_argument(
        "--base-branch",
        required=True,
        help="Base branch or commit used as the starting point for the release branch.",
    )
    parser.add_argument(
        "--release-branch",
        required=True,
        help="Release staging branch to reset or create.",
    )
    parser.add_argument(
        "--models-ci-run-id",
        type=int,
        default=None,
        help="Nightly Models CI workflow run ID used to update model_spec.py.",
    )
    parser.add_argument(
        "--tt-metal-commits",
        nargs="+",
        default=None,
        help=(
            "Only keep template diffs whose resulting tt_metal_commit exactly "
            "matches one of these values; revert other tt_metal_commit diffs to "
            "the previous release template when possible."
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit generated outputs and force-push the release branch.",
    )
    parser.add_argument(
        "--start-release-workflow",
        action="store_true",
        help="After pushing the release branch, dispatch tt-shield release.yml.",
    )
    return parser.parse_args(argv)


def run_command(
    command: Sequence[str],
    cwd: Path,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run one command and raise a readable error on failure."""
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if result.returncode != 0:
        rendered_command = shlex.join(command)
        details = (result.stderr or result.stdout or "").strip()
        if details:
            raise RuntimeError(f"Command failed: {rendered_command}\n{details}")
        raise RuntimeError(f"Command failed: {rendered_command}")
    return result


def run_git_command(
    repo_root: Path,
    *args: str,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Run one git command in the repository root."""
    return run_command(["git", *args], cwd=repo_root, capture_output=capture_output)


def git_command_succeeds(repo_root: Path, *args: str) -> bool:
    """Return True when a git command exits successfully."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def is_local_branch(repo_root: Path, ref_name: str) -> bool:
    """Return True when the ref is a local branch name."""
    return git_command_succeeds(
        repo_root, "show-ref", "--verify", "--quiet", f"refs/heads/{ref_name}"
    )


def get_current_branch(repo_root: Path) -> Optional[str]:
    """Return the current branch name or None when detached."""
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def list_changed_paths(repo_root: Path) -> List[str]:
    """Return changed and untracked paths from git status porcelain output."""
    result = run_git_command(
        repo_root, "status", "--short", "--untracked-files=all", capture_output=True
    )
    changed_paths = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        changed_paths.append(path_text)
    return changed_paths


def prepare_release_branch(repo_root: Path, base_ref: str, release_branch: str) -> None:
    """Reset the local release branch to the chosen base ref."""
    run_git_command(repo_root, "checkout", base_ref, capture_output=False)
    if is_local_branch(repo_root, base_ref):
        run_git_command(repo_root, "pull", "--ff-only", capture_output=False)
    run_git_command(
        repo_root, "branch", "-f", release_branch, base_ref, capture_output=False
    )
    run_git_command(repo_root, "checkout", release_branch, capture_output=False)


def has_manual_model_spec_changes(repo_root: Path) -> bool:
    """Return True when model_spec.py differs from HEAD."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", MODEL_SPEC_PATH.as_posix()],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    details = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(
        f"Failed to inspect manual model_spec.py changes: {details or 'git diff failed'}"
    )


def run_update_model_spec(
    repo_root: Path,
    models_ci_run_id: Optional[int],
    tt_metal_commits: Optional[Sequence[str]] = None,
) -> None:
    """Delegate model-spec generation to update_model_spec.py."""
    command = [sys.executable, str(Path(__file__).with_name("update_model_spec.py"))]
    if models_ci_run_id is None:
        command.append("--output-only")
    else:
        command.extend(["--models-ci-run-id", str(models_ci_run_id)])
    if tt_metal_commits:
        command.extend(["--tt-metal-commits", *tt_metal_commits])
    run_command(command, cwd=repo_root, capture_output=False)


def read_version(version_file: Path) -> str:
    """Read the current release version string."""
    version = version_file.read_text().strip()
    if not version:
        raise ValueError(f"VERSION file is empty: {version_file}")
    return version


def resolve_pre_release_diff_path(version: str) -> Path:
    """Return the versioned pre-release diff JSON path."""
    return get_versioned_release_logs_dir(version) / PRE_RELEASE_DIFF_JSON


def start_release_models_ci(base_ref: str, release_branch: str) -> None:
    """Dispatch the Release Models CI workflow from the pre-release diff JSON."""
    version = read_version(REPO_ROOT / "VERSION")
    release_diff_path = REPO_ROOT / resolve_pre_release_diff_path(version)
    models_ci_config_path = REPO_ROOT / MODELS_CI_CONFIG_PATH

    tt_metal_ref, vllm_ref = resolve_release_workflow_refs(release_diff_path)
    prune_release_models_ci_config(release_diff_path, models_ci_config_path)
    validate_release_models_ci_config(release_diff_path, models_ci_config_path)
    run_url = dispatch_release_workflow(
        base_ref=base_ref,
        release_branch=release_branch,
        tt_metal_ref=tt_metal_ref,
        vllm_ref=vllm_ref,
    )
    if run_url:
        print(f"Started Release Models CI workflow: {run_url}")
    else:
        print(
            "Dispatched Release Models CI workflow, but could not determine the run URL yet."
        )


def collect_pre_release_paths(version: str) -> List[Path]:
    """Return the generated pre-release paths that should be staged."""
    return [
        MODEL_SPEC_PATH,
        get_versioned_release_logs_dir(version),
    ]


def stage_pre_release_outputs(repo_root: Path, version: str) -> None:
    """Stage the expected pre-release output paths."""
    paths_to_stage = [
        path.as_posix()
        for path in collect_pre_release_paths(version)
        if (repo_root / path).exists()
    ]
    if not paths_to_stage:
        return
    run_git_command(repo_root, "add", "--", *paths_to_stage, capture_output=False)


def has_staged_changes(repo_root: Path) -> bool:
    """Return True when the index contains staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    details = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(
        f"Failed to inspect staged changes: {details or 'git diff --cached failed'}"
    )


def print_next_steps(version: str, release_branch: str) -> None:
    """Print the follow-up commands for manual commit/push."""
    paths_text = " ".join(
        path.as_posix() for path in collect_pre_release_paths(version)
    )
    print(
        "\nPre-release outputs generated. Review manual model_spec.py edits if needed, then run:"
    )
    print(f"git add -- {paths_text}")
    print(f"git commit -m 'pre-release-v{version}'")
    print(f"git push --force-with-lease origin {release_branch}")


def main() -> int:
    """Main entry point for pre-release preparation."""
    args = parse_args()
    if args.start_release_workflow and not args.commit:
        start_release_models_ci(args.base_branch, args.release_branch)
        return 0

    version = read_version(REPO_ROOT / "VERSION")
    changed_paths = list_changed_paths(REPO_ROOT)
    current_branch = get_current_branch(REPO_ROOT)

    skip_branch_setup = False
    if changed_paths:
        manual_only_change_set = {MODEL_SPEC_PATH.as_posix()}
        if (
            args.models_ci_run_id is None
            and current_branch == args.release_branch
            and set(changed_paths) == manual_only_change_set
        ):
            skip_branch_setup = True
            print(
                "Detected manual edits to workflows/model_spec.py on the release branch; "
                "skipping release branch reset."
            )
        else:
            formatted_paths = ", ".join(changed_paths)
            raise RuntimeError(
                "Working tree must be clean before resetting the release branch. "
                "Only manual workflows/model_spec.py edits are allowed when rerunning "
                f"without --models-ci-run-id on {args.release_branch}. "
                f"Current changes: {formatted_paths}"
            )

    if not skip_branch_setup:
        prepare_release_branch(REPO_ROOT, args.base_branch, args.release_branch)

    if args.models_ci_run_id is None and not has_manual_model_spec_changes(REPO_ROOT):
        raise RuntimeError(
            "No manual changes to workflows/model_spec.py were found. "
            "Edit workflows/model_spec.py on the release branch, then rerun "
            "pre_release.py to generate output-only artifacts."
        )

    run_update_model_spec(REPO_ROOT, args.models_ci_run_id, args.tt_metal_commits)

    if not args.commit:
        print_next_steps(version, args.release_branch)
        return 0

    stage_pre_release_outputs(REPO_ROOT, version)
    if has_staged_changes(REPO_ROOT):
        run_git_command(
            REPO_ROOT,
            "commit",
            "-m",
            f"pre-release-v{version}",
            capture_output=False,
        )
    else:
        print("No pre-release file changes were staged; skipping commit.")

    run_git_command(
        REPO_ROOT,
        "push",
        "--force-with-lease",
        "origin",
        args.release_branch,
        capture_output=False,
    )
    if args.start_release_workflow:
        start_release_models_ci(args.base_branch, args.release_branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

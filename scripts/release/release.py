#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Run release artifact generation and cut the versioned release branch."""

import argparse
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

try:
    from generate_release_artifacts import (
        add_cli_arguments as add_release_artifact_cli_arguments,
        configure_logging,
        get_versioned_release_logs_dir,
        run_from_args as run_release_artifacts,
        validate_args as validate_release_artifact_args,
    )
except ImportError:
    from scripts.release.generate_release_artifacts import (
        add_cli_arguments as add_release_artifact_cli_arguments,
        configure_logging,
        get_versioned_release_logs_dir,
        run_from_args as run_release_artifacts,
        validate_args as validate_release_artifact_args,
    )

from workflows.utils import get_version

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REMOTE_NAME = "origin"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the release wrapper."""
    parser = argparse.ArgumentParser(
        description="Generate release artifacts, then cut and push the release branch."
    )
    add_release_artifact_cli_arguments(
        parser,
        default_output_dir=get_versioned_release_logs_dir(),
        include_target_flags=False,
    )
    parser.add_argument(
        "--release-branch",
        required=True,
        help="Existing release staging branch that must already be checked out.",
    )
    args = parser.parse_args(argv)
    validate_release_artifact_args(parser, args, require_target_flag=False)
    return args


def run_command(
    command: Sequence[str],
    *,
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
    if result.returncode == 0:
        return result

    rendered_command = shlex.join(command)
    details = (result.stderr or result.stdout or "").strip()
    if details:
        raise RuntimeError(f"Command failed: {rendered_command}\n{details}")
    raise RuntimeError(f"Command failed: {rendered_command}")


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


def is_local_branch(repo_root: Path, ref_name: str) -> bool:
    """Return True when the ref is a local branch name."""
    return git_command_succeeds(
        repo_root, "show-ref", "--verify", "--quiet", f"refs/heads/{ref_name}"
    )


def is_remote_branch(repo_root: Path, remote_name: str, branch_name: str) -> bool:
    """Return True when the branch exists on the named remote."""
    return git_command_succeeds(
        repo_root,
        "ls-remote",
        "--exit-code",
        "--heads",
        remote_name,
        branch_name,
    )


def require_release_branch(repo_root: Path, release_branch: str) -> None:
    """Require that the wrapper already runs from the release branch."""
    current_branch = get_current_branch(repo_root)
    if current_branch == release_branch:
        return

    current_label = current_branch or "detached HEAD"
    raise RuntimeError(
        f"Expected to already be on release branch {release_branch}, found {current_label}."
    )


def build_release_artifact_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the Step 5 argument namespace in forced release mode."""
    return argparse.Namespace(
        ci_artifacts_path=args.ci_artifacts_path,
        models_ci_run_id=args.models_ci_run_id,
        out_root=args.out_root,
        dev=False,
        release=True,
        output_dir=args.output_dir,
        model_spec_path=args.model_spec_path,
        readme_path=args.readme_path,
        release_model_spec_path=args.release_model_spec_path,
        dry_run=args.dry_run,
    )


def create_and_push_release_branch(
    repo_root: Path,
    *,
    version: str,
    dry_run: bool,
    remote_name: str = REMOTE_NAME,
) -> str:
    """Create the versioned release branch and push it to the remote."""
    version_branch = f"v{version}"

    if is_local_branch(repo_root, version_branch):
        raise RuntimeError(f"Release branch already exists locally: {version_branch}")
    if is_remote_branch(repo_root, remote_name, version_branch):
        raise RuntimeError(
            f"Release branch already exists on {remote_name}: {version_branch}"
        )

    if dry_run:
        logger.info("[DRY-RUN] Would create local branch %s", version_branch)
        logger.info(
            "[DRY-RUN] Would push branch with upstream tracking: %s/%s",
            remote_name,
            version_branch,
        )
        return version_branch

    run_git_command(repo_root, "checkout", "-b", version_branch, capture_output=False)
    run_git_command(
        repo_root,
        "push",
        "-u",
        remote_name,
        version_branch,
        capture_output=False,
    )
    return version_branch


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run Step 5 artifact generation and Step 6 branch creation."""
    configure_logging()
    args = parse_args(argv)
    require_release_branch(REPO_ROOT, args.release_branch)

    logger.info("Verified current branch: %s", args.release_branch)
    logger.info("Starting Step 5 release artifact generation...")
    artifact_exit_code = run_release_artifacts(build_release_artifact_args(args))
    if artifact_exit_code != 0:
        return artifact_exit_code

    version = get_version()
    logger.info("\nStep 6: Creating and pushing release branch...")
    version_branch = create_and_push_release_branch(
        REPO_ROOT,
        version=version,
        dry_run=args.dry_run,
    )
    logger.info("Release branch ready: %s", version_branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

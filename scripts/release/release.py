#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Run the full Step-5/6/7/8 release flow and cut the versioned release branch."""

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from generate_release_artifacts import (
        add_cli_arguments as add_release_artifact_cli_arguments,
        configure_logging,
        get_versioned_release_logs_dir,
        run_from_args as run_release_artifacts,
        validate_args as validate_release_artifact_args,
    )
    from generate_release_notes import run_from_args as run_release_notes
    from release_images import run_from_args as run_release_images
    from release_paths import resolve_release_output_dir
except ImportError:
    from scripts.release.generate_release_artifacts import (
        add_cli_arguments as add_release_artifact_cli_arguments,
        configure_logging,
        get_versioned_release_logs_dir,
        run_from_args as run_release_artifacts,
        validate_args as validate_release_artifact_args,
    )
    from scripts.release.generate_release_notes import (
        run_from_args as run_release_notes,
    )
    from scripts.release.release_images import run_from_args as run_release_images
    from scripts.release.release_paths import resolve_release_output_dir

from workflows.utils import get_version

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REMOTE_NAME = "origin"
RELEASE_PERFORMANCE_PATH = Path(
    "benchmarking/benchmark_targets/release_performance.json"
)
MODEL_SUPPORT_DIR = Path("docs/model_support")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the full release wrapper."""
    parser = argparse.ArgumentParser(
        description="Run release steps 5-8 and cut the versioned release branch."
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run Step 5 and Step 6 validation only, then stop before notes/commit/branch.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Do not build missing release images locally during Step 6.",
    )
    parser.add_argument(
        "--accept-images",
        action="store_true",
        help="Skip Step-6 Enter-to-continue confirmation prompts.",
    )
    args = parser.parse_args(argv)
    validate_release_artifact_args(parser, args, require_target_flag=False)
    if args.report_data_json:
        parser.error(
            "--report-data-json is supported for Step-5 reruns only; do not use it with release.py."
        )
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
    """Build the Step-5 argument namespace in forced release mode."""
    return argparse.Namespace(
        ci_artifacts_path=args.ci_artifacts_path,
        models_ci_run_id=args.models_ci_run_id,
        report_data_json=None,
        out_root=args.out_root,
        dev=False,
        release=True,
        output_dir=args.output_dir,
        model_spec_path=args.model_spec_path,
        readme_path=args.readme_path,
        release_model_spec_path=args.release_model_spec_path,
        dry_run=args.dry_run,
    )


def build_release_images_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the Step-6 argument namespace."""
    return argparse.Namespace(
        ci_artifacts_path=args.ci_artifacts_path,
        models_ci_run_id=args.models_ci_run_id,
        out_root=args.out_root,
        output_dir=args.output_dir,
        release_model_spec_path=args.release_model_spec_path,
        readme_path=args.readme_path,
        validate_only=args.validate_only,
        no_build=args.no_build,
        accept_images=args.accept_images,
        dry_run=args.dry_run,
    )


def build_release_notes_args(
    output_dir: Path,
    version: str,
) -> argparse.Namespace:
    """Build the Step-7 argument namespace."""
    return argparse.Namespace(
        version=version,
        version_file="VERSION",
        artifacts_summary_json=str(output_dir / "release_artifacts_summary.json"),
        model_diff_json=str(output_dir / "pre_release_models_diff.json"),
        release_performance_json=str(REPO_ROOT / RELEASE_PERFORMANCE_PATH),
        base_release_performance_json=None,
        output=str(output_dir / f"release_notes_v{version}.md"),
    )


def _repo_relative_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _normalize_repo_path(path_like: Path) -> str:
    resolved_path = path_like.resolve()
    try:
        return str(resolved_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path_like)


def get_allowed_release_paths(
    *,
    output_dir: Path,
    version: str,
    release_model_spec_path: Path,
    readme_path: Path,
) -> Set[str]:
    """Return the repo-relative files and directories the wrapper is allowed to touch."""
    output_dir_rel = _repo_relative_path(output_dir)
    return {
        _normalize_repo_path(release_model_spec_path),
        _normalize_repo_path(readme_path),
        str(RELEASE_PERFORMANCE_PATH),
        str(MODEL_SUPPORT_DIR),
        f"{output_dir_rel}/release_acceptance_warnings.json",
        f"{output_dir_rel}/release_performance_diff.json",
        f"{output_dir_rel}/release_artifacts_summary.json",
        f"{output_dir_rel}/release_artifacts_summary.md",
        f"{output_dir_rel}/release_notes_v{version}.md",
    }


def list_worktree_changes(repo_root: Path) -> List[str]:
    """Return repo-relative changed paths from git status porcelain."""
    status_output = run_git_command(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout.splitlines()
    changed_paths: List[str] = []
    for line in status_output:
        if not line:
            continue
        path_field = line[3:]
        if " -> " in path_field:
            path_field = path_field.split(" -> ", 1)[1]
        changed_paths.append(path_field)
    return changed_paths


def _is_allowed_release_path(path_str: str, allowed_paths: Iterable[str]) -> bool:
    for allowed_path in allowed_paths:
        normalized_allowed = allowed_path.rstrip("/")
        if path_str == normalized_allowed or path_str.startswith(
            normalized_allowed + "/"
        ):
            return True
    return False


def require_only_allowed_changes(
    repo_root: Path,
    *,
    allowed_paths: Set[str],
    context: str,
) -> None:
    """Reject unexpected worktree changes before staging and committing release outputs."""
    unexpected_changes = sorted(
        path
        for path in list_worktree_changes(repo_root)
        if not _is_allowed_release_path(path, allowed_paths)
    )
    if unexpected_changes:
        raise RuntimeError(
            f"Unexpected worktree changes {context}: " + ", ".join(unexpected_changes)
        )


def stage_release_outputs(
    repo_root: Path,
    *,
    allowed_paths: Set[str],
    dry_run: bool,
) -> None:
    """Stage the deterministic release outputs."""
    stage_paths = sorted(allowed_paths)
    if dry_run:
        logger.info("[DRY-RUN] Would stage release outputs: %s", ", ".join(stage_paths))
        return
    run_git_command(repo_root, "add", "--", *stage_paths, capture_output=False)


def has_staged_changes(repo_root: Path) -> bool:
    """Return True when the index contains staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0


def commit_release_outputs(repo_root: Path, *, version: str, dry_run: bool) -> None:
    """Create the release commit if staged changes are present."""
    if not has_staged_changes(repo_root):
        logger.info("No staged release output changes detected; skipping commit.")
        return
    if dry_run:
        logger.info("[DRY-RUN] Would commit release-v%s", version)
        return
    run_git_command(
        repo_root,
        "commit",
        "-m",
        f"release-v{version}",
        capture_output=False,
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
    """Run Step 5 artifact generation, Step 6 image handling, Step 7 notes, and Step 8 branch creation."""
    configure_logging()
    args = parse_args(argv)
    version = get_version()
    output_dir = resolve_release_output_dir(args.output_dir)
    allowed_paths = get_allowed_release_paths(
        output_dir=output_dir,
        version=version,
        release_model_spec_path=Path(args.release_model_spec_path),
        readme_path=Path(args.readme_path),
    )

    require_release_branch(REPO_ROOT, args.release_branch)
    require_only_allowed_changes(
        REPO_ROOT,
        allowed_paths=allowed_paths,
        context="before running the release wrapper",
    )

    logger.info("Verified current branch: %s", args.release_branch)
    logger.info("Starting Step 5 release artifact generation...")
    artifact_exit_code = run_release_artifacts(build_release_artifact_args(args))
    if artifact_exit_code != 0:
        return artifact_exit_code

    logger.info("\nStep 6: Handling release Docker images...")
    image_exit_code = run_release_images(build_release_images_args(args))
    if image_exit_code != 0:
        return image_exit_code

    if args.validate_only:
        logger.info("Validate-only mode completed after Step 6.")
        return 0

    logger.info("\nStep 7: Generating release notes...")
    notes_exit_code = run_release_notes(build_release_notes_args(output_dir, version))
    if notes_exit_code != 0:
        return notes_exit_code

    require_only_allowed_changes(
        REPO_ROOT,
        allowed_paths=allowed_paths,
        context="after generating release outputs",
    )

    logger.info("\nStep 8: Committing release outputs...")
    stage_release_outputs(REPO_ROOT, allowed_paths=allowed_paths, dry_run=args.dry_run)
    commit_release_outputs(REPO_ROOT, version=version, dry_run=args.dry_run)

    logger.info("\nCreating and pushing release branch...")
    version_branch = create_and_push_release_branch(
        REPO_ROOT,
        version=version,
        dry_run=args.dry_run,
    )
    logger.info("Release branch ready: %s", version_branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import logging
from pathlib import Path
import subprocess
import jwt

logger = logging.getLogger(__name__)


def create_model_symlink(symlinks_dir, model_name, weights_dir, file_symlinks_map={}):
    """Helper function to create and manage model symlinks.

    Args:
        symlinks_dir: Directory to store symlinks
        model_name: Model name to use for the symlink
        weights_dir: Path to the model weights
        file_symlinks_map: Dict of {target_file: source_file} for creating file-specific symlinks

    Returns:
        Path to the created symlink or directory
    """
    symlink_path = symlinks_dir / model_name

    # Handle file-specific symlinks (for vision models)
    if file_symlinks_map:
        # Clean up any existing symlinks
        if symlink_path.exists():
            for _link in symlink_path.iterdir():
                if _link.is_symlink():
                    _link.unlink()
        symlink_path.mkdir(parents=True, exist_ok=True)

        # Create individual file symlinks
        for target_file, source_file in file_symlinks_map.items():
            (symlink_path / target_file).symlink_to(weights_dir / source_file)

        return symlink_path

    # Handle single directory/file symlink (standard case)
    if symlink_path.is_symlink():
        symlink_path.unlink()
    assert not symlink_path.exists(), (
        f"symlink location: {symlink_path} has a non-symlink there."
    )
    symlink_path.symlink_to(weights_dir)
    return symlink_path


def get_encoded_api_key(jwt_secret):
    if jwt_secret is None:
        return None
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def resolve_commit(commit: str, repo_path: Path) -> str:
    repo_path = Path(repo_path)
    assert repo_path.is_dir(), f"The path '{repo_path}' is not a valid directory."
    result = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stdout.strip() not in ["commit", "tag"]:
        logger.warning(f"Commit '{commit}' is not in the repository.")
        return None

    result = subprocess.run(
        ["git", "rev-parse", commit],
        cwd=str(repo_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def is_head_eq_or_after_commit(commit: str, repo_path: str = ".") -> bool:
    """
    Checks if the current HEAD of repo_path is the same as or a descendant (i.e., comes after) commit in the git history.

    Args:
        commit (str): The SHA or tag of the commit to compare with HEAD.
        repo_path (str): The path to the git repository (default is current directory).

    Returns:
        bool: True if commit is the same or newer than HEAD, False otherwise.
    """
    repo = Path(repo_path)
    assert repo.is_dir(), f"The path '{repo}' is not a valid directory."

    try:
        # Resolve full commit hashes in case they are tags or shortened SHAs
        commit_full = resolve_commit(commit, repo)
        head_commit = resolve_commit("HEAD", repo)

        if not commit_full:
            logger.warning(
                f"Commit '{commit}' was not resolved. Assuming it is an external or future commit."
            )
            return False  # If the commit is unknown, assume it is newer than HEAD

        # If the commits are the same, return True
        if commit_full == head_commit:
            return True

        # Run the git command to check ancestry
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_full, head_commit],
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # A return code of 0 means commit is an ancestor of HEAD
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Unexpected error while checking commit ancestry: {e}")
        return False

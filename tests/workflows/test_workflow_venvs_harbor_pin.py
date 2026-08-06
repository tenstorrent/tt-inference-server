#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for the pinned Harbor checkout used by the agentic-eval venv.

The agentic harness runs a Tenstorrent fork of Harbor rather than a release
from PyPI, so the revision CI installs is decided here rather than by a
version specifier. Two properties matter and neither is free:

* the checkout lands on an exact commit, so a result can be traced back to
  the code that produced it, and a force-push cannot change what CI runs
  without a diff in this repo;
* it converges from whatever an earlier job left behind. Self-hosted
  runners keep ``.workflow_venvs/`` between jobs, so "clone only when the
  directory is missing" silently keeps installing a stale revision after
  the pin is bumped -- which looks exactly like a fix that did not work.

These use real git repositories in a temp dir: the whole point is the git
behaviour, and mocking the commands would only assert that the strings were
formatted the way the implementation happens to format them.
"""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflows.workflow_venvs import checkout_pinned_repo


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """A two-commit repository standing in for the Harbor fork."""
    repo = tmp_path / "origin"
    repo.mkdir()
    _git("init", "--initial-branch=main", cwd=repo)
    _git("config", "user.email", "ci@example.com", cwd=repo)
    _git("config", "user.name", "CI", cwd=repo)

    (repo / "marker.txt").write_text("first\n")
    _git("add", "marker.txt", cwd=repo)
    _git("commit", "-m", "first", cwd=repo)

    (repo / "marker.txt").write_text("second\n")
    _git("add", "marker.txt", cwd=repo)
    _git("commit", "-m", "second", cwd=repo)
    return repo


@pytest.fixture
def commits(origin: Path) -> tuple[str, str]:
    """``(first, second)`` commit SHAs of the origin fixture, oldest first."""
    log = _git("log", "--format=%H", cwd=origin).splitlines()
    return log[1], log[0]


def test_checkout_lands_on_the_pinned_commit(tmp_path, origin, commits):
    first, _second = commits
    dest = tmp_path / "harbor"

    assert checkout_pinned_repo(dest, str(origin), first) is True

    assert _git("rev-parse", "HEAD", cwd=dest) == first
    assert (dest / "marker.txt").read_text() == "first\n"


def test_checkout_moves_a_stale_existing_checkout_to_the_new_pin(
    tmp_path, origin, commits
):
    # The regression this file exists for: the directory survives from an
    # earlier job at the previous pin, and the pin has since been bumped.
    first, second = commits
    dest = tmp_path / "harbor"

    assert checkout_pinned_repo(dest, str(origin), first) is True
    assert _git("rev-parse", "HEAD", cwd=dest) == first

    assert checkout_pinned_repo(dest, str(origin), second) is True

    assert _git("rev-parse", "HEAD", cwd=dest) == second
    assert (dest / "marker.txt").read_text() == "second\n"


def test_checkout_is_idempotent_at_the_same_pin(tmp_path, origin, commits):
    _first, second = commits
    dest = tmp_path / "harbor"

    assert checkout_pinned_repo(dest, str(origin), second) is True
    assert checkout_pinned_repo(dest, str(origin), second) is True

    assert _git("rev-parse", "HEAD", cwd=dest) == second


def test_checkout_replaces_a_directory_that_is_not_a_git_repo(
    tmp_path, origin, commits
):
    # A killed job can leave a half-written directory behind. Reusing it
    # would fail every later git command in a way that reads like a network
    # problem, so it is discarded rather than repaired.
    _first, second = commits
    dest = tmp_path / "harbor"
    dest.mkdir()
    (dest / "leftover.txt").write_text("junk\n")

    assert checkout_pinned_repo(dest, str(origin), second) is True

    assert _git("rev-parse", "HEAD", cwd=dest) == second
    assert not (dest / "leftover.txt").exists()


def test_checkout_reports_failure_for_a_ref_that_does_not_exist(tmp_path, origin):
    dest = tmp_path / "harbor"

    assert checkout_pinned_repo(dest, str(origin), "0" * 40) is False, (
        "a bad pin must be reported, not silently left at some other revision"
    )


def test_the_configured_harbor_pin_is_a_full_commit_sha():
    # A branch name or short SHA here would reintroduce the ambiguity the
    # rest of this file guards against, and neither is rejected by git.
    from workflows.workflow_venvs import HARBOR_REF

    assert len(HARBOR_REF) == 40, f"{HARBOR_REF!r} is not a full commit SHA"
    assert all(c in "0123456789abcdef" for c in HARBOR_REF), (
        f"{HARBOR_REF!r} is not a full commit SHA"
    )

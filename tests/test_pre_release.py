from argparse import Namespace
from pathlib import Path
from unittest.mock import call, patch

import pytest

import scripts.release.pre_release as pr


def test_prepare_release_branch_pulls_local_branch_before_resetting_release_branch():
    repo_root = Path("/tmp/repo")

    with patch.object(pr, "run_git_command") as git_mock, patch.object(
        pr, "is_local_branch", return_value=True
    ):
        pr.prepare_release_branch(repo_root, "main", "stable")

    assert git_mock.call_args_list == [
        call(repo_root, "checkout", "main", capture_output=False),
        call(repo_root, "pull", "--ff-only", capture_output=False),
        call(repo_root, "branch", "-f", "stable", "main", capture_output=False),
        call(repo_root, "checkout", "stable", capture_output=False),
    ]


def test_prepare_release_branch_skips_pull_for_fixed_ref():
    repo_root = Path("/tmp/repo")

    with patch.object(pr, "run_git_command") as git_mock, patch.object(
        pr, "is_local_branch", return_value=False
    ):
        pr.prepare_release_branch(repo_root, "8f3e2d1", "stable")

    assert git_mock.call_args_list == [
        call(repo_root, "checkout", "8f3e2d1", capture_output=False),
        call(repo_root, "branch", "-f", "stable", "8f3e2d1", capture_output=False),
        call(repo_root, "checkout", "stable", capture_output=False),
    ]


def test_main_skips_branch_reset_for_manual_model_spec_rerun():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=None,
        commit=False,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(
        pr, "list_changed_paths", return_value=["workflows/model_spec.py"]
    ), patch.object(pr, "get_current_branch", return_value="stable"), patch.object(
        pr, "prepare_release_branch"
    ) as prepare_mock, patch.object(
        pr, "has_manual_model_spec_changes", return_value=True
    ), patch.object(pr, "run_update_model_spec") as update_mock, patch.object(
        pr, "print_next_steps"
    ) as next_steps_mock:
        assert pr.main() == 0

    prepare_mock.assert_not_called()
    update_mock.assert_called_once_with(pr.REPO_ROOT, None)
    next_steps_mock.assert_called_once_with("0.9.0", "stable")


def test_main_requires_clean_worktree_before_branch_reset():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        commit=False,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=["README.md"]), patch.object(
        pr, "get_current_branch", return_value="main"
    ):
        with pytest.raises(RuntimeError, match="Working tree must be clean"):
            pr.main()


def test_main_manual_mode_requires_existing_model_spec_edits():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=None,
        commit=False,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=[]), patch.object(
        pr, "get_current_branch", return_value="main"
    ), patch.object(pr, "prepare_release_branch") as prepare_mock, patch.object(
        pr, "has_manual_model_spec_changes", return_value=False
    ):
        with pytest.raises(
            RuntimeError, match="No manual changes to workflows/model_spec.py"
        ):
            pr.main()

    prepare_mock.assert_called_once_with(pr.REPO_ROOT, "main", "stable")


def test_main_commit_flow_stages_commits_and_pushes():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        commit=True,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=[]), patch.object(
        pr, "get_current_branch", return_value="main"
    ), patch.object(pr, "prepare_release_branch") as prepare_mock, patch.object(
        pr, "run_update_model_spec"
    ) as update_mock, patch.object(
        pr, "stage_pre_release_outputs"
    ) as stage_mock, patch.object(
        pr, "has_staged_changes", return_value=True
    ), patch.object(pr, "run_git_command") as git_mock:
        assert pr.main() == 0

    prepare_mock.assert_called_once_with(pr.REPO_ROOT, "main", "stable")
    update_mock.assert_called_once_with(pr.REPO_ROOT, 123)
    stage_mock.assert_called_once_with(pr.REPO_ROOT, "0.9.0")
    assert git_mock.call_args_list == [
        call(
            pr.REPO_ROOT,
            "commit",
            "-m",
            "pre-release-v0.9.0",
            capture_output=False,
        ),
        call(
            pr.REPO_ROOT,
            "push",
            "--force-with-lease",
            "origin",
            "stable",
            capture_output=False,
        ),
    ]

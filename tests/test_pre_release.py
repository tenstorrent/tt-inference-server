import sys
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
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=False,
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
    update_mock.assert_called_once_with(pr.REPO_ROOT, None, None)
    next_steps_mock.assert_called_once_with("0.9.0", "stable")


def test_main_requires_clean_worktree_before_branch_reset():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=False,
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
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=False,
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
        tt_metal_commits=None,
        commit=True,
        start_release_workflow=False,
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
    update_mock.assert_called_once_with(pr.REPO_ROOT, 123, None)
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


def test_main_does_not_require_github_token_without_release_workflow():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=False,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=[]), patch.object(
        pr, "get_current_branch", return_value="main"
    ), patch.object(pr, "prepare_release_branch"), patch.object(
        pr, "run_update_model_spec"
    ), patch.object(pr, "print_next_steps"), patch.object(
        pr, "get_github_token"
    ) as token_mock:
        assert pr.main() == 0

    token_mock.assert_not_called()


def test_parse_args_allows_start_release_workflow_without_commit():
    args = pr.parse_args(
        [
            "--base-branch",
            "main",
            "--release-branch",
            "stable",
            "--start-release-workflow",
        ]
    )

    assert args.start_release_workflow is True
    assert args.commit is False
    assert args.tt_metal_commits is None


def test_parse_args_accepts_tt_metal_commits():
    args = pr.parse_args(
        [
            "--base-branch",
            "main",
            "--release-branch",
            "stable",
            "--tt-metal-commits",
            "abc1234",
            "def5678",
        ]
    )

    assert args.tt_metal_commits == ["abc1234", "def5678"]


@pytest.mark.parametrize(
    ("models_ci_run_id", "tt_metal_commits", "expected_command"),
    [
        (
            None,
            None,
            [
                sys.executable,
                str(Path(pr.__file__).with_name("update_model_spec.py")),
                "--output-only",
            ],
        ),
        (
            None,
            ["abc1234"],
            [
                sys.executable,
                str(Path(pr.__file__).with_name("update_model_spec.py")),
                "--output-only",
                "--tt-metal-commits",
                "abc1234",
            ],
        ),
        (
            123,
            ["abc1234", "def5678"],
            [
                sys.executable,
                str(Path(pr.__file__).with_name("update_model_spec.py")),
                "--models-ci-run-id",
                "123",
                "--tt-metal-commits",
                "abc1234",
                "def5678",
            ],
        ),
    ],
)
def test_run_update_model_spec_forwards_tt_metal_commits(
    models_ci_run_id, tt_metal_commits, expected_command
):
    repo_root = Path("/tmp/repo")

    with patch.object(pr, "run_command") as run_command_mock:
        pr.run_update_model_spec(repo_root, models_ci_run_id, tt_metal_commits)

    run_command_mock.assert_called_once_with(
        expected_command, cwd=repo_root, capture_output=False
    )


def test_main_dispatch_only_mode_skips_generation_and_git_writes():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=None,
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=True,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ) as read_version_mock, patch.object(
        pr, "get_github_token", return_value="token"
    ) as token_mock, patch.object(
        pr, "resolve_release_workflow_refs", return_value=("metal-sha", "vllm-sha")
    ) as refs_mock, patch.object(
        pr, "start_release_models_ci"
    ) as dispatch_mock, patch.object(
        pr, "prepare_release_branch"
    ) as prepare_mock, patch.object(
        pr, "run_update_model_spec"
    ) as update_mock, patch.object(pr, "run_git_command") as git_mock:
        assert pr.main() == 0

    token_mock.assert_called_once_with()
    refs_mock.assert_called_once_with(
        pr.REPO_ROOT
        / pr.get_versioned_release_logs_dir("0.9.0")
        / pr.PRE_RELEASE_DIFF_JSON
    )
    dispatch_mock.assert_called_once_with("stable")
    read_version_mock.assert_called_once_with(pr.REPO_ROOT / "VERSION")
    prepare_mock.assert_not_called()
    update_mock.assert_not_called()
    git_mock.assert_not_called()


def test_main_release_workflow_requires_github_token():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=None,
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=True,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(
        pr, "get_github_token", side_effect=RuntimeError("GH_PAT is required")
    ), patch.object(pr, "start_release_models_ci") as dispatch_mock:
        with pytest.raises(RuntimeError, match="GH_PAT is required"):
            pr.main()

    dispatch_mock.assert_not_called()


@pytest.mark.parametrize(
    "dispatch_error",
    [
        FileNotFoundError("Pre-release diff JSON not found"),
        ValueError("contains multiple tt_metal_commit_after values"),
    ],
)
def test_main_dispatch_only_mode_fails_preconditions_before_dispatch(dispatch_error):
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=None,
        tt_metal_commits=None,
        commit=False,
        start_release_workflow=True,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "get_github_token", return_value="token"), patch.object(
        pr, "resolve_release_workflow_refs", side_effect=dispatch_error
    ), patch.object(pr, "start_release_models_ci") as dispatch_mock:
        with pytest.raises(RuntimeError, match="Release workflow preconditions failed"):
            pr.main()

    dispatch_mock.assert_not_called()


def test_start_release_models_ci_uses_versioned_diff_and_models_ci_config():
    expected_diff_path = (
        pr.REPO_ROOT
        / pr.get_versioned_release_logs_dir("0.9.0")
        / pr.PRE_RELEASE_DIFF_JSON
    )
    expected_models_ci_config_path = pr.REPO_ROOT / pr.MODELS_CI_CONFIG_PATH

    with patch.object(pr, "read_version", return_value="0.9.0"), patch.object(
        pr,
        "resolve_release_workflow_inputs",
        return_value=(expected_diff_path, "metal-sha", "vllm-sha"),
    ) as inputs_mock, patch.object(
        pr,
        "prune_release_models_ci_config",
        return_value={"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]},
    ) as prune_mock, patch.object(
        pr,
        "validate_release_models_ci_config",
        return_value={"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]},
    ) as validate_mock, patch.object(
        pr,
        "dispatch_release_workflow",
        return_value="https://github.com/run/123",
    ) as dispatch_mock:
        pr.start_release_models_ci("stable")

    inputs_mock.assert_called_once_with("0.9.0")
    prune_mock.assert_called_once_with(
        expected_diff_path, expected_models_ci_config_path
    )
    validate_mock.assert_called_once_with(
        expected_diff_path, expected_models_ci_config_path
    )
    dispatch_mock.assert_called_once_with(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
        run_ai_summary=False,
    )


def test_resolve_pre_release_diff_path_returns_versioned_json_path():
    assert pr.resolve_pre_release_diff_path("0.9.0") == (
        pr.get_versioned_release_logs_dir("0.9.0") / pr.PRE_RELEASE_DIFF_JSON
    )


def test_start_release_models_ci_uses_versioned_diff_path():
    with patch.object(
        pr,
        "read_version",
        return_value="0.9.0",
    ), patch.object(
        pr,
        "resolve_release_workflow_inputs",
        return_value=(
            pr.REPO_ROOT
            / pr.get_versioned_release_logs_dir("0.9.0")
            / pr.PRE_RELEASE_DIFF_JSON,
            "metal-sha",
            "vllm-sha",
        ),
    ) as inputs_mock, patch.object(
        pr,
        "prune_release_models_ci_config",
        return_value={"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]},
    ), patch.object(
        pr,
        "validate_release_models_ci_config",
        return_value={"Llama-3.2-1B-Instruct": ["N150", "N300", "T3K"]},
    ), patch.object(
        pr,
        "dispatch_release_workflow",
        return_value="https://github.com/run/123",
    ) as dispatch_mock:
        pr.start_release_models_ci("stable")

    inputs_mock.assert_called_once_with("0.9.0")
    dispatch_mock.assert_called_once_with(
        release_branch="stable",
        tt_metal_ref="metal-sha",
        vllm_ref="vllm-sha",
        run_ai_summary=False,
    )


def test_main_dispatches_release_workflow_after_push():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        tt_metal_commits=None,
        commit=True,
        start_release_workflow=True,
    )
    event_order = []

    def fake_run_git_command(repo_root, *git_args, **kwargs):
        if git_args and git_args[0] == "push":
            event_order.append("push")

    def fake_start_release_models_ci(release_branch):
        assert release_branch == "stable"
        event_order.append("dispatch")

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=[]), patch.object(
        pr, "get_current_branch", return_value="main"
    ), patch.object(pr, "get_github_token", return_value="token"), patch.object(
        pr, "resolve_release_workflow_refs", return_value=("metal-sha", "vllm-sha")
    ), patch.object(pr, "prepare_release_branch"), patch.object(
        pr, "run_update_model_spec"
    ), patch.object(pr, "stage_pre_release_outputs"), patch.object(
        pr, "has_staged_changes", return_value=True
    ), patch.object(
        pr, "run_git_command", side_effect=fake_run_git_command
    ), patch.object(
        pr, "start_release_models_ci", side_effect=fake_start_release_models_ci
    ) as dispatch_mock:
        assert pr.main() == 0

    assert event_order == ["push", "dispatch"]
    dispatch_mock.assert_called_once_with("stable")


def test_main_commit_flow_validates_dispatch_inputs_before_git_writes():
    args = Namespace(
        base_branch="main",
        release_branch="stable",
        models_ci_run_id=123,
        tt_metal_commits=None,
        commit=True,
        start_release_workflow=True,
    )

    with patch.object(pr, "parse_args", return_value=args), patch.object(
        pr, "read_version", return_value="0.9.0"
    ), patch.object(pr, "list_changed_paths", return_value=[]), patch.object(
        pr, "get_current_branch", return_value="main"
    ), patch.object(pr, "get_github_token", return_value="token"), patch.object(
        pr, "prepare_release_branch"
    ), patch.object(pr, "run_update_model_spec") as update_mock, patch.object(
        pr,
        "resolve_release_workflow_refs",
        side_effect=ValueError("contains multiple tt_metal_commit_after values"),
    ), patch.object(pr, "stage_pre_release_outputs") as stage_mock, patch.object(
        pr, "run_git_command"
    ) as git_mock, patch.object(pr, "start_release_models_ci") as dispatch_mock:
        with pytest.raises(RuntimeError, match="Release workflow preconditions failed"):
            pr.main()

    update_mock.assert_called_once_with(pr.REPO_ROOT, 123, None)
    stage_mock.assert_not_called()
    git_mock.assert_not_called()
    dispatch_mock.assert_not_called()

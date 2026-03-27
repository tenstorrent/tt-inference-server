from argparse import Namespace
from pathlib import Path
from unittest.mock import call, patch

import pytest

import scripts.release.release as rel


def make_args(**overrides):
    args = {
        "ci_artifacts_path": None,
        "models_ci_run_id": 23578993514,
        "report_data_json": None,
        "out_root": None,
        "output_dir": "release_logs/v0.10.0",
        "model_spec_path": "workflows/model_spec.py",
        "readme_path": "README.md",
        "release_model_spec_path": "release_model_spec.json",
        "dry_run": False,
        "release_branch": "stable",
    }
    args.update(overrides)
    return Namespace(**args)


def test_parse_args_accepts_release_branch_and_step5_inputs():
    args = rel.parse_args(
        [
            "--models-ci-run-id",
            "23578993514",
            "--release-branch",
            "stable",
        ]
    )

    assert args.models_ci_run_id == 23578993514
    assert args.ci_artifacts_path is None
    assert args.report_data_json is None
    assert args.release_branch == "stable"
    assert args.dry_run is False


def test_build_release_artifact_args_preserves_report_data_json():
    release_args = rel.build_release_artifact_args(
        make_args(
            models_ci_run_id=None, report_data_json="release_logs/v0.10.0/report.json"
        )
    )

    assert release_args.report_data_json == "release_logs/v0.10.0/report.json"
    assert release_args.release is True
    assert release_args.dev is False


def test_main_requires_current_release_branch():
    args = make_args(release_branch="stable")

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="main"), patch.object(
        rel, "run_release_artifacts"
    ) as release_mock:
        with pytest.raises(
            RuntimeError, match="Expected to already be on release branch stable"
        ):
            rel.main()

    release_mock.assert_not_called()


def test_create_and_push_release_branch_dry_run_skips_git_mutation():
    repo_root = Path("/tmp/repo")

    with patch.object(rel, "is_local_branch", return_value=False), patch.object(
        rel, "is_remote_branch", return_value=False
    ), patch.object(rel, "run_git_command") as git_mock:
        branch_name = rel.create_and_push_release_branch(
            repo_root,
            version="0.10.0",
            dry_run=True,
        )

    assert branch_name == "v0.10.0"
    git_mock.assert_not_called()


def test_main_runs_step5_then_creates_and_pushes_release_branch():
    args = make_args(
        ci_artifacts_path="release_logs/v0.10.0",
        models_ci_run_id=None,
        release_branch="stable",
    )

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="stable"), patch.object(
        rel, "run_release_artifacts", return_value=0
    ) as artifacts_mock, patch.object(
        rel, "get_version", return_value="0.10.0"
    ), patch.object(rel, "is_local_branch", return_value=False), patch.object(
        rel, "is_remote_branch", return_value=False
    ), patch.object(rel, "run_git_command") as git_mock:
        assert rel.main() == 0

    release_args = artifacts_mock.call_args.args[0]
    assert release_args.ci_artifacts_path == "release_logs/v0.10.0"
    assert release_args.models_ci_run_id is None
    assert release_args.report_data_json is None
    assert release_args.dev is False
    assert release_args.release is True
    assert release_args.release_model_spec_path == "release_model_spec.json"
    assert git_mock.call_args_list == [
        call(rel.REPO_ROOT, "checkout", "-b", "v0.10.0", capture_output=False),
        call(
            rel.REPO_ROOT,
            "push",
            "-u",
            "origin",
            "v0.10.0",
            capture_output=False,
        ),
    ]


def test_main_does_not_create_release_branch_when_step5_fails():
    args = make_args(release_branch="stable")

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="stable"), patch.object(
        rel, "run_release_artifacts", return_value=1
    ), patch.object(rel, "run_git_command") as git_mock, patch.object(
        rel, "get_version"
    ) as version_mock:
        assert rel.main() == 1

    git_mock.assert_not_called()
    version_mock.assert_not_called()

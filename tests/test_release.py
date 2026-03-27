from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

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
        "validate_only": False,
        "no_build": False,
        "accept_images": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_parse_args_accepts_release_branch_and_step_inputs():
    args = rel.parse_args(
        [
            "--models-ci-run-id",
            "23578993514",
            "--release-branch",
            "stable",
            "--no-build",
            "--accept-images",
        ]
    )

    assert args.models_ci_run_id == 23578993514
    assert args.ci_artifacts_path is None
    assert args.report_data_json is None
    assert args.release_branch == "stable"
    assert args.no_build is True
    assert args.accept_images is True
    assert args.validate_only is False


def test_parse_args_rejects_report_data_json():
    with pytest.raises(SystemExit):
        rel.parse_args(
            [
                "--report-data-json",
                "release_logs/v0.10.0/report.json",
                "--release-branch",
                "stable",
            ]
        )


def test_build_release_artifact_args_forces_release_mode():
    release_args = rel.build_release_artifact_args(make_args())

    assert release_args.report_data_json is None
    assert release_args.release is True
    assert release_args.dev is False


def test_build_release_images_args_preserves_step6_flags():
    image_args = rel.build_release_images_args(
        make_args(no_build=True, validate_only=True, accept_images=True)
    )

    assert image_args.no_build is True
    assert image_args.validate_only is True
    assert image_args.accept_images is True
    assert image_args.models_ci_run_id == 23578993514


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


def test_main_runs_steps_5_through_8_and_creates_release_branch(tmp_path):
    args = make_args(ci_artifacts_path="release_logs/v0.10.0", models_ci_run_id=None)
    output_dir = tmp_path / "release_logs" / "v0.10.0"

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="stable"), patch.object(
        rel, "get_version", return_value="0.10.0"
    ), patch.object(
        rel, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(rel, "require_only_allowed_changes") as clean_mock, patch.object(
        rel, "run_release_artifacts", return_value=0
    ) as artifacts_mock, patch.object(
        rel, "run_release_images", return_value=0
    ) as images_mock, patch.object(
        rel, "run_release_notes", return_value=0
    ) as notes_mock, patch.object(
        rel, "stage_release_outputs"
    ) as stage_mock, patch.object(
        rel, "commit_release_outputs"
    ) as commit_mock, patch.object(
        rel, "create_and_push_release_branch", return_value="v0.10.0"
    ) as branch_mock:
        assert rel.main() == 0

    assert clean_mock.call_count == 2
    assert artifacts_mock.call_count == 1
    assert images_mock.call_count == 1
    assert notes_mock.call_count == 1
    stage_mock.assert_called_once()
    commit_mock.assert_called_once_with(rel.REPO_ROOT, version="0.10.0", dry_run=False)
    branch_mock.assert_called_once_with(rel.REPO_ROOT, version="0.10.0", dry_run=False)


def test_main_validate_only_stops_after_step6(tmp_path):
    args = make_args(validate_only=True)
    output_dir = tmp_path / "release_logs" / "v0.10.0"

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="stable"), patch.object(
        rel, "get_version", return_value="0.10.0"
    ), patch.object(
        rel, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(rel, "require_only_allowed_changes"), patch.object(
        rel, "run_release_artifacts", return_value=0
    ), patch.object(rel, "run_release_images", return_value=0), patch.object(
        rel, "run_release_notes"
    ) as notes_mock, patch.object(
        rel, "stage_release_outputs"
    ) as stage_mock, patch.object(
        rel, "commit_release_outputs"
    ) as commit_mock, patch.object(
        rel, "create_and_push_release_branch"
    ) as branch_mock:
        assert rel.main() == 0

    notes_mock.assert_not_called()
    stage_mock.assert_not_called()
    commit_mock.assert_not_called()
    branch_mock.assert_not_called()


def test_main_stops_when_step5_fails(tmp_path):
    args = make_args()
    output_dir = tmp_path / "release_logs" / "v0.10.0"

    with patch.object(rel, "configure_logging"), patch.object(
        rel, "parse_args", return_value=args
    ), patch.object(rel, "get_current_branch", return_value="stable"), patch.object(
        rel, "get_version", return_value="0.10.0"
    ), patch.object(
        rel, "resolve_release_output_dir", return_value=output_dir
    ), patch.object(rel, "require_only_allowed_changes"), patch.object(
        rel, "run_release_artifacts", return_value=1
    ), patch.object(rel, "run_release_images") as images_mock:
        assert rel.main() == 1

    images_mock.assert_not_called()

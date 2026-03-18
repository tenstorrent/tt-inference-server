from pathlib import Path
from unittest.mock import Mock, patch
import json

import pytest

from scripts.release.release_diff import build_template_key
from scripts.release.update_model_spec import (
    apply_release_version_to_manual_updates_from_git,
    build_release_diff_records_from_git,
    generate_release_diff_outputs_from_git,
    main,
    resolve_latest_release_branch_ref,
    update_template_fields,
)


def make_snapshot(
    impl_id,
    occurrence_index,
    template_text,
    *,
    impl="tt",
    model_arch="DemoModel",
    inference_engine="vllm",
    weights=None,
    devices=None,
    status=None,
    tt_metal_commit=None,
    vllm_commit=None,
):
    weights = weights or ["demo/model"]
    devices = devices or ["N150"]
    return {
        "impl": impl,
        "impl_id": impl_id,
        "model_arch": model_arch,
        "inference_engine": inference_engine,
        "weights": weights,
        "devices": devices,
        "status": status,
        "tt_metal_commit": tt_metal_commit,
        "vllm_commit": vllm_commit,
        "template_text": template_text,
        "template_key": build_template_key(impl_id, weights, devices, inference_engine),
        "occurrence_key": (impl_id, occurrence_index),
    }


def test_resolve_latest_release_branch_ref_selects_highest_semver_branch():
    repo_root = Path("/tmp/repo")
    git_output = "\n".join(
        [
            "refs/heads/main main",
            "refs/heads/stable stable",
            "refs/heads/v0.9.0 v0.9.0",
            "refs/remotes/origin/HEAD origin/HEAD",
            "refs/remotes/origin/v0.10.0 origin/v0.10.0",
            "refs/remotes/origin/patch-v0.10.0 origin/patch-v0.10.0",
            "refs/heads/v0.11.0 v0.11.0",
        ]
    )

    with patch(
        "scripts.release.update_model_spec.subprocess.run",
        return_value=Mock(returncode=0, stdout=git_output, stderr=""),
    ):
        assert resolve_latest_release_branch_ref(repo_root) == "v0.11.0"


def test_resolve_latest_release_branch_ref_errors_without_release_branches():
    repo_root = Path("/tmp/repo")
    git_output = "\n".join(
        [
            "refs/heads/main main",
            "refs/heads/stable stable",
            "refs/remotes/origin/HEAD origin/HEAD",
            "refs/remotes/origin/patch-v0.10.0 origin/patch-v0.10.0",
        ]
    )

    with patch(
        "scripts.release.update_model_spec.subprocess.run",
        return_value=Mock(returncode=0, stdout=git_output, stderr=""),
    ):
        with pytest.raises(
            RuntimeError,
            match="Could not find any release branches matching vMAJOR.MINOR.PATCH",
        ):
            resolve_latest_release_branch_ref(repo_root)


def test_build_release_diff_records_from_git_includes_ci_and_manual_changes():
    model_spec_path = Path("/tmp/workflows/model_spec.py")
    base_snapshots = [
        make_snapshot(
            "ci_impl",
            0,
            "before-ci",
            status="FUNCTIONAL",
            tt_metal_commit="aaaaaaa",
            vllm_commit="1111111",
        )
    ]
    after_snapshots = [
        make_snapshot(
            "ci_impl",
            0,
            "after-ci",
            status="COMPLETE",
            tt_metal_commit="bbbbbbb",
            vllm_commit="2222222",
        ),
        make_snapshot(
            "manual_impl",
            0,
            "after-manual",
            model_arch="ManualModel",
            weights=["manual/model"],
            devices=["T3K"],
            status="EXPERIMENTAL",
            tt_metal_commit="ccccccc",
            vllm_commit="3333333",
        ),
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[base_snapshots, after_snapshots],
    ):
        records = build_release_diff_records_from_git(
            model_spec_path,
            current_content="current-content",
            ci_metadata_by_occurrence={
                ("ci_impl", 0): {
                    "ci_job_url": "https://example.com/run/123",
                    "ci_run_number": 123,
                }
            },
        )

    assert len(records) == 2

    ci_record = records[0]
    assert ci_record["template_key"] == build_template_key(
        "ci_impl", ["demo/model"], ["N150"], "vllm"
    )
    assert ci_record["impl_id"] == "ci_impl"
    assert ci_record["inference_engine"] == "vllm"
    assert ci_record["status_before"] == "FUNCTIONAL"
    assert ci_record["status_after"] == "COMPLETE"
    assert ci_record["tt_metal_commit_before"] == "aaaaaaa"
    assert ci_record["tt_metal_commit_after"] == "bbbbbbb"
    assert ci_record["ci_job_url"] == "https://example.com/run/123"
    assert ci_record["ci_run_number"] == 123

    manual_record = records[1]
    assert manual_record["template_key"] == build_template_key(
        "manual_impl", ["manual/model"], ["T3K"], "vllm"
    )
    assert manual_record["impl_id"] == "manual_impl"
    assert manual_record["inference_engine"] == "vllm"
    assert manual_record["status_before"] is None
    assert manual_record["status_after"] == "EXPERIMENTAL"
    assert manual_record["tt_metal_commit_before"] is None
    assert manual_record["tt_metal_commit_after"] == "ccccccc"
    assert manual_record["ci_job_url"] is None
    assert manual_record["ci_run_number"] is None


def test_generate_release_diff_outputs_from_git_writes_markdown_and_json(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("# dummy model spec\n")
    output_dir = tmp_path / "release_logs" / "v0.10.0"

    base_snapshots = [
        make_snapshot(
            "ci_impl",
            0,
            "before-ci",
            status="FUNCTIONAL",
            tt_metal_commit="aaaaaaa",
        )
    ]
    after_snapshots = [
        make_snapshot(
            "ci_impl",
            0,
            "after-ci",
            status="COMPLETE",
            tt_metal_commit="bbbbbbb",
        ),
        make_snapshot(
            "manual_impl",
            0,
            "after-manual",
            model_arch="ManualModel",
            weights=["manual/model"],
            devices=["T3K"],
            status="EXPERIMENTAL",
            tt_metal_commit="ccccccc",
        ),
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[base_snapshots, after_snapshots],
    ):
        markdown_path, json_path = generate_release_diff_outputs_from_git(
            model_spec_path,
            output_dir,
            current_content="current-content",
            ci_metadata_by_occurrence={
                ("ci_impl", 0): {
                    "ci_job_url": "https://example.com/run/123",
                    "ci_run_number": 123,
                }
            },
        )

    markdown_content = markdown_path.read_text()
    assert markdown_path == output_dir / "pre_release_models_diff.md"
    assert "# Model Spec Release Updates" in markdown_content
    assert (
        "| Impl | Model Arch | Weights | Devices | TT-Metal Commit Change | Status Change | CI Job Link |"
        in markdown_content
    )
    assert "[Run 123](https://example.com/run/123)" in markdown_content
    assert (
        "| `manual_impl` | `ManualModel` | `manual/model` | T3K | New: `ccccccc` | New: EXPERIMENTAL | N/A |"
        in markdown_content
    )

    output_records = json.loads(json_path.read_text())
    assert json_path == output_dir / "pre_release_models_diff.json"
    assert [record["impl_id"] for record in output_records] == [
        "ci_impl",
        "manual_impl",
    ]
    assert output_records[0]["template_key"] == build_template_key(
        "ci_impl", ["demo/model"], ["N150"], "vllm"
    )
    assert output_records[0]["inference_engine"] == "vllm"
    assert output_records[0]["ci_run_number"] == 123
    assert output_records[1]["ci_job_url"] is None


def test_generate_release_diff_outputs_from_git_handles_no_changes(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("# dummy model spec\n")
    output_dir = tmp_path / "release_logs" / "v0.10.0"

    unchanged_snapshot = make_snapshot(
        "same_impl",
        0,
        "same-text",
        status="FUNCTIONAL",
        tt_metal_commit="aaaaaaa",
        vllm_commit="1111111",
    )

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[[unchanged_snapshot], [unchanged_snapshot]],
    ):
        markdown_path, json_path = generate_release_diff_outputs_from_git(
            model_spec_path,
            output_dir,
            current_content="current-content",
        )

    assert "No updates were made." in markdown_path.read_text()
    assert json.loads(json_path.read_text()) == []


def test_build_release_diff_records_from_git_uses_provided_base_ref():
    model_spec_path = Path("/tmp/workflows/model_spec.py")
    changed_snapshot = make_snapshot(
        "ci_impl",
        0,
        "changed-text",
        status="COMPLETE",
        tt_metal_commit="bbbbbbb",
        vllm_commit="2222222",
    )

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ) as read_base_mock, patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[[changed_snapshot], [changed_snapshot]],
    ):
        build_release_diff_records_from_git(
            model_spec_path,
            current_content="current-content",
            base_ref="origin/v0.10.0",
        )

    read_base_mock.assert_called_once_with(model_spec_path, ref="origin/v0.10.0")


def test_release_diff_uses_latest_release_branch_not_head():
    model_spec_path = Path("/tmp/workflows/model_spec.py")
    current_snapshot = make_snapshot(
        "release_impl",
        0,
        "release-template",
        tt_metal_commit="bbbbbbb",
        vllm_commit="2222222",
        status="COMPLETE",
    )
    head_snapshot = make_snapshot(
        "release_impl",
        0,
        "head-template",
        tt_metal_commit="aaaaaaa",
        vllm_commit="1111111",
        status="FUNCTIONAL",
    )

    def read_base_content(_, ref=None):
        if ref == "HEAD":
            return "head-base"
        return "release-base"

    def build_snapshots(_, content, __):
        if content == "head-base":
            return [head_snapshot]
        if content in {"release-base", "current-content"}:
            return [current_snapshot]
        raise AssertionError(f"Unexpected content: {content}")

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        side_effect=read_base_content,
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=build_snapshots,
    ):
        head_records = build_release_diff_records_from_git(
            model_spec_path,
            current_content="current-content",
            base_ref="HEAD",
        )
        release_records = build_release_diff_records_from_git(
            model_spec_path,
            current_content="current-content",
            base_ref="v0.10.0",
        )

    assert len(head_records) == 1
    assert release_records == []


def test_build_release_diff_records_from_git_rejects_duplicate_template_keys():
    model_spec_path = Path("/tmp/workflows/model_spec.py")
    duplicate_before_snapshots = [
        make_snapshot(
            "dup_impl",
            0,
            "before-one",
            weights=["dup/model"],
            devices=["N150"],
        ),
        make_snapshot(
            "dup_impl",
            1,
            "before-two",
            weights=["dup/model"],
            devices=["N150"],
        ),
    ]
    after_snapshots = [
        make_snapshot(
            "dup_impl",
            0,
            "after-one",
            weights=["dup/model"],
            devices=["N150"],
        )
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[duplicate_before_snapshots, after_snapshots],
    ):
        with pytest.raises(ValueError, match="Duplicate template identities detected"):
            build_release_diff_records_from_git(
                model_spec_path,
                current_content="current-content",
            )


def test_update_template_fields_inserts_release_version_when_absent():
    template_text = (
        "ModelSpecTemplate(\n"
        '        weights=["org/model"],\n'
        "        impl=demo_impl,\n"
        '        tt_metal_commit="aaaaaaa",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    )"
    )
    result = update_template_fields(
        template_text, "bbbbbbb", "2222222", None, release_version="0.10.0"
    )
    assert 'tt_metal_commit="bbbbbbb"' in result
    assert 'release_version="0.10.0"' in result
    assert 'vllm_commit="2222222"' in result
    assert result.index('tt_metal_commit="bbbbbbb"') < result.index(
        'release_version="0.10.0"'
    )


def test_update_template_fields_replaces_existing_release_version():
    template_text = (
        "ModelSpecTemplate(\n"
        '        weights=["org/model"],\n'
        "        impl=demo_impl,\n"
        '        tt_metal_commit="aaaaaaa",\n'
        '        release_version="0.9.0",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    )"
    )
    result = update_template_fields(
        template_text, "bbbbbbb", None, None, release_version="0.10.0"
    )
    assert 'release_version="0.10.0"' in result
    assert 'release_version="0.9.0"' not in result


def test_update_template_fields_skips_release_version_when_none():
    template_text = 'ModelSpecTemplate(\n        tt_metal_commit="aaaaaaa",\n    )'
    result = update_template_fields(
        template_text, None, None, None, release_version=None
    )
    assert "release_version" not in result


def test_apply_release_version_to_manual_updates_from_git_updates_tt_metal_changes_only(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)

    changed_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/changed"],\n'
        "        impl=changed_impl,\n"
        '        tt_metal_commit="bbbbbbb",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    unchanged_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/unchanged"],\n'
        "        impl=unchanged_impl,\n"
        '        tt_metal_commit="ddddddd",\n'
        '        vllm_commit="2222222",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    current_content = (
        f"spec_templates = [\n{changed_template}\n{unchanged_template}\n]\n"
    )
    model_spec_path.write_text(current_content)

    before_snapshots = [
        make_snapshot(
            "changed_impl",
            0,
            changed_template.replace("bbbbbbb", "aaaaaaa"),
            weights=["org/changed"],
            tt_metal_commit="aaaaaaa",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "unchanged_impl",
            0,
            unchanged_template,
            weights=["org/unchanged"],
            tt_metal_commit="ddddddd",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        ),
    ]
    after_snapshots = [
        make_snapshot(
            "changed_impl",
            0,
            changed_template,
            weights=["org/changed"],
            tt_metal_commit="bbbbbbb",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "unchanged_impl",
            0,
            unchanged_template,
            weights=["org/unchanged"],
            tt_metal_commit="ddddddd",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        ),
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[before_snapshots, after_snapshots],
    ):
        updated_content, updates_made = (
            apply_release_version_to_manual_updates_from_git(
                model_spec_path, current_content, "0.10.0"
            )
        )

    assert updates_made == 1
    assert updated_content.count('release_version="0.10.0"') == 1
    assert (
        updated_content.index('weights=["org/changed"]')
        < updated_content.index('release_version="0.10.0"')
        < updated_content.index('weights=["org/unchanged"]')
    )


def test_apply_release_version_to_manual_updates_from_git_uses_provided_base_ref(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    current_content = "spec_templates = []\n"
    model_spec_path.write_text(current_content)

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ) as read_base_mock, patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[[], []],
    ):
        apply_release_version_to_manual_updates_from_git(
            model_spec_path,
            current_content,
            "0.10.0",
            base_ref="origin/v0.10.0",
        )

    read_base_mock.assert_called_once_with(model_spec_path, ref="origin/v0.10.0")


def test_main_output_only_updates_release_version_before_generating_outputs(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    current_content = "spec_templates = []\n"
    updated_content = (
        "spec_templates = [\n"
        "    ModelSpecTemplate(\n"
        '        tt_metal_commit="bbbbbbb",\n'
        '        release_version="0.10.0",\n'
        "    ),\n"
        "]\n"
    )
    model_spec_path.write_text(current_content)

    readme_path = tmp_path / "README.md"
    readme_path.write_text("# README\n")
    output_json_path = tmp_path / "default_model_spec.json"
    release_output_dir = tmp_path / "release_logs" / "v0.10.0"

    args = type(
        "Args",
        (),
        {
            "last_good_json": None,
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_json": str(output_json_path),
            "output_only": True,
            "readme_path": str(readme_path),
            "ignore_perf_status": False,
            "models_ci_run_id": None,
            "out_root": None,
        },
    )()

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.update_model_spec.resolve_latest_release_branch_ref",
        return_value="origin/v0.10.0",
    ), patch(
        "scripts.release.update_model_spec.apply_release_version_to_manual_updates_from_git",
        return_value=(updated_content, 1),
    ) as apply_release_mock, patch(
        "scripts.release.update_model_spec.resolve_release_output_dir",
        return_value=release_output_dir,
    ), patch(
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ) as diff_mock, patch(
        "scripts.release.update_model_spec.regenerate_model_support_docs_and_update_readme"
    ) as readme_mock, patch(
        "scripts.release.update_model_spec.reload_and_export_model_specs_json"
    ) as export_mock:
        main()

    assert model_spec_path.read_text() == updated_content
    assert apply_release_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    assert diff_mock.call_args.kwargs["current_content"] == updated_content
    assert diff_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    readme_mock.assert_called_once_with(model_spec_path, str(readme_path))
    export_mock.assert_called_once_with(model_spec_path, output_json_path)


def test_main_uses_resolved_release_output_dir_for_diff_outputs(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")

    input_dir = tmp_path / "external_ci"
    input_dir.mkdir()
    last_good_json_path = input_dir / "models_ci_last_good.json"
    last_good_json_path.write_text("{}\n")

    output_json_path = tmp_path / "default_model_spec.json"
    release_output_dir = tmp_path / "release_logs" / "v0.10.0"

    args = type(
        "Args",
        (),
        {
            "last_good_json": str(last_good_json_path),
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_json": str(output_json_path),
            "output_only": False,
            "readme_path": str(tmp_path / "README.md"),
            "ignore_perf_status": False,
            "models_ci_run_id": None,
            "out_root": None,
        },
    )()

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.update_model_spec.resolve_latest_release_branch_ref",
        return_value="origin/v0.10.0",
    ), patch(
        "scripts.release.update_model_spec.resolve_release_output_dir",
        return_value=release_output_dir,
    ) as resolve_mock, patch(
        "scripts.release.update_model_spec.reload_and_export_model_specs_json"
    ) as export_mock, patch(
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ) as diff_mock:
        main()

    resolve_mock.assert_called_once_with(None)
    export_mock.assert_called_once_with(model_spec_path, output_json_path)
    assert diff_mock.call_args.args[1] == release_output_dir
    assert diff_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    assert diff_mock.call_args.kwargs["current_content"] == "spec_templates = []\n"

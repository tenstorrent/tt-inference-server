from pathlib import Path
from unittest.mock import Mock, patch
import json

import pytest

from scripts.release.release_diff import build_template_key
from scripts.release.update_model_spec import (
    apply_release_version_to_manual_updates_from_git,
    build_template_snapshots,
    build_release_diff_records_from_git,
    format_release_ref_for_display,
    generate_release_diff_outputs_from_git,
    load_model_spec_module_from_content,
    main,
    revert_disallowed_tt_metal_commit_diffs_from_git,
    resolve_latest_release_branch_ref,
    update_template_fields,
)


def test_load_model_spec_module_from_content_supports_legacy_perf_target_imports():
    model_spec_path = Path("workflows/model_spec.py")
    legacy_module_name = "workflows.utils_report"
    legacy_content = f"""
from {legacy_module_name} import BenchmarkTaskParams, PerformanceTarget

spec_templates = []
SAMPLE_TARGET = PerformanceTarget(ttft_ms=1.23, tolerance=0.05)
SAMPLE_PARAMS = BenchmarkTaskParams(targets={{"customer_complete": SAMPLE_TARGET}})
"""

    loaded_module = load_model_spec_module_from_content(
        model_spec_path,
        legacy_content,
        "legacy_model_spec_module",
    )

    assert loaded_module.SAMPLE_TARGET.ttft_ms == 1.23
    assert (
        loaded_module.SAMPLE_PARAMS.targets["customer_complete"]
        == loaded_module.SAMPLE_TARGET
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


def test_build_template_key_returns_readable_identity():
    assert build_template_key(
        "demo_impl",
        ["demo/model", "org/model with spaces"],
        ["N150", "GALAXY_T3K"],
        "vllm",
    ) == (
        "template:v1|impl_id=demo_impl|engine=vllm|"
        "weights=demo/model,org/model%20with%20spaces|devices=N150,GALAXY_T3K"
    )


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


def test_resolve_latest_release_branch_ref_ignores_nested_names_and_uses_tags():
    repo_root = Path("/tmp/repo")
    git_output = "\n".join(
        [
            "refs/heads/main main",
            "refs/heads/post-release/v0.10.0 post-release/v0.10.0",
            "refs/remotes/origin/HEAD origin/HEAD",
            "refs/remotes/origin/post-release/v0.10.0 origin/post-release/v0.10.0",
            "refs/tags/v0.10.0 v0.10.0",
        ]
    )

    with patch(
        "scripts.release.update_model_spec.subprocess.run",
        return_value=Mock(returncode=0, stdout=git_output, stderr=""),
    ):
        assert resolve_latest_release_branch_ref(repo_root) == "v0.10.0"


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
            match="Could not find any release branches or tags matching vMAJOR.MINOR.PATCH",
        ):
            resolve_latest_release_branch_ref(repo_root)


def test_format_release_ref_for_display_normalizes_release_ref_names():
    assert format_release_ref_for_display("origin/v0.10.0") == "v0.10.0"
    assert format_release_ref_for_display("refs/remotes/origin/v0.10.0") == "v0.10.0"
    assert format_release_ref_for_display("refs/tags/v0.10.0") == "v0.10.0"


def test_load_model_spec_module_from_content_supports_dataclasses_with_annotations(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    content = """
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DemoSpec:
    parent: Optional[DemoSpec] = None


spec_templates = []
""".lstrip()

    module = load_model_spec_module_from_content(
        model_spec_path,
        content,
        "test_model_spec_dynamic_module",
    )

    assert module.spec_templates == []
    assert module.DemoSpec(parent=None).parent is None


def test_build_template_snapshots_uses_source_order_when_spec_templates_reordered(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    content = """
from dataclasses import dataclass


class ModelStatusTypes:
    FUNCTIONAL = "FUNCTIONAL"


@dataclass(frozen=True)
class Device:
    name: str


@dataclass(frozen=True)
class DeviceModelSpec:
    device: Device


@dataclass(frozen=True)
class Impl:
    impl_id: str
    impl_name: str


@dataclass(frozen=True)
class ModelSpecTemplate:
    weights: list
    impl: Impl
    device_model_specs: list
    inference_engine: str
    tt_metal_commit: str = ""
    vllm_commit: str = ""
    status: str = ""


device = Device("N150")
first_impl = Impl("first_impl", "first")
second_impl = Impl("second_impl", "second")

first_template = ModelSpecTemplate(
    weights=["org/first"],
    impl=first_impl,
    device_model_specs=[DeviceModelSpec(device=device)],
    inference_engine="vllm",
    tt_metal_commit="aaaaaaa",
    vllm_commit="1111111",
    status=ModelStatusTypes.FUNCTIONAL,
)
second_template = ModelSpecTemplate(
    weights=["org/second"],
    impl=second_impl,
    device_model_specs=[DeviceModelSpec(device=device)],
    inference_engine="vllm",
    tt_metal_commit="bbbbbbb",
    vllm_commit="2222222",
    status=ModelStatusTypes.FUNCTIONAL,
)

spec_templates = [second_template, first_template]
""".lstrip()
    model_spec_path.write_text(content)

    snapshots = build_template_snapshots(
        model_spec_path,
        content,
        "test_model_spec_snapshot_source_order",
    )

    assert [snapshot["impl_id"] for snapshot in snapshots] == [
        "first_impl",
        "second_impl",
    ]
    assert 'weights=["org/first"]' in snapshots[0]["template_text"]
    assert 'weights=["org/second"]' in snapshots[1]["template_text"]


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


def test_build_release_diff_records_from_git_skips_status_only_changes():
    model_spec_path = Path("/tmp/workflows/model_spec.py")
    base_snapshots = [
        make_snapshot(
            "same_impl",
            0,
            "before-template",
            status="FUNCTIONAL",
            tt_metal_commit="aaaaaaa",
            vllm_commit="1111111",
        )
    ]
    after_snapshots = [
        make_snapshot(
            "same_impl",
            0,
            "after-template",
            status="COMPLETE",
            tt_metal_commit="aaaaaaa",
            vllm_commit="1111111",
        )
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
        )

    assert records == []


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


def test_revert_disallowed_tt_metal_commit_diffs_from_git_reverts_only_disallowed_changes(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)

    allowed_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/allowed"],\n'
        "        impl=allowed_impl,\n"
        '        tt_metal_commit="allow123",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    blocked_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/blocked"],\n'
        "        impl=blocked_impl,\n"
        '        tt_metal_commit="block123",\n'
        '        vllm_commit="2222222",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    current_content = f"spec_templates = [\n{allowed_template}\n{blocked_template}\n]\n"

    before_snapshots = [
        make_snapshot(
            "allowed_impl",
            0,
            allowed_template.replace("allow123", "base1234"),
            weights=["org/allowed"],
            tt_metal_commit="base1234",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "blocked_impl",
            0,
            blocked_template.replace("block123", "base9999"),
            weights=["org/blocked"],
            tt_metal_commit="base9999",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        ),
    ]
    after_snapshots = [
        make_snapshot(
            "allowed_impl",
            0,
            allowed_template,
            weights=["org/allowed"],
            tt_metal_commit="allow123",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "blocked_impl",
            0,
            blocked_template,
            weights=["org/blocked"],
            tt_metal_commit="block123",
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
        filtered_content, reverted_count, deleted_count = (
            revert_disallowed_tt_metal_commit_diffs_from_git(
                model_spec_path,
                current_content,
                tt_metal_commits=["allow123"],
            )
        )

    assert reverted_count == 1
    assert deleted_count == 0
    assert 'tt_metal_commit="allow123"' in filtered_content
    assert 'tt_metal_commit="block123"' not in filtered_content
    assert 'tt_metal_commit="base9999"' in filtered_content


def test_revert_disallowed_tt_metal_commit_diffs_from_git_reverts_release_version_only_diff(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)

    before_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/blocked"],\n'
        "        impl=blocked_impl,\n"
        '        tt_metal_commit="block123",\n'
        '        vllm_commit="2222222",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    after_template = (
        "ModelSpecTemplate(\n"
        '        weights=["org/blocked"],\n'
        "        impl=blocked_impl,\n"
        '        release_version="0.12.0",\n'
        '        tt_metal_commit="block123",\n'
        '        vllm_commit="2222222",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    current_content = f"spec_templates = [\n{after_template}\n]\n"
    before_snapshots = [
        make_snapshot(
            "blocked_impl",
            0,
            before_template,
            weights=["org/blocked"],
            tt_metal_commit="block123",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        )
    ]
    after_snapshots = [
        make_snapshot(
            "blocked_impl",
            0,
            after_template,
            weights=["org/blocked"],
            tt_metal_commit="block123",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        )
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[before_snapshots, after_snapshots],
    ):
        filtered_content, reverted_count, deleted_count = (
            revert_disallowed_tt_metal_commit_diffs_from_git(
                model_spec_path,
                current_content,
                tt_metal_commits=["allow123"],
            )
        )

    assert reverted_count == 1
    assert deleted_count == 0
    assert 'release_version="0.12.0"' not in filtered_content
    assert filtered_content == f"spec_templates = [\n{before_template}\n]\n"


def test_revert_disallowed_tt_metal_commit_diffs_from_git_deletes_unmatched_templates(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    current_content = (
        "spec_templates = [\n"
        "    ModelSpecTemplate(\n"
        '        weights=["org/new"],\n'
        "        impl=new_impl,\n"
        '        tt_metal_commit="new1234",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),\n"
        "]\n"
    )
    after_snapshots = [
        make_snapshot(
            "new_impl",
            0,
            current_content.split("[\n", 1)[1].rsplit("\n]\n", 1)[0],
            weights=["org/new"],
            tt_metal_commit="new1234",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        )
    ]

    with patch(
        "scripts.release.update_model_spec.read_git_base_model_spec_content",
        return_value="base-content",
    ), patch(
        "scripts.release.update_model_spec.build_template_snapshots",
        side_effect=[[], after_snapshots],
    ):
        filtered_content, reverted_count, deleted_count = (
            revert_disallowed_tt_metal_commit_diffs_from_git(
                model_spec_path,
                current_content,
                tt_metal_commits=["allow123"],
            )
        )

    assert filtered_content == "spec_templates = [\n]\n"
    assert reverted_count == 0
    assert deleted_count == 1


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
    assert result.index('release_version="0.10.0"') < result.index(
        'tt_metal_commit="bbbbbbb"'
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
    assert result.index('release_version="0.10.0"') < result.index(
        'tt_metal_commit="bbbbbbb"'
    )


def test_update_template_fields_moves_existing_release_version_above_tt_metal_commit():
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
        template_text, "bbbbbbb", "2222222", None, release_version="0.10.0"
    )
    assert 'release_version="0.10.0"' in result
    assert result.index('release_version="0.10.0"') < result.index(
        'tt_metal_commit="bbbbbbb"'
    )


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


def test_apply_release_version_to_manual_updates_from_git_sorts_mixed_match_indices(
    tmp_path,
):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    template_a = (
        "ModelSpecTemplate(\n"
        '        weights=["org/a"],\n'
        "        impl=a_impl,\n"
        '        tt_metal_commit="aaaaaaa",\n'
        '        vllm_commit="1111111",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    template_b = (
        "ModelSpecTemplate(\n"
        '        weights=["org/b"],\n'
        "        impl=b_impl,\n"
        '        tt_metal_commit="bbbbbbb",\n'
        '        vllm_commit="2222222",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    template_c = (
        "ModelSpecTemplate(\n"
        '        weights=["org/c"],\n'
        "        impl=c_impl,\n"
        '        tt_metal_commit="ccccccc",\n'
        '        vllm_commit="3333333",\n'
        "        status=ModelStatusTypes.FUNCTIONAL,\n"
        "    ),"
    )
    current_content = (
        f"spec_templates = [\n{template_a}\n{template_b}\n{template_c}\n]\n"
    )
    model_spec_path.write_text(current_content)

    before_snapshots = [
        make_snapshot(
            "a_impl",
            0,
            template_a.replace("aaaaaaa", "oldaaaa"),
            weights=["org/a"],
            tt_metal_commit="oldaaaa",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "b_impl",
            0,
            template_b.replace("bbbbbbb", "oldbbbb"),
            weights=["org/b-old"],
            tt_metal_commit="oldbbbb",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "c_impl",
            0,
            template_c.replace("ccccccc", "oldcccc"),
            weights=["org/c"],
            tt_metal_commit="oldcccc",
            vllm_commit="3333333",
            status="FUNCTIONAL",
        ),
    ]
    after_snapshots = [
        make_snapshot(
            "a_impl",
            0,
            template_a,
            weights=["org/a"],
            tt_metal_commit="aaaaaaa",
            vllm_commit="1111111",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "b_impl",
            0,
            template_b,
            weights=["org/b"],
            tt_metal_commit="bbbbbbb",
            vllm_commit="2222222",
            status="FUNCTIONAL",
        ),
        make_snapshot(
            "c_impl",
            0,
            template_c,
            weights=["org/c"],
            tt_metal_commit="ccccccc",
            vllm_commit="3333333",
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
                model_spec_path,
                current_content,
                "0.10.0",
            )
        )

    assert updates_made == 3
    assert updated_content.count('release_version="0.10.0"') == 3
    assert "ModelSModelSpecTemplate" not in updated_content
    assert "),odelStatusTypes" not in updated_content


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

    release_output_dir = tmp_path / "release_logs" / "v0.10.0"

    args = type(
        "Args",
        (),
        {
            "last_good_json": None,
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_only": True,
            "ignore_perf_status": False,
            "models_ci_run_id": None,
            "out_root": None,
            "tt_metal_commits": ["bbbbbbb"],
        },
    )()

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.update_model_spec.resolve_latest_release_branch_ref",
        return_value="origin/v0.10.0",
    ), patch(
        "scripts.release.update_model_spec.revert_disallowed_tt_metal_commit_diffs_from_git",
        return_value=("filtered-content\n", 0, 0),
    ), patch(
        "scripts.release.update_model_spec.apply_release_version_to_manual_updates_from_git",
        return_value=(updated_content, 1),
    ) as apply_release_mock, patch(
        "scripts.release.update_model_spec.resolve_release_output_dir",
        return_value=release_output_dir,
    ), patch(
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ) as diff_mock:
        main()

    assert model_spec_path.read_text() == updated_content
    assert apply_release_mock.call_args.args[1] == "filtered-content\n"
    assert apply_release_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    assert diff_mock.call_args.kwargs["current_content"] == updated_content
    assert diff_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"


def test_main_uses_resolved_release_output_dir_for_diff_outputs(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")

    input_dir = tmp_path / "external_ci"
    input_dir.mkdir()
    last_good_json_path = input_dir / "models_ci_last_good.json"
    last_good_json_path.write_text("{}\n")

    release_output_dir = tmp_path / "release_logs" / "v0.10.0"

    args = type(
        "Args",
        (),
        {
            "last_good_json": str(last_good_json_path),
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_only": False,
            "ignore_perf_status": False,
            "models_ci_run_id": None,
            "out_root": None,
            "tt_metal_commits": None,
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
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ) as diff_mock:
        main()

    resolve_mock.assert_called_once_with(None)
    assert diff_mock.call_args.args[1] == release_output_dir
    assert diff_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    assert diff_mock.call_args.kwargs["current_content"] == "spec_templates = []\n"


def test_main_ci_uses_filtered_content_for_outputs(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")

    last_good_json_path = tmp_path / "models_ci_last_good.json"
    last_good_json_path.write_text("{}\n")
    release_output_dir = tmp_path / "release_logs" / "v0.10.0"
    filtered_content = "spec_templates = []\n# filtered\n"

    args = type(
        "Args",
        (),
        {
            "last_good_json": str(last_good_json_path),
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_only": False,
            "ignore_perf_status": False,
            "models_ci_run_id": None,
            "out_root": None,
            "tt_metal_commits": ["allow123"],
        },
    )()

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.update_model_spec.resolve_latest_release_branch_ref",
        return_value="origin/v0.10.0",
    ), patch(
        "scripts.release.update_model_spec.spec_templates",
        [],
    ), patch(
        "scripts.release.update_model_spec.resolve_release_output_dir",
        return_value=release_output_dir,
    ), patch(
        "scripts.release.update_model_spec.revert_disallowed_tt_metal_commit_diffs_from_git",
        return_value=(filtered_content, 1, 0),
    ) as filter_mock, patch(
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ) as diff_mock:
        main()

    assert model_spec_path.read_text() == filtered_content
    assert filter_mock.call_args.kwargs["tt_metal_commits"] == ["allow123"]
    assert filter_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"
    assert diff_mock.call_args.kwargs["current_content"] == filtered_content
    assert diff_mock.call_args.kwargs["base_ref"] == "origin/v0.10.0"


def test_main_models_ci_run_id_keeps_nightly_reader_defaults(tmp_path):
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")
    release_output_dir = tmp_path / "release_logs" / "v0.10.0"
    last_good_json_path = tmp_path / "models_ci_last_good.json"
    last_good_json_path.write_text("{}\n")

    args = type(
        "Args",
        (),
        {
            "last_good_json": None,
            "model_spec_path": str(model_spec_path),
            "dry_run": False,
            "output_only": False,
            "ignore_perf_status": False,
            "models_ci_run_id": 19339722549,
            "out_root": None,
            "tt_metal_commits": None,
        },
    )()

    fake_models_ci_reader = Mock()
    fake_models_ci_reader.run_ci_pipeline = Mock(return_value=last_good_json_path)

    with patch.dict(
        "sys.modules", {"scripts.release.models_ci_reader": fake_models_ci_reader}
    ), patch(
        "argparse.ArgumentParser.parse_args",
        return_value=args,
    ), patch(
        "scripts.release.update_model_spec.resolve_latest_release_branch_ref",
        return_value="origin/v0.10.0",
    ), patch(
        "scripts.release.update_model_spec.resolve_release_output_dir",
        return_value=release_output_dir,
    ), patch(
        "scripts.release.update_model_spec.spec_templates",
        [],
    ), patch(
        "scripts.release.update_model_spec.generate_release_diff_outputs_from_git"
    ):
        main()

    assert fake_models_ci_reader.run_ci_pipeline.call_args.args == (
        19339722549,
        release_output_dir,
    )
    assert fake_models_ci_reader.run_ci_pipeline.call_args.kwargs == {}

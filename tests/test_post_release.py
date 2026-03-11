import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.release.post_release import (
    build_post_release_pr_markdown,
    build_updated_model_spec_content,
    increment_version,
    main,
)
from scripts.release.release_diff import build_template_key


def make_template_text(
    *,
    weight="org/model",
    impl="demo_impl",
    tt_metal_commit="aaaaaaa",
    vllm_commit="1111111",
    status="FUNCTIONAL",
    release_version=None,
):
    release_version_line = ""
    if release_version is not None:
        release_version_line = f'        release_version="{release_version}",\n'

    if vllm_commit is None:
        vllm_commit_line = "        vllm_commit=None,\n"
    else:
        vllm_commit_line = f'        vllm_commit="{vllm_commit}",\n'

    return (
        "ModelSpecTemplate(\n"
        f'        weights=["{weight}"],\n'
        f"        impl={impl},\n"
        f'        tt_metal_commit="{tt_metal_commit}",\n'
        f"{release_version_line}"
        f"{vllm_commit_line}"
        f"        status=ModelStatusTypes.{status},\n"
        "    ),"
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
    weights = weights or ["org/model"]
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


def make_diff_record(
    *,
    impl_id="demo_impl",
    inference_engine="vllm",
    weights=None,
    devices=None,
    tt_before="aaaaaaa",
    tt_after="bbbbbbb",
    vllm_before="1111111",
    vllm_after="2222222",
    status_before="FUNCTIONAL",
    status_after="COMPLETE",
    template_key=None,
):
    weights = weights or ["org/model"]
    devices = devices or ["N150"]
    return {
        "template_key": template_key
        or build_template_key(impl_id, weights, devices, inference_engine),
        "impl": "tt",
        "impl_id": impl_id,
        "model_arch": "DemoModel",
        "inference_engine": inference_engine,
        "weights": weights,
        "devices": devices,
        "tt_metal_commit_before": tt_before,
        "tt_metal_commit_after": tt_after,
        "vllm_commit_before": vllm_before,
        "vllm_commit_after": vllm_after,
        "status_before": status_before,
        "status_after": status_after,
        "ci_job_url": None,
        "ci_run_number": None,
    }


def test_increment_version_supports_major_minor_and_patch():
    assert increment_version("0.9.0", "major") == "1.0.0"
    assert increment_version("0.9.0", "minor") == "0.10.0"
    assert increment_version("0.9.0", "patch") == "0.9.1"


def test_build_updated_model_spec_content_applies_matching_release_updates():
    template_text = make_template_text()
    current_content = f"spec_templates = [\n{template_text}\n]\n"
    snapshot = make_snapshot(
        "demo_impl",
        0,
        template_text,
        status="FUNCTIONAL",
        tt_metal_commit="aaaaaaa",
        vllm_commit="1111111",
    )
    diff_records = [make_diff_record()]

    with patch(
        "scripts.release.post_release.build_template_snapshots",
        return_value=[snapshot],
    ):
        updated_content, summary = build_updated_model_spec_content(
            Path("/tmp/workflows/model_spec.py"),
            current_content,
            diff_records,
            "0.9.0",
        )

    assert 'tt_metal_commit="bbbbbbb"' in updated_content
    assert 'vllm_commit="2222222"' in updated_content
    assert "status=ModelStatusTypes.COMPLETE" in updated_content
    assert 'release_version="0.9.0"' in updated_content
    assert summary["updated_templates"] == 1
    assert summary["applied_records"][0]["applied_fields"] == [
        "tt_metal_commit",
        "vllm_commit",
        "status",
    ]
    assert summary["applied_records"][0]["discarded_fields"] == []


def test_build_updated_model_spec_content_discards_diverged_fields_without_updating_release_version():
    template_text = make_template_text(
        tt_metal_commit="ccccccc",
        vllm_commit="3333333",
        status="TOP_PERF",
    )
    current_content = f"spec_templates = [\n{template_text}\n]\n"
    snapshot = make_snapshot(
        "demo_impl",
        0,
        template_text,
        status="TOP_PERF",
        tt_metal_commit="ccccccc",
        vllm_commit="3333333",
    )
    diff_records = [make_diff_record()]

    with patch(
        "scripts.release.post_release.build_template_snapshots",
        return_value=[snapshot],
    ):
        updated_content, summary = build_updated_model_spec_content(
            Path("/tmp/workflows/model_spec.py"),
            current_content,
            diff_records,
            "0.9.0",
        )

    assert 'tt_metal_commit="ccccccc"' in updated_content
    assert 'vllm_commit="3333333"' in updated_content
    assert "status=ModelStatusTypes.TOP_PERF" in updated_content
    assert 'release_version="0.9.0"' not in updated_content
    assert summary["updated_templates"] == 0
    assert summary["applied_records"][0]["applied_fields"] == []
    assert len(summary["applied_records"][0]["discarded_fields"]) == 3
    assert summary["applied_records"][0]["release_version_updated"] is False


def test_build_updated_model_spec_content_raises_on_duplicate_template_keys():
    template_text_one = make_template_text(weight="org/model-a")
    template_text_two = make_template_text(weight="org/model-a")
    current_content = (
        f"spec_templates = [\n{template_text_one}\n{template_text_two}\n]\n"
    )
    snapshots = [
        make_snapshot(
            "demo_impl",
            0,
            template_text_one,
            weights=["org/model-a"],
        ),
        make_snapshot(
            "demo_impl",
            1,
            template_text_two,
            weights=["org/model-a"],
        ),
    ]
    diff_records = [make_diff_record(weights=["org/model-a"])]

    with patch(
        "scripts.release.post_release.build_template_snapshots",
        return_value=snapshots,
    ):
        with pytest.raises(ValueError, match="Duplicate template identities detected"):
            build_updated_model_spec_content(
                Path("/tmp/workflows/model_spec.py"),
                current_content,
                diff_records,
                "0.9.0",
            )


def test_build_updated_model_spec_content_skips_duplicate_diff_records():
    template_text = make_template_text()
    current_content = f"spec_templates = [\n{template_text}\n]\n"
    snapshot = make_snapshot(
        "demo_impl",
        0,
        template_text,
        status="FUNCTIONAL",
        tt_metal_commit="aaaaaaa",
        vllm_commit="1111111",
    )
    diff_records = [make_diff_record(), make_diff_record()]

    with patch(
        "scripts.release.post_release.build_template_snapshots",
        return_value=[snapshot],
    ):
        updated_content, summary = build_updated_model_spec_content(
            Path("/tmp/workflows/model_spec.py"),
            current_content,
            diff_records,
            "0.9.0",
        )

    assert 'tt_metal_commit="bbbbbbb"' in updated_content
    assert summary["matched_records"] == 1
    assert summary["updated_templates"] == 1
    assert summary["skipped_records"] == [
        {
            "label": "demo_impl [org/model] (N150)",
            "reason": "Duplicate release diff record matched the same template.",
        }
    ]


def test_build_post_release_pr_markdown_reports_discarded_and_skipped_records():
    summary = {
        "matched_records": 1,
        "updated_templates": 1,
        "applied_records": [
            {
                "label": "demo_impl [org/model] (N150)",
                "applied_fields": ["vllm_commit"],
                "discarded_fields": [
                    {
                        "field": "tt_metal_commit",
                        "expected": "aaaaaaa",
                        "current": "ccccccc",
                        "released": "bbbbbbb",
                    }
                ],
                "release_version_updated": True,
                "changed": True,
            }
        ],
        "skipped_records": [
            {
                "label": "other_impl [org/other] (T3K)",
                "reason": "No matching template found on main.",
            }
        ],
    }

    markdown = build_post_release_pr_markdown("0.9.0", "0.10.0", summary)

    assert "- Bumped `VERSION` from `0.9.0` to `0.10.0`." in markdown
    assert "discarded `tt_metal_commit` because `main` has `ccccccc`" in markdown
    assert "`other_impl [org/other] (T3K)`" in markdown
    assert "updated `release_version`" in markdown


def test_main_reads_release_diff_from_pre_bump_version_directory(tmp_path):
    version_file = tmp_path / "VERSION"
    version_file.write_text("0.9.0\n")
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")

    release_dir = tmp_path / "release_logs" / "v0.9.0"
    release_dir.mkdir(parents=True)
    diff_json_path = release_dir / "pre_release_models_diff.json"
    expected_records = [make_diff_record()]
    diff_json_path.write_text(json.dumps(expected_records))

    pr_output_path = tmp_path / "release_logs" / "post_release_pr.md"
    default_model_spec_path = tmp_path / "default_model_spec.json"
    captured = {}

    def fake_build_updated_model_spec_content(
        model_spec_path_arg,
        current_content_arg,
        diff_records_arg,
        released_version_arg,
    ):
        captured["model_spec_path"] = model_spec_path_arg
        captured["current_content"] = current_content_arg
        captured["diff_records"] = diff_records_arg
        captured["released_version"] = released_version_arg
        return current_content_arg, {
            "matched_records": 1,
            "updated_templates": 0,
            "applied_records": [],
            "skipped_records": [],
        }

    args = Namespace(
        increment="minor",
        version_file=str(version_file),
        model_spec_path=str(model_spec_path),
        diff_json=str(diff_json_path),
        default_model_spec_path=str(default_model_spec_path),
        pr_output=str(pr_output_path),
        dry_run=False,
    )

    with patch("scripts.release.post_release.parse_args", return_value=args), patch(
        "scripts.release.post_release.build_updated_model_spec_content",
        side_effect=fake_build_updated_model_spec_content,
    ), patch(
        "scripts.release.post_release.reload_and_export_model_specs_json"
    ) as export_mock, patch(
        "scripts.release.post_release.regenerate_model_support_docs_and_update_readme"
    ) as readme_mock:
        assert main() == 0

    assert version_file.read_text() == "0.10.0\n"
    assert captured["released_version"] == "0.9.0"
    assert captured["diff_records"] == expected_records
    assert captured["model_spec_path"] == model_spec_path
    assert pr_output_path.exists()
    export_mock.assert_called_once_with(model_spec_path, default_model_spec_path)
    readme_mock.assert_called_once_with(model_spec_path)


def test_main_keeps_version_unmodified_until_late_steps_succeed(tmp_path):
    version_file = tmp_path / "VERSION"
    version_file.write_text("0.9.0\n")
    model_spec_path = tmp_path / "workflows" / "model_spec.py"
    model_spec_path.parent.mkdir(parents=True)
    model_spec_path.write_text("spec_templates = []\n")

    updated_content = "spec_templates = [\n    # updated\n]\n"
    release_dir = tmp_path / "release_logs" / "v0.9.0"
    release_dir.mkdir(parents=True)
    (release_dir / "pre_release_models_diff.json").write_text(
        json.dumps([make_diff_record()])
    )

    pr_output_path = tmp_path / "release_logs" / "post_release_pr.md"
    default_model_spec_path = tmp_path / "default_model_spec.json"
    loaded_paths = []

    def fake_release_logs_dir(version):
        return tmp_path / "release_logs" / f"v{version}"

    def fake_load_release_diff_records(path):
        loaded_paths.append(path)
        return [make_diff_record()]

    def fake_build_updated_model_spec_content(
        model_spec_path_arg,
        current_content_arg,
        diff_records_arg,
        released_version_arg,
    ):
        return updated_content, {
            "matched_records": 1,
            "updated_templates": 1,
            "applied_records": [],
            "skipped_records": [],
        }

    args = Namespace(
        increment="minor",
        version_file=str(version_file),
        model_spec_path=str(model_spec_path),
        diff_json=None,
        default_model_spec_path=str(default_model_spec_path),
        pr_output=str(pr_output_path),
        dry_run=False,
    )

    with patch("scripts.release.post_release.parse_args", return_value=args), patch(
        "scripts.release.post_release.get_versioned_release_logs_dir",
        side_effect=fake_release_logs_dir,
    ), patch(
        "scripts.release.post_release.load_release_diff_records",
        side_effect=fake_load_release_diff_records,
    ), patch(
        "scripts.release.post_release.build_updated_model_spec_content",
        side_effect=fake_build_updated_model_spec_content,
    ), patch("scripts.release.post_release.reload_and_export_model_specs_json"), patch(
        "scripts.release.post_release.regenerate_model_support_docs_and_update_readme",
        side_effect=RuntimeError("docs failed"),
    ):
        with pytest.raises(RuntimeError, match="docs failed"):
            main()

    assert version_file.read_text() == "0.9.0\n"
    assert model_spec_path.read_text() == updated_content
    assert loaded_paths == [release_dir / "pre_release_models_diff.json"]

    with patch("scripts.release.post_release.parse_args", return_value=args), patch(
        "scripts.release.post_release.get_versioned_release_logs_dir",
        side_effect=fake_release_logs_dir,
    ), patch(
        "scripts.release.post_release.load_release_diff_records",
        side_effect=fake_load_release_diff_records,
    ), patch(
        "scripts.release.post_release.build_updated_model_spec_content",
        side_effect=fake_build_updated_model_spec_content,
    ), patch("scripts.release.post_release.reload_and_export_model_specs_json"), patch(
        "scripts.release.post_release.regenerate_model_support_docs_and_update_readme"
    ):
        assert main() == 0

    assert version_file.read_text() == "0.10.0\n"
    assert loaded_paths == [
        release_dir / "pre_release_models_diff.json",
        release_dir / "pre_release_models_diff.json",
    ]
    assert pr_output_path.exists()

import io
import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import ANY, patch

import scripts.release.backfill_release_performance as brp


def make_release_model_spec_export(*model_ids):
    model_specs = {}
    for index, model_id in enumerate(model_ids):
        model_specs.setdefault(f"repo-{index}", {}).setdefault("N150", {}).setdefault(
            "vllm", {}
        )[f"impl-{index}"] = {
            "model_id": model_id,
            "model_name": f"Model{index}",
        }
    return {
        "schema_version": "0.1.0",
        "release_version": "0.12.0",
        "model_specs": model_specs,
    }


def build_model_artifact_zip_bytes(model_id, include_report=True, report_payload=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(
            f"workflow_logs_{model_id}/runtime_model_specs/model_spec.json",
            json.dumps(
                {
                    "model_id": model_id,
                    "docker_image": f"ghcr.io/tenstorrent/{model_id}:tag",
                }
            ),
        )
        if include_report:
            if report_payload is None:
                report_payload = {"metadata": {"model_id": model_id}}
            if isinstance(report_payload, str):
                report_content = report_payload
            else:
                report_content = json.dumps(report_payload)
            zf.writestr(
                f"workflow_logs_{model_id}/reports_output/release/data/"
                f"report_data_{model_id}_123.json",
                report_content,
            )
        zf.writestr(
            f"workflow_logs_{model_id}/benchmarks_output/benchmark_{model_id}_123.json",
            json.dumps({"model_id": model_id, "metric": 1.23}),
        )
    return buffer.getvalue()


def build_release_bundle_zip_bytes(model_archives):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for archive_name, archive_bytes in model_archives.items():
            zf.writestr(f"model_archives/{archive_name}.zip", archive_bytes)
    return buffer.getvalue()


def test_list_releases_paginates():
    page_1 = [{"tag_name": "v0.12.0"}]
    page_2 = [{"tag_name": "v0.11.1"}]
    page_3 = []

    with patch.object(
        brp, "http_get", side_effect=[page_1, page_2, page_3]
    ) as get_mock:
        releases = brp.list_releases(
            "tenstorrent", "tt-inference-server", "token", per_page=1
        )

    assert releases == page_1 + page_2
    assert get_mock.call_count == 3


def test_parse_args_defaults_to_tt_inference_server_repo():
    args = brp.parse_args([])

    assert args.owner == "tenstorrent"
    assert args.repo == "tt-inference-server"


def test_select_release_asset_prefers_exact_tag_match():
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {"name": "release_artifacts_v0.11.0.zip"},
            {"name": "v0.11.0-release_artifacts.zip"},
        ],
    }

    asset = brp.select_release_asset(release)

    assert asset["name"] == "v0.11.0-release_artifacts.zip"


def test_select_release_asset_matches_legacy_release_artefacts_name():
    release = {
        "tag_name": "v0.5.0",
        "assets": [
            {"name": "notes.txt"},
            {"name": "12_15_25-v0.5.0-release_artefacts.zip"},
        ],
    }

    asset = brp.select_release_asset(release)

    assert asset["name"] == "12_15_25-v0.5.0-release_artefacts.zip"


def test_download_release_asset_zip_falls_back_to_browser_download_url():
    asset = {
        "name": "v0.11.0-release_artifacts.zip",
        "url": "https://api.github.com/repos/tenstorrent/tt-inference-server/releases/assets/1",
        "browser_download_url": "https://github.com/tenstorrent/tt-inference-server/releases/download/v0.11.0/v0.11.0-release_artifacts.zip",
    }
    browser_bytes = b"zip-bytes"
    api_error = HTTPError(asset["url"], 404, "Not Found", hdrs=None, fp=None)

    with patch.object(brp, "http_get", side_effect=api_error) as get_mock, patch.object(
        brp, "download_release_asset_from_browser_url", return_value=browser_bytes
    ) as browser_download_mock:
        result = brp.download_release_asset_zip(asset, "token")

    assert result == browser_bytes
    get_mock.assert_called_once()
    browser_download_mock.assert_called_once_with(asset["browser_download_url"])


def test_filter_releases_stops_before_v0_5_0():
    releases = [
        {"tag_name": "v0.11.0", "draft": False, "prerelease": False},
        {"tag_name": "v0.5.0", "draft": False, "prerelease": False},
        {"tag_name": "v0.4.0", "draft": False, "prerelease": False},
        {"tag_name": "nightly", "draft": False, "prerelease": False},
    ]

    filtered = brp.filter_releases(releases)

    assert [release["tag_name"] for release in filtered] == ["v0.11.0", "v0.5.0"]


def test_flatten_release_model_specs_indexes_by_model_id(tmp_path):
    spec_path = tmp_path / "release_model_spec.json"
    spec_path.write_text(
        json.dumps(
            make_release_model_spec_export("id_demo_model_n150", "id_other_model_t3k")
        )
    )

    flattened = brp._flatten_release_model_specs(spec_path)

    assert sorted(flattened.keys()) == ["id_demo_model_n150", "id_other_model_t3k"]


def test_collect_reports_for_release_copies_only_current_models(tmp_path):
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {
                "name": "v0.11.0-release_artifacts.zip",
                "url": "https://api.github.com/assets/1",
            }
        ],
    }
    release_bundle = build_release_bundle_zip_bytes(
        {
            "demo_model": build_model_artifact_zip_bytes(
                "demo-model", include_report=True
            ),
            "old_model": build_model_artifact_zip_bytes(
                "old-model", include_report=True
            ),
        }
    )

    with patch.object(
        brp, "download_release_asset_zip", return_value=release_bundle
    ), patch.object(brp, "merge_report_into_release_performance", return_value=True):
        stats = brp.collect_reports_for_release(
            release=release,
            output_root=tmp_path,
            current_model_ids={"demo-model"},
            token="token",
            release_performance_data={"schema_version": "0.1.0", "models": {}},
        )

    copied_report = (
        tmp_path
        / "backfill"
        / "v0.11.0"
        / "reports"
        / "demo-model"
        / "report_data_demo-model_123.json"
    )
    skipped_report = (
        tmp_path
        / "backfill"
        / "v0.11.0"
        / "reports"
        / "old-model"
        / "report_data_old-model_123.json"
    )

    assert stats.assets_downloaded == 1
    assert stats.matching_models_found == 1
    assert stats.reports_copied == 1
    assert stats.reports_patched == 1
    assert stats.baseline_entries_updated == 1
    assert stats.missing_reports == 0
    assert copied_report.exists()
    assert not skipped_report.exists()
    extracted_root = tmp_path / "backfill" / "v0.11.0" / "_extracted"
    extracted_report_paths = sorted(
        path.relative_to(extracted_root)
        for path in extracted_root.rglob("report_data*.json")
    )
    assert extracted_report_paths == [
        Path(
            "v0.11.0-release_artifacts/model_archives/demo_model__unzipped/"
            "workflow_logs_demo-model/reports_output/release/data/"
            "report_data_demo-model_123.json"
        ),
        Path(
            "v0.11.0-release_artifacts/model_archives/old_model__unzipped/"
            "workflow_logs_old-model/reports_output/release/data/"
            "report_data_old-model_123.json"
        ),
    ]
    assert not list(extracted_root.rglob("runtime_model_specs"))
    assert not list(extracted_root.rglob("benchmarks_output"))


def test_collect_reports_for_release_logs_missing_report_for_current_model(
    tmp_path, caplog
):
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {
                "name": "v0.11.0-release_artifacts.zip",
                "url": "https://api.github.com/assets/1",
            }
        ],
    }
    release_bundle = build_release_bundle_zip_bytes(
        {
            "demo_model": build_model_artifact_zip_bytes(
                "demo-model", include_report=False
            ),
        }
    )

    caplog.set_level(logging.ERROR)
    with patch.object(brp, "download_release_asset_zip", return_value=release_bundle):
        stats = brp.collect_reports_for_release(
            release=release,
            output_root=tmp_path,
            current_model_ids={"demo-model"},
            token="token",
            release_performance_data={"schema_version": "0.1.0", "models": {}},
        )

    assert stats.matching_models_found == 1
    assert stats.reports_copied == 0
    assert stats.missing_reports == 1
    assert (
        "Missing report_data JSON for current model demo-model in release v0.11.0"
        in caplog.text
    )


def test_patch_missing_release_version_adds_normalized_version(tmp_path):
    report_path = tmp_path / "report_data_demo.json"
    report_path.write_text(json.dumps({"metadata": {"model_id": "demo-model"}}))

    patched = brp.patch_missing_release_version(report_path, "v0.11.0")

    assert patched is True
    report_data = json.loads(report_path.read_text())
    assert report_data["metadata"]["release_version"] == "0.11.0"


def test_collect_reports_for_release_logs_processing_errors_and_continues(
    tmp_path, caplog
):
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {
                "name": "v0.11.0-release_artifacts.zip",
                "url": "https://api.github.com/assets/1",
            }
        ],
    }
    release_bundle = build_release_bundle_zip_bytes(
        {
            "broken_model": build_model_artifact_zip_bytes(
                "broken-model",
                include_report=True,
                report_payload="{not valid json",
            ),
            "good_model": build_model_artifact_zip_bytes(
                "good-model",
                include_report=True,
                report_payload={"metadata": {"model_id": "good-model"}},
            ),
        }
    )

    caplog.set_level(logging.ERROR)
    with patch.object(
        brp, "download_release_asset_zip", return_value=release_bundle
    ), patch.object(
        brp, "merge_report_into_release_performance", return_value=True
    ) as merge_mock:
        stats = brp.collect_reports_for_release(
            release=release,
            output_root=tmp_path,
            current_model_ids={"broken-model", "good-model"},
            token="token",
            release_performance_data={"schema_version": "0.1.0", "models": {}},
        )

    assert stats.reports_copied == 2
    assert stats.reports_patched == 1
    assert stats.report_processing_errors == 1
    assert stats.baseline_entries_updated == 1
    assert merge_mock.call_count == 1
    merge_mock.assert_called_once_with(
        tmp_path
        / "backfill"
        / "v0.11.0"
        / "reports"
        / "good-model"
        / "report_data_good-model_123.json",
        "good-model",
        ANY,
    )
    assert (
        "Failed to process staged report data for current model broken-model"
        in caplog.text
    )


def test_collect_reports_for_release_skips_invalid_nested_zip(tmp_path, caplog):
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {
                "name": "v0.11.0-release_artifacts.zip",
                "url": "https://api.github.com/assets/1",
            }
        ],
    }
    release_bundle = build_release_bundle_zip_bytes(
        {
            "demo_model": build_model_artifact_zip_bytes(
                "demo-model", include_report=True
            ),
            "not_a_zip": b"this is not actually a zip file",
        }
    )

    caplog.set_level(logging.WARNING)
    with patch.object(
        brp, "download_release_asset_zip", return_value=release_bundle
    ), patch.object(brp, "merge_report_into_release_performance", return_value=True):
        stats = brp.collect_reports_for_release(
            release=release,
            output_root=tmp_path,
            current_model_ids={"demo-model"},
            token="token",
            release_performance_data={"schema_version": "0.1.0", "models": {}},
        )

    assert stats.reports_copied == 1
    assert stats.report_processing_errors == 0
    assert "Skipping nested archive that is not a valid zip file" in caplog.text


def test_merge_report_into_release_performance_uses_shared_updater(tmp_path):
    report_path = tmp_path / "report_data_demo-model.json"
    report_path.write_text(json.dumps({"metadata": {"model_id": "demo-model"}}))
    model_spec = SimpleNamespace(
        model_name="DemoModel",
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_id="demo_impl"),
        inference_engine="vLLM",
    )
    record = SimpleNamespace(
        model_id="demo-model",
        model_spec=model_spec,
        ci_data={"release_report": {"metadata": {"release_version": "0.11.0"}}},
    )
    update_result = SimpleNamespace(
        artifacts=SimpleNamespace(
            records_with_entries=(
                SimpleNamespace(baseline_entry={"release_version": "0.11.0"}),
            )
        ),
        final_baseline_data={
            "schema_version": "0.1.0",
            "models": {
                "DemoModel": {
                    "n150": {
                        "demo_impl": {
                            "vLLM": {
                                "perf_status": "target",
                                "release_version": "0.11.0",
                                "ci_run_number": None,
                                "ci_run_url": None,
                                "ci_job_url": None,
                                "perf_target_results": [],
                            }
                        }
                    }
                }
            },
        },
        updated_count=1,
    )
    release_performance_data = {"schema_version": "0.1.0", "models": {}}

    with patch.object(
        brp,
        "build_merged_spec_from_report_data_json",
        return_value={"demo-model": record},
    ), patch.object(
        brp,
        "update_release_performance_outputs",
        return_value=update_result,
    ) as update_mock:
        updated = brp.merge_report_into_release_performance(
            report_path, "demo-model", release_performance_data
        )

    assert updated is True
    assert update_mock.call_args.args[0] == [record]
    assert (
        update_mock.call_args.kwargs["mode"]
        == brp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY
    )
    assert (
        update_mock.call_args.kwargs["existing_baseline_data"]
        is release_performance_data
    )
    assert (
        release_performance_data["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"][
            "release_version"
        ]
        == "0.11.0"
    )


def test_merge_report_into_release_performance_skips_when_no_perf_entry(tmp_path):
    report_path = tmp_path / "report_data_demo-model.json"
    report_path.write_text(json.dumps({"metadata": {"model_id": "demo-model"}}))
    model_spec = SimpleNamespace(
        model_name="DemoModel",
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_id="demo_impl"),
        inference_engine="vLLM",
    )
    record = SimpleNamespace(
        model_id="demo-model",
        model_spec=model_spec,
        ci_data={"release_report": {"metadata": {"release_version": "0.11.0"}}},
    )
    update_result = SimpleNamespace(
        artifacts=SimpleNamespace(records_with_entries=tuple()),
        final_baseline_data={"schema_version": "0.1.0", "models": {}},
        updated_count=0,
    )
    release_performance_data = {"schema_version": "0.1.0", "models": {}}

    with patch.object(
        brp,
        "build_merged_spec_from_report_data_json",
        return_value={"demo-model": record},
    ), patch.object(
        brp,
        "update_release_performance_outputs",
        return_value=update_result,
    ):
        updated = brp.merge_report_into_release_performance(
            report_path, "demo-model", release_performance_data
        )

    assert updated is False
    assert release_performance_data == {"schema_version": "0.1.0", "models": {}}


def test_collect_reports_for_release_patches_release_version_before_shared_updater(
    tmp_path,
):
    release = {
        "tag_name": "v0.11.0",
        "assets": [
            {
                "name": "v0.11.0-release_artifacts.zip",
                "url": "https://api.github.com/assets/1",
            }
        ],
    }
    release_bundle = build_release_bundle_zip_bytes(
        {
            "demo_model": build_model_artifact_zip_bytes(
                "demo-model",
                include_report=True,
                report_payload={"metadata": {"model_id": "demo-model"}},
            ),
        }
    )
    model_spec = SimpleNamespace(
        model_name="DemoModel",
        device_type=SimpleNamespace(name="N150"),
        impl=SimpleNamespace(impl_id="demo_impl"),
        inference_engine="vLLM",
    )
    record = SimpleNamespace(
        model_id="demo-model",
        model_spec=model_spec,
        ci_data={"release_report": {"metadata": {"release_version": "0.11.0"}}},
    )
    update_result = SimpleNamespace(
        artifacts=SimpleNamespace(
            records_with_entries=(
                SimpleNamespace(baseline_entry={"release_version": "0.11.0"}),
            )
        ),
        final_baseline_data={"schema_version": "0.1.0", "models": {}},
        updated_count=1,
    )

    def build_merged_spec_side_effect(report_path, is_dev):
        assert is_dev is False
        report_data = json.loads(report_path.read_text())
        assert report_data["metadata"]["release_version"] == "0.11.0"
        return {"demo-model": record}

    with patch.object(
        brp, "download_release_asset_zip", return_value=release_bundle
    ), patch.object(
        brp,
        "build_merged_spec_from_report_data_json",
        side_effect=build_merged_spec_side_effect,
    ), patch.object(
        brp,
        "update_release_performance_outputs",
        return_value=update_result,
    ) as update_mock:
        stats = brp.collect_reports_for_release(
            release=release,
            output_root=tmp_path,
            current_model_ids={"demo-model"},
            token="token",
            release_performance_data={"schema_version": "0.1.0", "models": {}},
        )

    assert stats.reports_copied == 1
    assert stats.reports_patched == 1
    assert stats.baseline_entries_updated == 1
    assert (
        update_mock.call_args.kwargs["mode"]
        == brp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY
    )

from types import SimpleNamespace
from unittest.mock import patch

import scripts.release.release_performance as rp
from workflows.perf_targets import PerfTarget, PerfTargetSet


def make_model_spec(
    *,
    model_name="DemoModel",
    device_name="N150",
    impl_id="demo_impl",
    inference_engine="vLLM",
):
    return SimpleNamespace(
        model_name=model_name,
        device_type=SimpleNamespace(name=device_name),
        impl=SimpleNamespace(impl_id=impl_id),
        inference_engine=inference_engine,
    )


def make_entry(release_version):
    return {
        "perf_status": "target",
        "release_version": release_version,
        "ci_run_number": None,
        "ci_run_url": None,
        "ci_job_url": None,
        "perf_target_results": [],
    }


def make_report_entry():
    entry = make_entry("0.12.0")
    entry["ci_run_url"] = "https://example.com/runs/123"
    entry["report_markdown"] = """## Release Report

### Acceptance Criteria

- Acceptance status: `PASS`

### Performance Benchmark Sweeps for DemoModel on n150

#### vLLM Text-to-Text Performance Benchmark Sweeps for DemoModel on n150

| ISL | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) |
| --- | --- | --- | --- | --- | --- |
| 128 | 128 | 1 | 42.0 | 11.0 | 22.0 |

### Benchmark Performance Results for DemoModel on n150

#### AIPerf Text Benchmarks - Detailed Percentiles

Canonical AIPerf detailed benchmark content.

### GenAI-Perf Benchmark Performance Results for DemoModel on n150

#### GenAI-Perf Text Benchmarks - Detailed Percentiles

Canonical GenAI-Perf detailed benchmark content.

### Accuracy Evaluations for DemoModel on n150

#### Canonical Evals Summary

| Task Name | Score |
| --- | --- |
| demo_eval | 0.95 |

### Test Results for DemoModel on n150

#### Canonical Parameter Support

| Test Case | Status |
| --- | --- |
| `test_smoke` | PASS |

### Stress Test Results for DemoModel on n150

#### Canonical Stress Summary

| ISL | OSL | Concurrency |
| --- | --- | --- |
| 128 | 32 | 1 |

### Server Test Results for DemoModel on n150

#### smoke_suite

All canonical server checks passed.
"""
    return entry


def make_record(*, release_version="0.12.0", model_spec=None):
    model_spec = model_spec or make_model_spec()
    return SimpleNamespace(
        model_id=f"{model_spec.model_name.lower()}-{model_spec.device_type.name.lower()}",
        model_spec=model_spec,
        ci_data={
            "perf_status": "target",
            "ci_run_number": 123,
            "ci_run_url": "https://example.com/runs/123",
            "ci_job_url": "https://example.com/jobs/456",
            "release_report": {
                "metadata": {"release_version": release_version},
                "benchmarks_summary": [],
            },
        },
    )


def test_write_release_performance_data_rejects_schema_invalid_entry(tmp_path):
    invalid_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_report_entry(),
                    }
                }
            }
        },
    }

    try:
        rp.write_release_performance_data(
            invalid_data, path=tmp_path / "release_performance.json"
        )
    except RuntimeError as exc:
        assert "schema validation failed" in str(exc)
        assert "report_markdown" in str(exc)
    else:
        raise AssertionError(
            "Expected schema-invalid release performance write to fail"
        )


def test_merge_release_performance_entry_replaces_older_release_version():
    model_spec = make_model_spec()
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_entry("0.9.0"),
                    }
                }
            }
        },
    }

    updated = rp.merge_release_performance_entry(
        release_performance_data,
        model_spec,
        make_entry("0.10.0"),
    )

    assert updated is True
    assert (
        release_performance_data["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"][
            "release_version"
        ]
        == "0.10.0"
    )


def test_merge_release_performance_entry_skips_same_or_newer_release_version():
    model_spec = make_model_spec()
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_entry("0.11.0"),
                    }
                }
            }
        },
    }

    same_version_updated = rp.merge_release_performance_entry(
        release_performance_data,
        model_spec,
        make_entry("0.11.0"),
    )
    older_version_updated = rp.merge_release_performance_entry(
        release_performance_data,
        model_spec,
        make_entry("0.10.0"),
    )

    assert same_version_updated is False
    assert older_version_updated is False
    assert (
        release_performance_data["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"][
            "release_version"
        ]
        == "0.11.0"
    )


def test_build_release_performance_artifacts_returns_rich_and_stripped_payloads():
    artifacts = rp.build_release_performance_artifacts([make_record()])

    rich_entry = artifacts.rich_data["models"]["DemoModel"]["n150"]["demo_impl"]["vLLM"]
    baseline_entry = artifacts.baseline_data["models"]["DemoModel"]["n150"][
        "demo_impl"
    ]["vLLM"]

    assert rich_entry["report_data"]["metadata"]["release_version"] == "0.12.0"
    assert "report_data" not in baseline_entry
    assert (
        artifacts.records_with_entries[0].baseline_entry["release_version"] == "0.12.0"
    )
    rp.validate_release_performance_data(artifacts.baseline_data)


def test_update_release_performance_outputs_replace_mode_writes_stripped_baseline():
    update_result = rp.update_release_performance_outputs(
        [make_record()],
        mode=rp.ReleasePerformanceWriteMode.REPLACE,
    )

    rich_entry = update_result.artifacts.rich_data["models"]["DemoModel"]["n150"][
        "demo_impl"
    ]["vLLM"]
    baseline_entry = update_result.final_baseline_data["models"]["DemoModel"]["n150"][
        "demo_impl"
    ]["vLLM"]

    assert update_result.updated_count == 1
    assert rich_entry["report_data"]["metadata"]["release_version"] == "0.12.0"
    assert "report_data" not in baseline_entry
    rp.validate_release_performance_data(update_result.final_baseline_data)


def test_update_release_performance_outputs_merge_newer_only_preserves_other_models():
    existing_baseline_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_entry("0.10.0"),
                    }
                }
            },
            "OtherModel": {
                "t3k": {
                    "other_impl": {
                        "TensorRT": make_entry("0.9.0"),
                    }
                }
            },
        },
    }

    update_result = rp.update_release_performance_outputs(
        [make_record(release_version="0.11.0")],
        mode=rp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY,
        existing_baseline_data=existing_baseline_data,
    )

    assert update_result.updated_count == 1
    assert (
        update_result.final_baseline_data["models"]["DemoModel"]["n150"]["demo_impl"][
            "vLLM"
        ]["release_version"]
        == "0.11.0"
    )
    assert (
        update_result.final_baseline_data["models"]["OtherModel"]["t3k"]["other_impl"][
            "TensorRT"
        ]["release_version"]
        == "0.9.0"
    )


def test_update_release_performance_outputs_merge_newer_only_skips_same_or_older():
    existing_baseline_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": make_entry("0.11.0"),
                    }
                }
            }
        },
    }

    same_version_result = rp.update_release_performance_outputs(
        [make_record(release_version="0.11.0")],
        mode=rp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY,
        existing_baseline_data=existing_baseline_data,
    )
    older_version_result = rp.update_release_performance_outputs(
        [make_record(release_version="0.10.0")],
        mode=rp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY,
        existing_baseline_data=existing_baseline_data,
    )

    assert same_version_result.updated_count == 0
    assert older_version_result.updated_count == 0
    assert (
        older_version_result.final_baseline_data["models"]["DemoModel"]["n150"][
            "demo_impl"
        ]["vLLM"]["release_version"]
        == "0.11.0"
    )


def test_update_release_performance_outputs_normalizes_release_version_before_merge():
    update_result = rp.update_release_performance_outputs(
        [make_record(release_version="v0.11.0")],
        mode=rp.ReleasePerformanceWriteMode.MERGE_NEWER_ONLY,
        existing_baseline_data={"schema_version": "0.1.0", "models": {}},
    )

    rich_entry = update_result.artifacts.rich_data["models"]["DemoModel"]["n150"][
        "demo_impl"
    ]["vLLM"]
    baseline_entry = update_result.final_baseline_data["models"]["DemoModel"]["n150"][
        "demo_impl"
    ]["vLLM"]

    assert update_result.updated_count == 1
    assert rich_entry["release_version"] == "0.11.0"
    assert baseline_entry["release_version"] == "0.11.0"
    assert "report_data" not in baseline_entry


def test_build_release_performance_entry_skips_all_null_measured_metrics():
    perf_target = PerfTargetSet(
        model_name="DemoModel",
        device=SimpleNamespace(name="N150"),
        perf_targets=[
            PerfTarget(
                isl=128,
                osl=128,
                max_concurrency=1,
                task_type="text",
                ttft_ms=50.0,
                tput_user=10.0,
                is_summary=True,
            )
        ],
    )
    ci_data = {
        "perf_status": "target",
        "ci_run_number": 123,
        "ci_run_url": "https://example.com/runs/123",
        "ci_job_url": "https://example.com/jobs/456",
        "release_report": {
            "metadata": {"release_version": "0.12.0"},
            "benchmarks_summary": [],
        },
    }

    with patch.object(rp, "get_perf_target", return_value=perf_target):
        entry = rp.build_release_performance_entry(
            make_model_spec(),
            ci_data,
            include_report_data=True,
        )

    assert entry is None


def test_render_release_report_section_prefers_full_report_data():
    markdown = rp.render_release_report_section(make_report_entry())

    assert "## Release Report" in markdown
    assert "Source: [CI run](https://example.com/runs/123)" in markdown
    assert "### Acceptance Criteria" in markdown
    assert "#### vLLM Text-to-Text Performance Benchmark Sweeps" in markdown
    assert "#### AIPerf Text Benchmarks - Detailed Percentiles" in markdown
    assert "#### GenAI-Perf Text Benchmarks - Detailed Percentiles" in markdown
    assert "#### Canonical Evals Summary" in markdown
    assert "#### Canonical Parameter Support" in markdown
    assert "#### Canonical Stress Summary" in markdown
    assert "#### smoke_suite" in markdown


def test_render_release_report_section_falls_back_without_prebuilt_markdown():
    entry = make_entry("0.12.0")
    entry["ci_run_url"] = "https://example.com/runs/123"
    entry["perf_target_results"] = [
        {
            "is_summary_data_point": True,
            "config": {
                "task_type": "text",
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
            },
            "measured_metrics": {
                "ttft": 42.0,
                "tput_user": 11.0,
                "tput": 22.0,
                "ttft_streaming_ms": None,
                "tput_prefill": None,
                "e2el_ms": None,
                "rtr": None,
            },
        }
    ]

    markdown = rp.render_release_report_section(entry)

    assert "## Release Report" in markdown
    assert "Source: [CI run](https://example.com/runs/123)" in markdown
    assert "### Performance Benchmark Targets" in markdown
    assert "#### Text-to-Text" in markdown
    assert "#### Canonical Evals Summary" not in markdown

import json
import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows import run_reports
from workflows.reports_schema import (
    generate_reports_schema,
    load_reports_schema,
    validate_report_data,
    validate_report_file,
    write_reports_schema,
)


def make_benchmark_target_evaluation():
    support_level = {"checked": True, "passed": True, "failures": []}
    return {
        "reference_available": True,
        "status": "functional",
        "support_levels": {
            "functional": support_level,
            "complete": support_level,
            "target": support_level,
        },
        "next_status": "complete",
        "next_status_failures": [],
        "regression": {"checked": False, "passed": True, "failures": []},
        "errors": [],
    }


def make_report_data():
    return {
        "metadata": {
            "report_id": "demo-model_2026-03-25_12-00-00",
            "model_name": "DemoModel",
            "model_id": "demo-model",
            "runtime_model_spec_json": "/tmp/runtime_model_spec.json",
            "model_repo": "demo/model",
            "model_impl": "demo_impl",
            "inference_engine": "vLLM",
            "device": "N150",
            "server_mode": "API",
            "tt_metal_commit": "a" * 40,
            "vllm_commit": "b" * 7,
            "run_command": "python run.py --model demo-model --tt-device N150 --workflow release ",
        },
        "benchmarks_summary": [
            {
                "task_type": "text",
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "max_con": 1,
                "mean_ttft_ms": 60.0,
                "mean_tps": 15.0,
            }
        ],
        "aiperf_benchmarks": [],
        "evals": [
            {
                "model": "DemoModel",
                "device": "N150",
                "task_name": "hellaswag",
                "accuracy_check": 2,
                "score": 0.77,
                "gpu_reference_score": 0.8,
                "ratio_to_reference": 0.9625,
            }
        ],
        "stress_tests": None,
        "benchmarks": [{"model_id": "demo-model", "device": "N150"}],
        "aiperf_benchmarks_detailed": [],
        "parameter_support_tests": {
            "results": {
                "test_temperature": [
                    {
                        "status": "passed",
                        "message": "",
                        "test_node_name": "test_temperature[param]",
                    }
                ]
            }
        },
        "spec_tests": {
            "reports": [
                {
                    "summary": {"failed": 0},
                    "tests": [
                        {
                            "test_name": "device_liveness",
                            "success": True,
                            "duration": 1.2,
                        }
                    ],
                }
            ],
            "results": [
                {
                    "test_name": "device_liveness",
                    "success": True,
                    "duration": 1.2,
                    "report_index": 0,
                    "test_index": 0,
                }
            ],
        },
        "benchmark_target_evaluation": make_benchmark_target_evaluation(),
        "acceptance_criteria": True,
        "acceptance_blockers": {},
        "acceptance_summary_markdown": "### Acceptance Criteria\n- Acceptance status: `PASS`",
    }


def _install_main_monkeypatches(
    monkeypatch, tmp_path, validate_hook, generate_report_schema=False
):
    args = Namespace(
        output_path=str(tmp_path / "reports_output"),
        runtime_model_spec_json=str(tmp_path / "runtime_model_spec.json"),
    )
    Path(args.runtime_model_spec_json).write_text("{}", encoding="utf-8")

    model_spec = SimpleNamespace(
        model_name="DemoModel",
        model_id="demo-model",
        hf_model_repo="demo/model",
        impl=SimpleNamespace(impl_name="demo_impl"),
        inference_engine="vLLM",
        tt_metal_commit="a" * 40,
        vllm_commit="b" * 7,
        device_type=run_reports.DeviceTypes.N150,
        device_model_spec=SimpleNamespace(default_impl=True),
    )
    runtime_config = SimpleNamespace(
        model="demo-model",
        device="N150",
        docker_server=False,
        percentile_report=False,
        generate_report_schema=generate_report_schema,
    )

    monkeypatch.setattr(run_reports, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_reports, "setup_workflow_script_logger", lambda logger: None
    )
    monkeypatch.setattr(
        run_reports.ModelSpec,
        "from_json",
        staticmethod(lambda _: model_spec),
    )
    monkeypatch.setattr(
        run_reports.RuntimeConfig,
        "from_json",
        staticmethod(lambda _: runtime_config),
    )
    monkeypatch.setattr(
        run_reports,
        "benchmark_generate_report",
        lambda *_, **__: (
            "benchmarks",
            make_report_data()["benchmarks_summary"],
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        run_reports,
        "aiperf_benchmark_generate_report",
        lambda *_, **__: ("", [], None, None),
    )
    monkeypatch.setattr(
        run_reports,
        "genai_perf_benchmark_generate_report",
        lambda *_, **__: ("", [], None, None),
    )
    monkeypatch.setattr(
        run_reports,
        "evals_generate_report",
        lambda *_, **__: ("evals", make_report_data()["evals"], None, None),
    )
    monkeypatch.setattr(
        run_reports,
        "generate_tests_report",
        lambda *_, **__: (
            "tests",
            make_report_data()["parameter_support_tests"],
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        run_reports,
        "stress_test_generate_report",
        lambda *_, **__: ("", None, None, None),
    )
    monkeypatch.setattr(
        run_reports,
        "server_tests_generate_report",
        lambda *_, **__: ("", make_report_data()["spec_tests"]["reports"]),
    )
    monkeypatch.setattr(
        run_reports,
        "evaluate_benchmark_targets",
        lambda *_: make_benchmark_target_evaluation(),
    )
    monkeypatch.setattr(
        run_reports,
        "acceptance_criteria_check",
        lambda *_, **__: (True, {}),
    )
    monkeypatch.setattr(
        run_reports,
        "format_acceptance_summary_markdown",
        lambda *_, **__: "### Acceptance Criteria\n- Acceptance status: `PASS`",
    )
    monkeypatch.setattr(run_reports, "validate_report_file", validate_hook)
    monkeypatch.setattr(run_reports, "project_root", tmp_path)

    return Path(args.output_path)


def test_validate_report_data_accepts_representative_release_report():
    validate_report_data(make_report_data())


def test_reports_schema_uses_draft_2020_12():
    assert (
        load_reports_schema()["$schema"]
        == "https://json-schema.org/draft/2020-12/schema"
    )


def test_write_reports_schema_generates_schema_from_report_data(tmp_path):
    schema_path = tmp_path / "reports-schema.json"
    report_data = make_report_data()

    written_path = write_reports_schema(report_data, schema_path)

    assert written_path == schema_path
    generated_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert generated_schema["$schema"] == load_reports_schema()["$schema"]
    validate_report_data(report_data, schema=generated_schema)


def test_generate_reports_schema_preserves_top_level_benchmarks_key():
    generated_schema = generate_reports_schema(make_report_data())

    assert generated_schema["properties"]["benchmarks"]["type"] == "array"


def test_generate_reports_schema_puts_metadata_keys_first():
    generated_schema = generate_reports_schema(make_report_data())

    assert list(generated_schema.keys())[:3] == [
        "$schema",
        "title",
        "description",
    ]


def test_validate_report_data_allows_current_placeholder_sections():
    report_data = make_report_data()
    report_data["evals"] = {"summary": "raw eval payload"}
    report_data["parameter_support_tests"] = [
        {"model": "demo-model", "device": "N150"},
    ]
    report_data["stress_tests"] = [{"isl": 128, "osl": 32}]

    validate_report_data(report_data)


def test_validate_report_data_rejects_missing_required_top_level_field(caplog):
    report_data = make_report_data()
    del report_data["benchmark_target_evaluation"]

    with caplog.at_level(logging.ERROR, logger="workflows.reports_schema"):
        validate_report_data(report_data)

    assert "benchmark_target_evaluation" in caplog.text
    assert "required property" in caplog.text


def test_validate_report_file_rejects_invalid_field_type(tmp_path, caplog):
    report_file = tmp_path / "report_data_demo-model.json"
    report_data = make_report_data()
    report_data["acceptance_criteria"] = "yes"
    report_file.write_text(json.dumps(report_data), encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="workflows.reports_schema"):
        validate_report_file(report_file)

    assert "acceptance_criteria" in caplog.text
    assert "is not of type 'boolean'" in caplog.text


def test_run_reports_main_validates_generated_raw_report(tmp_path, monkeypatch):
    validated_paths = []

    def validate_hook(report_file: Path):
        validated_paths.append(report_file)
        validate_report_file(report_file)

    output_path = _install_main_monkeypatches(monkeypatch, tmp_path, validate_hook)

    return_code = run_reports.main()

    assert return_code == 0
    assert len(validated_paths) == 1
    assert validated_paths[0].exists()
    assert validated_paths[0].parent == output_path / "release" / "data"


def test_run_reports_main_raises_after_writing_raw_report_on_validation_failure(
    tmp_path, monkeypatch
):
    validated_paths = []

    def validate_hook(report_file: Path):
        validated_paths.append(report_file)
        raise RuntimeError("schema failed")

    output_path = _install_main_monkeypatches(monkeypatch, tmp_path, validate_hook)

    with pytest.raises(RuntimeError, match="schema failed"):
        run_reports.main()

    assert len(validated_paths) == 1
    assert validated_paths[0].exists()
    assert list((output_path / "release").glob("report_*.md")) == []


def test_run_reports_main_logs_missing_optional_aiperf_csv_as_info(
    tmp_path, monkeypatch, caplog
):
    output_path = _install_main_monkeypatches(
        monkeypatch, tmp_path, validate_report_file
    )
    missing_aiperf_csv = (
        output_path
        / "benchmarks_aiperf"
        / "data"
        / "aiperf_benchmark_text_stats_demo-model_2026-03-25_12-00-00.csv"
    )
    missing_aiperf_csv.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        run_reports,
        "aiperf_benchmark_generate_report",
        lambda *_, **__: ("", [], None, missing_aiperf_csv),
    )

    with caplog.at_level(logging.INFO, logger=run_reports.logger.name):
        return_code = run_reports.main()

    assert return_code == 0
    assert "Could not read AIPerf CSV data" not in caplog.text
    assert "AIPerf CSV data is optional and was not found" in caplog.text


def test_run_reports_main_generates_schema_from_raw_report_before_validation(
    tmp_path, monkeypatch
):
    observed_calls = []

    def validate_hook(report_file: Path):
        observed_calls.append(("validate", report_file.exists()))

    _install_main_monkeypatches(
        monkeypatch,
        tmp_path,
        validate_hook,
        generate_report_schema=True,
    )

    def write_schema_hook(report_file: Path):
        report_data = json.loads(report_file.read_text(encoding="utf-8"))
        observed_calls.append(
            (
                "generate",
                report_file.exists(),
                "benchmarks" in report_data,
                report_data["metadata"]["model_id"],
            )
        )
        return tmp_path / "reports-schema.json"

    monkeypatch.setattr(run_reports, "write_reports_schema", write_schema_hook)

    return_code = run_reports.main()

    assert return_code == 0
    assert observed_calls == [
        ("generate", True, True, "demo-model"),
        ("validate", True),
    ]

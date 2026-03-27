import json
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = (
    REPO_ROOT / "benchmarking" / "benchmark_targets" / "release_performance_schema.json"
)
BASELINE_PATH = (
    REPO_ROOT / "benchmarking" / "benchmark_targets" / "release_performance.json"
)


def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_release_performance_schema_validates_checked_in_baseline():
    schema = load_schema()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(instance=baseline, schema=schema)


def test_release_performance_schema_validates_representative_generated_entry():
    schema = load_schema()
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "perf_status": "target",
                            "ci_run_number": 123,
                            "ci_run_url": "https://example.com/runs/123",
                            "ci_job_url": "https://example.com/jobs/456",
                            "perf_target_results": [
                                {
                                    "is_summary_data_point": True,
                                    "config": {
                                        "task_type": "text",
                                        "isl": 128,
                                        "osl": 128,
                                        "max_concurrency": 1,
                                    },
                                    "targets": {
                                        "ttft_ms": 50.0,
                                        "ttft_streaming_ms": None,
                                        "tput_user": 10.0,
                                        "tput_prefill": None,
                                        "e2el_ms": None,
                                        "tput": 12.0,
                                        "rtr": None,
                                        "tolerance": 0.05,
                                    },
                                    "measured_metrics": {
                                        "ttft": 45.0,
                                        "tput_user": 12.0,
                                        "tput": 12.0,
                                        "ttft_streaming_ms": None,
                                        "tput_prefill": None,
                                        "e2el_ms": None,
                                        "rtr": None,
                                    },
                                    "benchmark_summary": {
                                        "task_type": "text",
                                        "isl": 128,
                                        "osl": 128,
                                        "max_concurrency": 1,
                                        "ttft": 45.0,
                                        "tput_user": 12.0,
                                    },
                                }
                            ],
                        }
                    }
                }
            }
        },
    }

    jsonschema.validate(instance=release_performance_data, schema=schema)

from scripts.release.generate_model_support_docs import (
    HardwarePageGroup,
    generate_model_support_docs,
    generate_model_page_group_page,
    generate_model_type_page,
    generate_models_by_hardware_page,
)
from unittest.mock import patch
from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpecTemplate
from workflows.workflow_types import DeviceTypes, ModelStatusTypes, ModelType


def make_template():
    return ModelSpecTemplate(
        weights=["acme/DemoModel"],
        impl=ImplSpec(
            impl_id="demo_impl",
            impl_name="demo-impl",
            repo_url="https://github.com/tenstorrent/demo",
            code_path="models/demo",
        ),
        tt_metal_commit="aaaaaaa",
        vllm_commit="1111111",
        inference_engine="vLLM",
        device_model_specs=[
            DeviceModelSpec(
                device=DeviceTypes.N150,
                max_concurrency=1,
                max_context=2048,
            )
        ],
        status=ModelStatusTypes.COMPLETE,
        docker_image="ghcr.io/tenstorrent/demo:tag",
        model_type=ModelType.LLM,
    )


def test_generate_model_page_group_page_embeds_matching_release_report():
    release_performance_data = {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "ci_run_url": "https://example.com/runs/123",
                            "benchmarks_summary": [
                                {
                                    "task_type": "text",
                                    "isl": 128,
                                    "osl": 128,
                                    "max_concurrency": 1,
                                    "ttft": 42.0,
                                    "tput_user": 11.0,
                                    "target_checks": {
                                        "target": {
                                            "ttft_check": 2,
                                            "tput_user_check": 2,
                                            "ttft": 50.0,
                                            "tput_user": 10.0,
                                        }
                                    },
                                }
                            ],
                            "report_data": {
                                "benchmarks_summary": [
                                    {
                                        "task_type": "text",
                                        "isl": 128,
                                        "osl": 128,
                                        "max_concurrency": 1,
                                        "ttft": 42.0,
                                        "tput_user": 11.0,
                                        "target_checks": {
                                            "target": {
                                                "ttft_check": 2,
                                                "tput_user_check": 2,
                                                "ttft": 50.0,
                                                "tput_user": 10.0,
                                            }
                                        },
                                    }
                                ],
                                "parameter_support_tests": {
                                    "endpoint_url": "http://localhost:8000",
                                    "test_run_timestamp_utc": "2026-03-11T12:00:00",
                                    "results": {
                                        "test_smoke": [
                                            {
                                                "status": "passed",
                                                "test_node_name": "test_smoke[param]",
                                                "message": "",
                                            }
                                        ]
                                    },
                                },
                            },
                        }
                    }
                }
            }
        },
    }

    markdown = generate_model_page_group_page(
        "DemoModel",
        [make_template()],
        HardwarePageGroup.from_device_type(DeviceTypes.N150),
        release_performance_data=release_performance_data,
    )

    assert "## Release Report" in markdown
    assert "### Performance Benchmark Targets" in markdown
    assert "### Test Results" in markdown
    assert "### LLM API Test Metadata" in markdown
    assert "Source: [CI run](https://example.com/runs/123)" in markdown


def test_generate_model_page_group_page_skips_release_report_without_match():
    markdown = generate_model_page_group_page(
        "DemoModel",
        [make_template()],
        HardwarePageGroup.from_device_type(DeviceTypes.N150),
        release_performance_data={"schema_version": "0.1.0", "models": {}},
    )

    assert "Release Report" not in markdown


def make_summary_release_performance_data():
    return {
        "schema_version": "0.1.0",
        "models": {
            "DemoModel": {
                "n150": {
                    "demo_impl": {
                        "vLLM": {
                            "model": "DemoModel",
                            "device": "n150",
                            "impl_id": "demo_impl",
                            "inference_engine": "vLLM",
                            "perf_target_summary": {
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
                            },
                        }
                    }
                }
            }
        },
    }


def test_generate_model_type_page_renders_llm_summary_metrics():
    markdown = generate_model_type_page(
        [make_template()],
        ModelType.LLM,
        release_performance_data=make_summary_release_performance_data(),
    )

    assert "Output Tput: 22 (tok/s), TTFT: 42 (ms)" in markdown
    assert "Input Sequence Length: 128 tokens" in markdown
    assert "Concurrency: 1" in markdown


def test_generate_models_by_hardware_page_renders_performance_summary_column():
    markdown = generate_models_by_hardware_page(
        [make_template()],
        release_performance_data=make_summary_release_performance_data(),
    )

    assert "| Status | Type | Model | Performance Summary |" in markdown
    assert "11 (tok/s/user), TTFT: 42 (ms)" in markdown


def test_generate_model_support_docs_cleans_stale_generated_files_and_uses_passed_perf_data(
    tmp_path,
):
    output_dir = tmp_path / "docs" / "model_support"
    stale_file = output_dir / "llm" / "stale.md"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale")
    preserved_file = output_dir / "custom.md"
    preserved_file.write_text("keep me")

    with patch(
        "scripts.release.generate_model_support_docs.load_templates_from_model_spec",
        return_value=[make_template()],
    ), patch(
        "scripts.release.generate_model_support_docs.load_release_performance_data",
        side_effect=AssertionError("should use explicit release performance data"),
    ):
        generate_model_support_docs(
            model_spec_path=tmp_path / "workflows" / "model_spec.py",
            output_dir=output_dir,
            release_performance_data=make_summary_release_performance_data(),
        )

    assert not stale_file.exists()
    assert preserved_file.exists()
    assert (output_dir / "models_by_hardware.md").exists()
    assert (output_dir / "llm" / "README.md").exists()

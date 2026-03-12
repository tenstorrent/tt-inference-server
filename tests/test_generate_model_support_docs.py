from scripts.release.generate_model_support_docs import (
    HardwarePageGroup,
    generate_model_page_group_page,
)
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
        "schema_version": 1,
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
        release_performance_data={"schema_version": 1, "models": {}},
    )

    assert "Release Report" not in markdown

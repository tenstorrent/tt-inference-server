from unittest.mock import patch

from scripts.release.generate_model_support_docs import (
    HardwarePageGroup,
    generate_model_page_group_page,
    generate_model_support_docs,
    generate_model_type_page,
    generate_models_by_hardware_page,
)
from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpecTemplate
from workflows.workflow_types import DeviceTypes, ModelStatusTypes, ModelType


def make_template(
    model_name="DemoModel",
    *,
    device=DeviceTypes.N150,
    status=ModelStatusTypes.COMPLETE,
    model_type=ModelType.LLM,
    release_version=None,
    weights=None,
):
    return ModelSpecTemplate(
        weights=weights or [f"acme/{model_name}"],
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
                device=device,
                max_concurrency=1,
                max_context=2048,
            )
        ],
        status=status,
        docker_image="ghcr.io/tenstorrent/demo:tag",
        model_type=model_type,
        release_version=release_version,
    )


def make_release_entry(
    *,
    release_version="0.12.0",
    ci_run_url=None,
    tput_user=11.0,
    tput=22.0,
    ttft=42.0,
    report_markdown=None,
):
    return {
        "release_version": release_version,
        "ci_run_url": ci_run_url,
        "perf_target_results": [
            {
                "is_summary_data_point": True,
                "config": {
                    "task_type": "text",
                    "isl": 128,
                    "osl": 128,
                    "max_concurrency": 1,
                },
                "measured_metrics": {
                    "ttft": ttft,
                    "tput_user": tput_user,
                    "tput": tput,
                    "ttft_streaming_ms": None,
                    "tput_prefill": None,
                    "e2el_ms": None,
                    "rtr": None,
                },
            }
        ],
        "report_markdown": report_markdown
        or make_report_markdown(
            model_name="DemoModel",
            device="n150",
            release_version=release_version,
            ttft=ttft,
            tput_user=tput_user,
            tput=tput,
        ),
    }


def make_release_performance_data(entries):
    models = {}
    for model_name, device, entry in entries:
        models.setdefault(model_name, {}).setdefault(device, {}).setdefault(
            "demo_impl", {}
        )["vLLM"] = entry
    return {"schema_version": "0.1.0", "models": models}


def make_report_data(
    *,
    model_name,
    device,
    release_version,
    ttft,
    tput_user,
    tput,
):
    return f"""## Release Report

### Acceptance Criteria

- Acceptance status: `PASS`

### Performance Benchmark Sweeps for {model_name} on {device}

#### vLLM Text-to-Text Performance Benchmark Sweeps for {model_name} on {device}

| ISL | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) |
| --- | --- | --- | --- | --- | --- |
| 128 | 128 | 1 | {ttft} | {tput_user} | {tput} |

#### AIPerf Text-to-Text Performance Benchmark Sweeps for {model_name} on {device}

| ISL | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) |
| --- | --- | --- | --- | --- | --- |
| 128 | 128 | 1 | 50.0 | 10.0 | 20.0 |

### Benchmark Performance Results for {model_name} on {device}

#### AIPerf Text Benchmarks - Detailed Percentiles

Canonical AIPerf detailed benchmark content.

### GenAI-Perf Benchmark Performance Results for {model_name} on {device}

#### GenAI-Perf Text Benchmarks - Detailed Percentiles

Canonical GenAI-Perf detailed benchmark content.

### Accuracy Evaluations for {model_name} on {device}

#### Canonical Evals Summary

| Task Name | Score |
| --- | --- |
| demo_eval | 0.95 |

### Test Results for {model_name} on {device}

#### Canonical Parameter Support

| Test Case | Status |
| --- | --- |
| `test_smoke` | PASS |

### Stress Test Results for {model_name} on {device}

#### Canonical Stress Summary

| ISL | OSL | Concurrency |
| --- | --- | --- |
| 128 | 32 | 1 |

### Server Test Results for {model_name} on {device}

#### smoke_suite

All canonical server checks passed.
"""


make_report_markdown = make_report_data


def test_generate_model_page_group_page_embeds_matching_release_report():
    release_performance_data = make_release_performance_data(
        [
            (
                "DemoModel",
                "n150",
                make_release_entry(
                    release_version="0.12.0",
                    ci_run_url="https://example.com/runs/123",
                ),
            )
        ]
    )

    markdown = generate_model_page_group_page(
        "DemoModel",
        [make_template()],
        HardwarePageGroup.from_device_type(DeviceTypes.N150),
        release_performance_data=release_performance_data,
    )

    assert "| Release Support | 🟢 v0.12.0 |" in markdown
    assert "## Release Report" in markdown
    assert "### Acceptance Criteria" in markdown
    assert (
        "#### vLLM Text-to-Text Performance Benchmark Sweeps for DemoModel on n150"
        in markdown
    )
    assert (
        "#### AIPerf Text-to-Text Performance Benchmark Sweeps for DemoModel on n150"
        in markdown
    )
    assert "#### AIPerf Text Benchmarks - Detailed Percentiles" in markdown
    assert "#### GenAI-Perf Text Benchmarks - Detailed Percentiles" in markdown
    assert "#### Canonical Evals Summary" in markdown
    assert "#### Canonical Parameter Support" in markdown
    assert "#### Canonical Stress Summary" in markdown
    assert "#### smoke_suite" in markdown
    assert "Source: [CI run](https://example.com/runs/123)" in markdown


def test_generate_model_page_group_page_skips_release_report_without_match():
    markdown = generate_model_page_group_page(
        "DemoModel",
        [make_template(release_version="0.9.0")],
        HardwarePageGroup.from_device_type(DeviceTypes.N150),
        release_performance_data={"schema_version": "0.1.0", "models": {}},
    )

    assert "Release Report" not in markdown
    assert "v0.9.0" not in markdown


def test_generate_model_type_page_renders_llm_release_badge_and_summary_metrics():
    markdown = generate_model_type_page(
        [make_template()],
        ModelType.LLM,
        release_performance_data=make_release_performance_data(
            [("DemoModel", "n150", make_release_entry(release_version="0.12.0"))]
        ),
    )

    assert (
        "[🟢 v0.12.0](DemoModel_n150.md)<br>Output Tput: 22 (tok/s)<br>TTFT: 42 (ms)"
        in markdown
    )
    assert "Input Sequence Length: 128 tokens" in markdown
    assert "Concurrency: 1" in markdown


def test_generate_model_type_page_falls_back_to_template_release_without_match():
    markdown = generate_model_type_page(
        [make_template(release_version="0.9.0")],
        ModelType.LLM,
        release_performance_data={"schema_version": "0.1.0", "models": {}},
    )

    assert (
        "| [DemoModel](DemoModel_n150.md) | [🟢 v0.9.0](DemoModel_n150.md) |"
        in markdown
    )


def test_generate_models_by_hardware_page_renders_performance_summary_column():
    markdown = generate_models_by_hardware_page(
        [make_template()],
        release_performance_data=make_release_performance_data(
            [("DemoModel", "n150", make_release_entry(release_version="0.12.0"))]
        ),
    )

    assert "| Release | Type | Model | Performance Summary |" in markdown
    assert (
        "| 🟢 v0.12.0 | LLM | [DemoModel](llm/DemoModel_n150.md) | "
        "11 (tok/s/user), TTFT: 42 (ms) |"
    ) in markdown


def test_generate_models_by_hardware_page_falls_back_to_template_release_without_match():
    markdown = generate_models_by_hardware_page(
        [make_template(release_version="0.9.0")],
        release_performance_data={"schema_version": "0.1.0", "models": {}},
    )

    assert "| 🟢 v0.9.0 | LLM | [DemoModel](llm/DemoModel_n150.md) | - |" in markdown


def test_generate_model_page_group_page_uses_concrete_device_release_entries():
    galaxy_group = HardwarePageGroup(
        name=DeviceTypes.GALAXY.to_product_str(),
        device_ordering=(DeviceTypes.GALAXY, DeviceTypes.GALAXY_T3K),
    )
    markdown = generate_model_page_group_page(
        "DemoModel",
        [
            make_template(device=DeviceTypes.GALAXY, status=ModelStatusTypes.COMPLETE),
            make_template(
                device=DeviceTypes.GALAXY_T3K, status=ModelStatusTypes.FUNCTIONAL
            ),
        ],
        galaxy_group,
        release_performance_data=make_release_performance_data(
            [
                (
                    "DemoModel",
                    "galaxy",
                    make_release_entry(
                        release_version="0.12.0",
                        ci_run_url="https://example.com/runs/galaxy",
                    ),
                ),
                (
                    "DemoModel",
                    "galaxy_t3k",
                    make_release_entry(
                        release_version="0.11.0",
                        ci_run_url="https://example.com/runs/t3k",
                    ),
                ),
            ]
        ),
    )

    assert "| Release Support | 🟢 v0.12.0 |" in markdown
    assert "| Release Support | 🟡 v0.11.0 |" in markdown
    assert "Source: [CI run](https://example.com/runs/galaxy)" in markdown
    assert "Source: [CI run](https://example.com/runs/t3k)" in markdown
    assert "## GALAXY_T3K Configuration" in markdown


def test_generate_models_by_hardware_page_matches_release_data_from_template_weights():
    markdown = generate_models_by_hardware_page(
        [
            make_template(
                model_name="Llama-3.2-1B",
                device=DeviceTypes.N300,
                status=ModelStatusTypes.EXPERIMENTAL,
                weights=[
                    "meta-llama/Llama-3.2-1B",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ],
            )
        ],
        release_performance_data=make_release_performance_data(
            [
                (
                    "Llama-3.2-1B-Instruct",
                    "n300",
                    make_release_entry(
                        release_version="0.12.0",
                        tput_user=56.0,
                        tput=56.0,
                        ttft=338.5,
                    ),
                )
            ]
        ),
    )

    assert (
        "| 🛠️ v0.12.0 | LLM | [Llama-3.2-1B](llm/Llama-3.2-1B_n300.md) | "
        "56 (tok/s/user), TTFT: 338.5 (ms) |"
    ) in markdown


def test_generate_model_type_page_matches_release_data_from_group_template_weights():
    markdown = generate_model_type_page(
        [
            make_template(
                model_name="Llama-3.2-1B",
                device=DeviceTypes.N150,
                status=ModelStatusTypes.FUNCTIONAL,
                release_version="0.12.0",
                weights=[
                    "meta-llama/Llama-3.2-1B",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ],
            ),
            make_template(
                model_name="Llama-3.2-1B",
                device=DeviceTypes.N300,
                status=ModelStatusTypes.FUNCTIONAL,
                release_version="0.12.0",
                weights=[
                    "meta-llama/Llama-3.2-1B",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ],
            ),
        ],
        ModelType.LLM,
        release_performance_data=make_release_performance_data(
            [
                (
                    "Llama-3.2-1B-Instruct",
                    "n300",
                    make_release_entry(
                        release_version="0.12.0",
                        tput_user=56.0,
                        tput=56.0,
                        ttft=338.5,
                    ),
                )
            ]
        ),
    )

    assert (
        "[🟡 v0.12.0](Llama-3.2-1B_n150.md)<br>Output Tput: 56 (tok/s)<br>"
        "TTFT: 338.5 (ms)"
    ) in markdown


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
            release_performance_data=make_release_performance_data(
                [("DemoModel", "n150", make_release_entry(release_version="0.12.0"))]
            ),
        )

    assert not stale_file.exists()
    assert preserved_file.exists()
    assert (output_dir / "models_by_hardware.md").exists()
    assert (output_dir / "llm" / "README.md").exists()

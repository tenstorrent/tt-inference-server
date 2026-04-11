import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from workflows import run_reports


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_text_payload(backend):
    return {
        "model_id": "demo-model-id",
        "backend": backend,
        "mean_ttft_ms": 50.0,
        "std_ttft_ms": 5.0,
        "mean_tpot_ms": 20.0,
        "std_tpot_ms": 2.0,
        "mean_e2el_ms": 120.0,
        "request_throughput": 4.5,
        "total_input_tokens": 256,
        "total_output_tokens": 64,
        "total_token_throughput": 128.0,
        "num_prompts": 2,
    }


def make_minimal_model_spec():
    return SimpleNamespace(
        model_id="demo-model",
        model_name="Demo Model",
        device_model_spec=SimpleNamespace(
            max_concurrency=4,
            max_context=4096,
            max_tokens_all_users=4096,
            perf_reference=[],
        ),
    )


def test_extract_params_from_filename_parses_multiple_backends():
    vllm_params = run_reports.extract_params_from_filename(
        "benchmark_demo-model_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2.json"
    )
    genai_params = run_reports.extract_params_from_filename(
        "genai_benchmark_qwen2.5-vl_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2_images-1_height-224_width-224.json"
    )
    aiperf_params = run_reports.extract_params_from_filename(
        "aiperf_benchmark_demo-model_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2.json"
    )

    assert vllm_params["task_type"] == "text"
    assert genai_params["task_type"] == "vlm"
    assert genai_params["images_per_prompt"] == 1
    assert aiperf_params["backend"] == "aiperf"


def test_process_benchmark_file_formats_text_metrics(tmp_path):
    json_path = write_json(
        tmp_path
        / "benchmark_demo-model_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2.json",
        make_text_payload("vllm"),
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "text"
    assert result["backend"] == "vllm"
    assert result["max_con"] == 2
    assert result["mean_tps"] == 50.0
    assert result["tps_decode_throughput"] == 100.0
    assert result["tps_prefill_throughput"] == 5120.0


def test_process_benchmark_file_formats_vlm_metrics(tmp_path):
    json_path = write_json(
        tmp_path
        / "genai_benchmark_qwen2.5-vl_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2_images-1_height-224_width-224.json",
        make_text_payload("genai-perf"),
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "vlm"
    assert result["backend"] == "genai-perf"
    assert result["images_per_prompt"] == 1
    assert result["image_height"] == 224
    assert result["image_width"] == 224


def test_process_benchmark_file_formats_audio_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_reports,
        "MODEL_SPECS",
        {
            "demo": SimpleNamespace(
                model_name="whisper-demo",
                model_type=run_reports.ModelType.AUDIO,
            )
        },
    )
    json_path = write_json(
        tmp_path / "benchmark_id_tt-transformers_whisper-demo_N150_12345.0.json",
        {
            "model": "whisper-demo",
            "benchmarks: ": {
                "benchmarks": {
                    "num_requests": 2,
                    "ttft": 0.25,
                    "accuracy_check": 0.98,
                    "t/s/u": 1.5,
                    "rtr": 0.9,
                }
            },
            "streaming_enabled": True,
            "preprocessing_enabled": False,
        },
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "audio"
    assert result["backend"] == "audio"
    assert result["mean_ttft_ms"] == 250.0
    assert result["accuracy_check"] == 0.98
    assert result["num_eval_runs"] == 2
    assert result["tput_user"] == 1.5
    assert result["rtr"] == 0.9


def test_process_benchmark_file_formats_image_metrics_with_perf_aliases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        run_reports,
        "MODEL_SPECS",
        {
            "demo": SimpleNamespace(
                model_name="sdxl-demo",
                model_type=run_reports.ModelType.IMAGE,
            )
        },
    )
    json_path = write_json(
        tmp_path / "benchmark_id_tt-metal_sdxl-demo_N150_12345.0.json",
        {
            "model": "sdxl-demo",
            "benchmarks: ": {
                "benchmarks": {
                    "num_requests": 2,
                    "concurrency": 1,
                    "num_inference_steps": 20,
                    "ttft": 12.5,
                    "inference_steps_per_second": 0.08,
                    "end_to_end_latency_ms": 13000,
                }
            },
        },
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "image"
    assert result["backend"] == "image"
    assert result["max_con"] == 1
    assert result["max_concurrency"] == 1
    assert result["num_inference_steps"] == 20
    assert result["tput_user"] == 0.08
    assert result["tput"] == 0.08
    assert result["e2el_ms"] == 13000


def test_process_benchmark_file_formats_embedding_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_reports,
        "MODEL_SPECS",
        {
            "demo": SimpleNamespace(
                model_name="embed-demo",
                model_type=run_reports.ModelType.EMBEDDING,
            )
        },
    )
    json_path = write_json(
        tmp_path / "benchmark_id_tt-metal_embed-demo_N150_12345.0.json",
        {
            "model": "embed-demo",
            "benchmarks: ": {
                "benchmarks": {
                    "num_requests": 3,
                    "isl": 256,
                    "concurrency": 2,
                    "embedding_dimension": 768,
                    "tput_user": 123.45,
                    "tput_prefill": 456.7,
                    "e2el": 12.34,
                    "req_tput": 5.67,
                }
            },
        },
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "embedding"
    assert result["backend"] == "embedding"
    assert result["isl"] == 256
    assert result["max_concurrency"] == 2
    assert result["embedding_dimension"] == 768
    assert result["mean_ttft_ms"] == run_reports.NOT_MEASURED_STR
    assert result["mean_tps"] == 123.45
    assert result["tput_user"] == 123.45
    assert result["tput_prefill"] == 456.7
    assert result["e2el_ms"] == 12.34


def test_process_benchmark_file_formats_video_metrics_with_perf_aliases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        run_reports,
        "MODEL_SPECS",
        {
            "demo": SimpleNamespace(
                model_name="video-demo",
                model_type=run_reports.ModelType.VIDEO,
            )
        },
    )
    json_path = write_json(
        tmp_path / "benchmark_id_tt-metal_video-demo_N150_12345.0.json",
        {
            "model": "video-demo",
            "benchmarks: ": {
                "benchmarks": {
                    "num_requests": 1,
                    "num_inference_steps": 40,
                    "ttft": 330.0,
                    "inference_steps_per_second": 0.12,
                    "end_to_end_latency_ms": 321490,
                }
            },
        },
    )

    result = run_reports.process_benchmark_file(str(json_path))

    assert result["task_type"] == "video"
    assert result["backend"] == "video"
    assert result["max_con"] == 1
    assert result["max_concurrency"] == 1
    assert result["tput_user"] == 0.12
    assert result["tput"] == 0.12
    assert result["e2el_ms"] == 321490


def test_get_markdown_table_escapes_pipes():
    markdown = run_reports.get_markdown_table(
        [
            {"Name": "foo|bar", "Value": "12.5"},
            {"Name": "baz", "Value": "2.0"},
        ]
    )

    assert "foo\\|bar" in markdown
    assert "Note: all metrics are means across benchmark run" in markdown


def test_benchmark_generate_report_writes_combined_artifacts_in_backend_order(
    tmp_path, monkeypatch
):
    workflow_root = tmp_path / "workflow_logs"
    benchmarks_output = workflow_root / "benchmarks_output"
    benchmarks_output.mkdir(parents=True)

    write_json(
        benchmarks_output
        / "benchmark_demo-model_N150_2026-03-19_12-00-00_isl-128_osl-32_maxcon-2_n-2.json",
        make_text_payload("vllm"),
    )
    write_json(
        benchmarks_output
        / "aiperf_benchmark_demo-model_N150_2026-03-19_12-00-01_isl-128_osl-32_maxcon-2_n-2.json",
        {
            "model_id": "demo-model-id",
            "mean_ttft_ms": 55.0,
            "std_ttft_ms": 5.0,
            "mean_tpot_ms": 22.0,
            "std_tpot_ms": 2.0,
            "mean_e2el_ms": 130.0,
            "request_throughput": 4.0,
            "total_input_tokens": 256,
            "total_output_tokens": 64,
            "total_token_throughput": 120.0,
            "num_prompts": 2,
        },
    )
    write_json(
        benchmarks_output
        / "genai_benchmark_demo-model_N150_2026-03-19_12-00-02_isl-128_osl-32_maxcon-2_n-2.json",
        make_text_payload("genai-perf"),
    )

    monkeypatch.setattr(
        run_reports,
        "get_default_workflow_root_log_dir",
        lambda: str(workflow_root),
    )

    args = Namespace(
        output_path=str(tmp_path / "reports_output"),
        device="N150",
        model="demo-model",
    )
    model_spec = make_minimal_model_spec()

    release_str, release_raw, disp_md_path, stats_file_path = (
        run_reports.benchmark_generate_report(
            args,
            server_mode=None,
            model_spec=model_spec,
            report_id="demo-report",
            metadata={"model_name": model_spec.model_name},
        )
    )

    markdown = disp_md_path.read_text(encoding="utf-8")

    assert len(release_raw) == 3
    assert release_str == markdown
    assert stats_file_path.name == "benchmark_stats_demo-report.csv"
    assert disp_md_path.name == "benchmark_display_demo-report.md"
    assert stats_file_path.exists()
    assert disp_md_path.exists()
    assert markdown.startswith(
        "### Performance Benchmark Sweeps for Demo Model on N150"
    )
    assert markdown.index(
        "#### vLLM Text-to-Text Performance Benchmark Sweeps"
    ) < markdown.index("#### AIPerf Text-to-Text Performance Benchmark Sweeps")
    assert markdown.index(
        "#### AIPerf Text-to-Text Performance Benchmark Sweeps"
    ) < markdown.index("#### GenAI-Perf Text-to-Text Performance Benchmark Sweeps")

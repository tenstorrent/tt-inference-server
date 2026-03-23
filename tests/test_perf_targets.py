import pytest

from workflows.perf_targets import (
    DeviceTypes,
    PerfTarget,
    PerfTargetSet,
    get_named_perf_reference,
    get_perf_reference_for_device,
    get_perf_reference_map,
    get_perf_target,
    get_regression_perf_target,
    get_performance_targets,
)


def test_get_perf_target_uses_first_datapoint_as_summary():
    perf_target_set = get_perf_target("Llama-3.3-70B-Instruct", DeviceTypes.GALAXY)

    assert perf_target_set is not None
    assert len(perf_target_set.perf_targets) >= 2
    assert (
        sum(perf_target.is_summary for perf_target in perf_target_set.perf_targets) == 1
    )
    assert perf_target_set.summary_perf_target.is_summary is True
    assert perf_target_set.summary_perf_target.isl == 128
    assert perf_target_set.summary_perf_target.osl == 128
    assert perf_target_set.summary_perf_target.max_concurrency == 1


@pytest.mark.parametrize(
    "model_name,device_str,model_type,expected_targets",
    [
        (
            "stable-diffusion-xl-base-1.0",
            "n150",
            "IMAGE",
            {
                "ttft_ms": 12500,
                "ttft_streaming_ms": None,
                "tput_user": 0.08,
                "tput": None,
                "rtr": None,
                "tolerance": 0.05,
                "max_concurrency": 1,
                "num_eval_runs": None,
                "task_type": "image",
            },
        ),
        (
            "distil-large-v3",
            "n150",
            "AUDIO",
            {
                "ttft_ms": 400,
                "ttft_streaming_ms": 842.3,
                "tput_user": 112.62,
                "tput": None,
                "rtr": 15.61,
                "tolerance": 0.05,
                "max_concurrency": 1,
                "num_eval_runs": 2,
                "task_type": "audio",
            },
        ),
    ],
)
def test_get_performance_targets_returns_summary_perf_target(
    model_name, device_str, model_type, expected_targets
):
    perf_target = get_performance_targets(
        model_name=model_name, device_str=device_str, model_type=model_type
    )

    assert isinstance(perf_target, PerfTarget)
    for field_name, expected_value in expected_targets.items():
        assert getattr(perf_target, field_name) == expected_value
    assert perf_target.is_summary is True


def test_get_perf_reference_map_builds_threshold_targets():
    perf_reference_map = get_perf_reference_map(
        "Qwen3-32B",
        {"functional": 0.10, "complete": 0.50, "target": 1.0},
    )

    summary_task = perf_reference_map[DeviceTypes.GALAXY][0]

    assert summary_task.isl == 128
    assert summary_task.osl == 128
    assert isinstance(summary_task.targets["target"], PerfTarget)
    assert summary_task.targets["target"].is_derived is True
    assert summary_task.targets["target"].target_name == "target"
    assert summary_task.targets["target"].ttft_ms == 30.0
    assert summary_task.targets["target"].tput == 5329.0
    assert summary_task.targets["complete"].ttft_ms == 60.0
    assert summary_task.targets["functional"].tput_user == pytest.approx(15.6)


def test_get_perf_reference_for_device_scales_data_parallel_targets():
    perf_reference_map = get_perf_reference_map(
        "Qwen3-32B",
        {"target": 1.0},
    )

    scaled_reference = get_perf_reference_for_device(
        device=DeviceTypes.GALAXY,
        override_tt_config={"data_parallel": 4},
        perf_reference_map=perf_reference_map,
    )

    assert scaled_reference
    assert scaled_reference[0].isl == 128
    assert scaled_reference[0].targets["target"].ttft_ms == 62.0
    assert scaled_reference[0].targets["target"].tput == 5328.0
    assert scaled_reference[0].max_concurrency == 1


def test_perf_target_matches_summary_measurement_row():
    perf_target_set = get_perf_target("Llama-3.3-70B-Instruct", DeviceTypes.GALAXY)
    benchmark_rows = [
        {
            "task_type": "text",
            "isl": 128,
            "osl": 128,
            "max_concurrency": 1,
            "ttft": 70.8,
            "tput_user": 44.0,
            "tput": 44.0,
        },
        {
            "task_type": "text",
            "isl": 2048,
            "osl": 128,
            "max_concurrency": 1,
            "ttft": 800.0,
            "tput_user": 80.0,
        },
    ]

    matched_row = perf_target_set.find_matching_row(benchmark_rows)

    assert matched_row == benchmark_rows[0]


def test_perf_target_matches_vlm_measurement_requires_image_identity_fields():
    perf_target = PerfTarget(
        isl=128,
        osl=32,
        max_concurrency=2,
        task_type="vlm",
        image_height=224,
        image_width=224,
        images_per_prompt=1,
    )

    assert perf_target.matches_measurement(
        {
            "task_type": "vlm",
            "input_sequence_length": 128,
            "output_sequence_length": 32,
            "max_con": 2,
            "image_height": 224,
            "image_width": 224,
            "images_per_prompt": 1,
        }
    )
    assert not perf_target.matches_measurement(
        {
            "task_type": "vlm",
            "input_sequence_length": 128,
            "output_sequence_length": 32,
            "max_con": 2,
            "image_height": 224,
            "image_width": 512,
            "images_per_prompt": 1,
        }
    )


def test_find_matching_row_uses_num_inference_steps_for_image_rows():
    perf_target_set = PerfTargetSet(
        model_name="DemoImageModel",
        device=DeviceTypes.N150,
        perf_targets=[
            PerfTarget(
                task_type="image",
                max_concurrency=1,
                num_inference_steps=20,
                ttft_ms=12500.0,
                tput_user=0.08,
                is_summary=True,
            )
        ],
    )
    benchmark_rows = [
        {
            "task_type": "image",
            "num_inference_steps": 28,
            "mean_ttft_ms": 60000.0,
            "inference_steps_per_second": 0.02,
        },
        {
            "task_type": "image",
            "num_inference_steps": 20,
            "mean_ttft_ms": 11000.0,
            "inference_steps_per_second": 0.09,
        },
    ]

    assert perf_target_set.find_matching_row(benchmark_rows) == benchmark_rows[1]


def test_perf_target_matches_audio_measurement_using_num_eval_runs():
    perf_target = PerfTarget(
        task_type="audio",
        max_concurrency=1,
        num_eval_runs=2,
        ttft_ms=400.0,
        tput_user=112.62,
    )

    assert perf_target.matches_measurement(
        {
            "task_type": "audio",
            "num_eval_runs": 2,
            "mean_ttft_ms": 250.0,
            "t/s/u": 112.62,
        }
    )
    assert not perf_target.matches_measurement(
        {
            "task_type": "audio",
            "num_eval_runs": 4,
            "mean_ttft_ms": 250.0,
            "t/s/u": 112.62,
        }
    )


def test_perf_target_matches_embedding_measurement_with_reference_wildcards():
    perf_target = PerfTarget(
        task_type="embedding",
        max_concurrency=2,
        tput_user=123.45,
    )

    assert perf_target.matches_measurement(
        {
            "task_type": "embedding",
            "input_sequence_length": 256,
            "max_con": 2,
            "mean_tps": 123.45,
        }
    )
    assert not perf_target.matches_measurement(
        {
            "task_type": "embedding",
            "input_sequence_length": 256,
            "max_con": 4,
            "mean_tps": 123.45,
        }
    )


def test_find_matching_perf_target_uses_reference_side_identity_rules():
    perf_target_set = PerfTargetSet(
        model_name="DemoEmbeddingModel",
        device=DeviceTypes.N150,
        perf_targets=[
            PerfTarget(
                isl=256,
                max_concurrency=2,
                task_type="embedding",
                tput_user=123.45,
                is_summary=True,
            )
        ],
    )

    matched_perf_target = perf_target_set.find_matching_perf_target(
        PerfTarget(
            max_concurrency=2,
            task_type="embedding",
            tput_user=100.0,
        )
    )

    assert matched_perf_target == perf_target_set.summary_perf_target


def test_perf_target_from_dict_maps_non_text_metric_aliases():
    perf_target = PerfTarget.from_dict(
        {
            "task_type": "video",
            "max_concurrency": 1,
            "num_inference_steps": 40,
            "targets": {
                "theoretical": {
                    "ttft_ms": 540000,
                    "end_to_end_latency_ms": 971590,
                    "inference_steps_per_second": 0.0412,
                }
            },
        },
        is_summary=True,
    )

    assert perf_target.ttft_ms == 540000
    assert perf_target.e2el_ms == 971590
    assert perf_target.tput == pytest.approx(0.0412)


def test_build_threshold_targets_supports_non_llm_metrics():
    perf_target = PerfTarget(
        ttft_ms=200.0,
        ttft_streaming_ms=100.0,
        tput_user=50.0,
        tput_prefill=4000.0,
        e2el_ms=800.0,
        tput=250.0,
        rtr=2.0,
        tolerance=0.05,
    )

    targets = perf_target.build_threshold_targets(
        {"functional": 0.10, "complete": 0.50, "target": 1.0}
    )

    assert targets["functional"].target_name == "functional"
    assert targets["functional"].is_derived is True
    assert targets["functional"].ttft_ms == pytest.approx(2000.0)
    assert targets["functional"].ttft_streaming_ms == pytest.approx(1000.0)
    assert targets["functional"].tput_user == pytest.approx(5.0)
    assert targets["functional"].tput_prefill == pytest.approx(400.0)
    assert targets["functional"].e2el_ms == pytest.approx(8000.0)
    assert targets["functional"].tput == pytest.approx(25.0)
    assert targets["functional"].rtr == pytest.approx(0.2)


def test_get_regression_perf_target_reads_release_baseline(monkeypatch):
    monkeypatch.setattr(
        "workflows.perf_targets.release_performance_reference",
        {
            "schema_version": 1,
            "models": {
                "DemoModel": {
                    "n150": {
                        "demo_impl": {
                            "vLLM": {
                                "perf_target_results": [
                                    {
                                        "is_summary_data_point": True,
                                        "config": {
                                            "task_type": "text",
                                            "isl": 128,
                                            "osl": 128,
                                            "max_concurrency": 1,
                                        },
                                        "targets": {},
                                        "measured_metrics": {
                                            "ttft": 45.0,
                                            "tput_user": 11.0,
                                            "tput": 12.0,
                                        },
                                    }
                                ]
                            }
                        }
                    }
                }
            },
        },
    )

    perf_target_set = get_regression_perf_target("DemoModel", DeviceTypes.N150)

    assert perf_target_set is not None
    assert perf_target_set.summary_perf_target.is_summary is True
    assert perf_target_set.summary_perf_target.ttft_ms == pytest.approx(45.0)
    assert perf_target_set.summary_perf_target.tput_user == pytest.approx(11.0)
    assert perf_target_set.summary_perf_target.tolerance == pytest.approx(0.05)


def test_get_named_perf_reference_merges_regression_targets(monkeypatch):
    monkeypatch.setattr(
        "workflows.perf_targets.release_performance_reference",
        {
            "schema_version": 1,
            "models": {
                "Qwen3-32B": {
                    "galaxy": {
                        "demo_impl": {
                            "vLLM": {
                                "perf_target_results": [
                                    {
                                        "is_summary_data_point": True,
                                        "config": {
                                            "task_type": "text",
                                            "isl": 128,
                                            "osl": 128,
                                            "max_concurrency": 1,
                                        },
                                        "targets": {"tolerance": 0.02},
                                        "measured_metrics": {
                                            "ttft": 28.0,
                                            "tput_user": 18.0,
                                            "tput": 5400.0,
                                        },
                                    }
                                ]
                            }
                        }
                    }
                }
            },
        },
    )

    perf_reference = get_named_perf_reference("Qwen3-32B", DeviceTypes.GALAXY)

    assert perf_reference
    assert "regression" in perf_reference[0].targets
    assert perf_reference[0].targets["regression"].target_name == "regression"
    assert perf_reference[0].targets["regression"].ttft_ms == pytest.approx(28.0)
    assert perf_reference[0].targets["regression"].tput_user == pytest.approx(18.0)
    assert perf_reference[0].targets["regression"].tolerance == pytest.approx(0.02)

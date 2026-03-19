import pytest

from workflows.perf_targets import (
    DeviceTypes,
    PerfTarget,
    get_perf_reference_for_device,
    get_perf_reference_map,
    get_perf_target,
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for video benchmark dispatch guards and T2V/I2V routing."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from report_module.status import TestStatus

from test_module._test_common import SkipTest
from test_module._test_common import video_generation_routing as routing
from test_module.benchmark_tests import video_benchmark_tests as mod


def _ctx(model_name):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model_name, hf_model_repo=model_name),
        device=SimpleNamespace(name="t3k"),
    )


def test_unsupported_video_model_raises_skip():
    # A model with no inference-step profile is a visible, non-blocking SkipTest
    # (which run_media_task maps to SKIP) rather than a KeyError crash.
    with pytest.raises(SkipTest) as exc:
        mod.run_video_benchmark(_ctx("some-unlisted-video-model"))
    assert "not implemented" in str(exc.value)
    assert "some-unlisted-video-model" in str(exc.value)


def test_i2v_model_has_inference_step_profile():
    # I2V benchmarks are implemented: the model must carry a step profile so it
    # is not skipped, mirroring model_performance_reference.json.
    assert "Wan-AI/Wan2.2-I2V-A14B-Diffusers" in mod.VIDEO_INFERENCE_STEPS


def _capture_submit(monkeypatch, job_id: str) -> dict:
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": job_id}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", lambda *a, **k: "/tmp/out.mp4")
    return captured


@pytest.mark.parametrize("device", ["T3K", "GALAXY", "BLACKHOLE_GALAXY"])
def test_wan_i2v_a14b_routes_to_the_generations_api(monkeypatch, device):
    # Wan2.2-I2V-A14B is served by the generations API: its own submit path,
    # the image as a data URI, and 480p, which every device serves.
    captured = _capture_submit(monkeypatch, "job-123")
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-I2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name=device),
    )
    ok, _elapsed, job_id, video_path = mod._generate_video(
        ctx,
        prompt="a volcano erupting",
        num_inference_steps=40,
        image_b64="ZmFrZQ==",
    )

    assert ok is True
    assert job_id == "job-123"
    assert video_path == "/tmp/out.mp4"
    assert captured["url"].endswith("v1/generations/wan-ai/wan2.2-i2v-a14b")
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "negative_prompt": "blurry, low quality, distorted",
        "seed": 42,
        "first_frame": "data:image/jpeg;base64,ZmFrZQ==",
        "size": "480p",
        "aspect_ratio": "16:9",
    }


def test_other_i2v_variants_keep_the_video_api(monkeypatch):
    captured = _capture_submit(monkeypatch, "job-456")
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-I2V-Lightning"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="GALAXY"),
    )
    mod._generate_video(ctx, prompt="x", num_inference_steps=4, image_b64="ZmFrZQ==")
    assert captured["url"].endswith("v1/videos/generations/i2v")
    assert captured["payload"]["image_prompts"][0]["image"] == "ZmFrZQ=="
    assert captured["payload"]["num_inference_steps"] == 4


@pytest.mark.parametrize(
    ("model_name", "status", "download"),
    [
        (
            "Wan2.2-I2V-A14B-Diffusers",
            "v1/jobs/job-1",
            "http://h/v1/jobs/job-1/content/0",
        ),
        (
            "MiniMax-H3-Ref2VA-LightX2V",
            "v1/jobs/job-1",
            "http://h/v1/jobs/job-1/content/0",
        ),
        (
            "Wan2.2-I2V-Lightning",
            "v1/videos/generations/job-1",
            "http://h/v1/videos/generations/job-1/download",
        ),
    ],
)
def test_job_status_and_download_urls(monkeypatch, model_name, status, download):
    # The generations API's video is behind the completed job's output url.
    requested = []

    class _Result:
        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"outputs": [{"url": "v1/jobs/job-1/content/0"}]}

    def _fake_get(url, headers, timeout):
        requested.append(url)
        return _Result()

    monkeypatch.setattr(routing.requests, "get", _fake_get)
    assert routing.get_video_job_endpoint(model_name, "job-1") == status
    assert (
        routing.get_video_download_url("http://h", model_name, "job-1", {}) == download
    )
    if routing.uses_generations_api(model_name):
        assert requested == ["http://h/v1/jobs/job-1"]


def test_t2v_generation_routes_to_base_endpoint(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-t2v"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", lambda *a, **k: "/tmp/out.mp4")

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-T2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="T3K"),
    )
    mod._generate_video(ctx, prompt="a sunset", num_inference_steps=40)

    assert captured["url"].endswith("v1/videos/generations")
    assert "image_prompts" not in captured["payload"]


def test_minimax_h3_fl2va_lightx2v_has_its_four_step_profile():
    assert mod.VIDEO_INFERENCE_STEPS["MiniMaxAI/MiniMax-H3-FL2VA-LightX2V"] == 4


def test_minimax_h3_fl2va_generation_sends_only_fields_h3_accepts(monkeypatch):
    # H3 takes no num_inference_steps or negative_prompt; the shape is chosen with
    # duration and an explicit canvas, and the keyframe goes in as the first frame.
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-h3"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    def _fake_poll(ctx, job_id, headers, **kwargs):
        captured["poll_timeout"] = kwargs.get("timeout")
        return "/tmp/h3.mp4"

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="MiniMax-H3-FL2VA-LightX2V"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )
    ok, _elapsed, job_id, _path = mod._generate_video(
        ctx, prompt="a volcano erupting", num_inference_steps=4, image_b64="ZmFrZQ=="
    )

    assert ok is True
    assert job_id == "job-h3"
    assert captured["url"].endswith(
        "v1/generations/tenstorrent/minimax-h3-fl2va-lightx2v"
    )
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "duration": 5,
        "seed": 0,
        "first_frame": "data:image/jpeg;base64,ZmFrZQ==",
    }
    assert captured["poll_timeout"] > mod.DEFAULT_VIDEO_TIMEOUT_SECONDS


def test_minimax_h3_fl2va_benchmark_loads_the_keyframe(monkeypatch):
    seen = {}

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64, **_):
        seen["image_b64"] = image_b64
        return True, 9.0, "job", "/tmp/h3.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    monkeypatch.setattr(mod, "_load_fixture_image_base64", lambda: "a2V5")
    ctx = _ctx("MiniMaxAI/MiniMax-H3-FL2VA-LightX2V")
    ctx.model_spec.model_name = "MiniMax-H3-FL2VA-LightX2V"
    (status,) = mod._run_video_generation_benchmark(ctx, 1)
    assert seen["image_b64"] == "a2V5"
    assert status.num_inference_steps == 4


def test_minimax_h3_ref2va_lightx2v_has_its_four_step_profile():
    assert mod.VIDEO_INFERENCE_STEPS["MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V"] == 4


def test_minimax_h3_ref2va_generation_sends_one_reference_image(monkeypatch):
    # The image goes in as a reference, and the adapter's own canvas is left to the server
    # (aspect_ratio only).
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-ref"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    def _fake_poll(ctx, job_id, headers, **kwargs):
        captured["poll_timeout"] = kwargs.get("timeout")
        return "/tmp/ref.mp4"

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="MiniMax-H3-Ref2VA-LightX2V"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )
    ok, _elapsed, job_id, _path = mod._generate_video(
        ctx, prompt="a volcano erupting", num_inference_steps=4, image_b64="ZmFrZQ=="
    )

    assert ok is True
    assert job_id == "job-ref"
    assert captured["url"].endswith(
        "v1/generations/tenstorrent/minimax-h3-ref2va-lightx2v"
    )
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "duration": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "images": ["data:image/jpeg;base64,ZmFrZQ=="],
    }
    assert captured["poll_timeout"] > mod.DEFAULT_VIDEO_TIMEOUT_SECONDS


def test_minimax_h3_ref2va_benchmark_loads_the_reference(monkeypatch):
    seen = {}

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64, **_):
        seen["image_b64"] = image_b64
        return True, 9.0, "job", "/tmp/ref.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    monkeypatch.setattr(mod, "_load_fixture_image_base64", lambda: "a2V5")
    ctx = _ctx("MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V")
    ctx.model_spec.model_name = "MiniMax-H3-Ref2VA-LightX2V"
    (status,) = mod._run_video_generation_benchmark(ctx, 1)
    assert seen["image_b64"] == "a2V5"
    assert status.num_inference_steps == 4


@pytest.mark.parametrize(
    ("model_name", "endpoint", "keys"),
    [
        (
            "MiniMax-H3-FL2VA",
            "v1/generations/minimaxai/minimax-h3-fl2va",
            {"prompt", "duration", "seed", "first_frame"},
        ),
        (
            "MiniMax-H3-Ref2VA",
            "v1/generations/minimaxai/minimax-h3-ref2va",
            {"prompt", "duration", "seed", "aspect_ratio", "images"},
        ),
    ],
)
def test_every_minimax_h3_model_routes_to_the_generations_api(
    model_name, endpoint, keys
):
    assert routing.get_video_generation_submit_endpoint(model_name) == endpoint
    payload = routing.build_video_generation_payload(
        prompt="x", num_inference_steps=50, model_name=model_name, image_b64="ZmFrZQ=="
    )
    assert set(payload) == keys


# -- measurement: polling, seed, burst ----------------------------------------


def _poll_ctx():
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-T2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
    )


def test_polling_defaults_to_one_second():
    assert mod.DEFAULT_VIDEO_POLLING_INTERVAL_SECONDS == 1


def test_poll_records_when_the_job_completed_before_the_download(monkeypatch):
    answers = iter(["queued", "in_progress", "completed"])

    class _Status:
        status_code = 200

        @staticmethod
        def json():
            return {"status": next(answers)}

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Status())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    order = []
    monkeypatch.setattr(
        mod, "_download_video", lambda *a: order.append("download") or "/tmp/v.mp4"
    )
    timings = {}
    path = mod._poll_video_completion(_poll_ctx(), "job", {}, timings=timings)
    assert path == "/tmp/v.mp4"
    assert order == ["download"]
    assert timings["completed_at"] <= mod.time.time()


def test_poll_does_not_record_a_completion_for_a_failed_job(monkeypatch):
    class _Status:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "failed"}

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Status())
    timings = {}
    assert mod._poll_video_completion(_poll_ctx(), "job", {}, timings=timings) == ""
    assert timings == {}


def test_generate_video_reports_generation_time_without_the_download(monkeypatch):
    # start, 202 answered, [in progress seen, completed], downloaded
    clock = iter([100.0, 102.0, 180.0, 183.0])
    monkeypatch.setattr(mod.time, "time", lambda: next(clock))

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-1"}

    def _fake_poll(ctx, job_id, headers, timings=None, **kwargs):
        timings["completed_at"] = mod.time.time()
        return "/tmp/v.mp4"

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _Resp())
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)
    timings = {}
    ok, elapsed, _job, _path = mod._generate_video(
        _poll_ctx(), "p", 40, timings=timings
    )
    assert ok is True
    assert timings["generation_seconds"] == 80.0
    assert elapsed == 83.0


def test_a_request_is_split_into_submit_queue_processing_and_download():
    phases = mod._request_phases(
        start_time=100.0,
        submitted_at=100.2,  # the 202 came after 0.2 s
        poll_times={"first_in_progress_at": 103.0, "completed_at": 228.0},
        finished_at=228.5,
    )
    assert phases["submit_s"] == pytest.approx(0.2)
    assert phases["queue_wait_s"] == pytest.approx(2.8)
    assert phases["processing_s"] == pytest.approx(125.0)
    assert phases["download_s"] == pytest.approx(0.5)
    # the four phases add up to the whole request
    assert sum(phases.values()) == pytest.approx(128.5)


def test_without_an_in_progress_poll_the_whole_wait_counts_as_processing():
    phases = mod._request_phases(
        start_time=10.0,
        submitted_at=10.1,
        poll_times={"completed_at": 12.1},  # never seen in progress
        finished_at=12.2,
    )
    assert phases["queue_wait_s"] == 0.0
    assert phases["processing_s"] == pytest.approx(2.0)


def test_poll_notes_the_first_time_the_job_is_in_progress(monkeypatch):
    answers = iter(["queued", "in_progress", "in_progress", "completed"])

    class _Status:
        status_code = 200

        @staticmethod
        def json():
            return {"status": next(answers)}

    clock = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Status())
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod.time, "time", lambda: next(clock))
    monkeypatch.setattr(mod, "_download_video", lambda *a: "/tmp/v.mp4")
    timings = {}
    mod._poll_video_completion(_poll_ctx(), "job", {}, timings=timings)
    # a later in-progress answer does not move the first one
    assert timings["first_in_progress_at"] < timings["completed_at"]
    assert timings["first_in_progress_at"] == 4.0


def test_the_benchmark_reports_the_mean_phases_of_the_requests(monkeypatch):
    statuses = [
        mod.VideoGenerationTestStatus(
            status=True,
            elapsed=130.0,
            num_inference_steps=40,
            phases={
                "submit_s": 0.1,
                "queue_wait_s": 1.0,
                "processing_s": 120.0,
                "download_s": 0.2,
            },
        ),
        mod.VideoGenerationTestStatus(
            status=True,
            elapsed=132.0,
            num_inference_steps=40,
            phases={
                "submit_s": 0.3,
                "queue_wait_s": 3.0,
                "processing_s": 124.0,
                "download_s": 0.4,
            },
        ),
    ]
    monkeypatch.setattr(mod, "require_health", lambda ctx: None)
    monkeypatch.setattr(mod, "get_num_calls", lambda ctx: 2)
    monkeypatch.setattr(mod, "_run_video_generation_benchmark", lambda c, n: statuses)
    monkeypatch.setattr(mod, "_video_target_checks", lambda *a: ({}, "n/a"))
    monkeypatch.setattr(mod, "block_id", lambda ctx: "id")
    monkeypatch.setattr(mod, "scrape_video_metrics", lambda base_url: {})
    monkeypatch.delenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, raising=False)
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ctx.base_url = "http://localhost:8000"
    data = mod.run_video_benchmark(ctx).data["Benchmarks"]
    assert data["phases"]["submit_s"] == pytest.approx(0.2)
    assert data["phases"]["queue_wait_s"] == pytest.approx(2.0)
    assert data["phases"]["processing_s"] == pytest.approx(122.0)
    assert data["phases"]["download_s"] == pytest.approx(0.3)
    assert len(data["phases_per_request"]) == 2


def test_t2v_benchmark_request_has_a_fixed_seed(monkeypatch):
    captured = _capture_submit(monkeypatch, "job-seed")
    mod._generate_video(_poll_ctx(), mod.VIDEO_BENCHMARK_PROMPT, 40)
    assert captured["payload"]["seed"] == mod.VIDEO_BENCHMARK_SEED
    assert captured["payload"]["prompt"] == mod.VIDEO_BENCHMARK_PROMPT


def test_sequential_benchmark_repeats_the_same_request(monkeypatch):
    prompts = []

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64, timings=None):
        prompts.append(prompt)
        timings["generation_seconds"] = 70.0
        return True, 75.0, "job", "/tmp/v.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    statuses = mod._run_video_generation_benchmark(
        _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers"), 3
    )
    assert prompts == [mod.VIDEO_BENCHMARK_PROMPT] * 3
    assert [s.generation_seconds for s in statuses] == [70.0] * 3


def test_burst_submits_all_jobs_and_reports_throughput_and_latency(monkeypatch):
    latencies = iter([40.0, 80.0, 120.0, 160.0])
    lock = threading.Lock()

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64):
        with lock:
            return True, next(latencies), "job", "/tmp/v.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    out = mod._run_video_burst_benchmark(_ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers"), 4)
    assert out["burst_jobs"] == 4 and out["burst_succeeded"] == 4
    assert out["burst_latency_p50"] == 80.0
    assert out["burst_latency_p95"] == 160.0
    assert out["burst_videos_per_minute"] > 0


def test_burst_counts_only_successful_jobs(monkeypatch):
    results = iter(
        [(True, 50.0, "a", "/a"), (False, 1.0, "", ""), (True, 90.0, "c", "/c")]
    )
    lock = threading.Lock()

    def _fake_generate(*args):
        with lock:
            return next(results)

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    out = mod._run_video_burst_benchmark(_ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers"), 3)
    assert out["burst_succeeded"] == 2
    assert out["burst_latency_p50"] == 50.0  # the failed job's 1 s is not counted


def test_burst_is_off_unless_the_environment_asks_for_it(monkeypatch):
    monkeypatch.delenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, raising=False)
    assert mod._burst_jobs_from_env() == 0
    monkeypatch.setenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, "6")
    assert mod._burst_jobs_from_env() == 6
    monkeypatch.setenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, "many")
    assert mod._burst_jobs_from_env() == 0
    monkeypatch.setenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, "-3")
    assert mod._burst_jobs_from_env() == 0


def test_a_failed_burst_job_fails_the_benchmark(monkeypatch):
    status = mod.VideoGenerationTestStatus(
        status=True, elapsed=80.0, num_inference_steps=40, generation_seconds=78.0
    )
    monkeypatch.setattr(mod, "require_health", lambda ctx: None)
    monkeypatch.setattr(mod, "get_num_calls", lambda ctx: 1)
    monkeypatch.setattr(mod, "_run_video_generation_benchmark", lambda c, n: [status])
    monkeypatch.setattr(
        mod,
        "_run_video_burst_benchmark",
        lambda c, n: {"burst_jobs": n, "burst_succeeded": n - 1},
    )
    monkeypatch.setattr(mod, "_video_target_checks", lambda *a: ({}, "n/a"))
    monkeypatch.setattr(mod, "block_id", lambda ctx: "id")
    monkeypatch.setattr(mod, "scrape_video_metrics", lambda base_url: {})
    monkeypatch.setenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, "4")
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ctx.base_url = "http://localhost:8000"
    block = mod.run_video_benchmark(ctx)
    assert block.data["status"] == TestStatus.FAIL.value
    assert block.data["Benchmarks"]["generation_seconds"] == 78.0
    assert block.data["Benchmarks"]["burst_jobs"] == 4


def test_server_side_metrics_are_reported_per_phase(monkeypatch):
    status = mod.VideoGenerationTestStatus(
        status=True, elapsed=80.0, num_inference_steps=40
    )
    monkeypatch.setattr(mod, "require_health", lambda ctx: None)
    monkeypatch.setattr(mod, "get_num_calls", lambda ctx: 1)
    monkeypatch.setattr(mod, "_run_video_generation_benchmark", lambda c, n: [status])
    monkeypatch.setattr(
        mod,
        "_run_video_burst_benchmark",
        lambda c, n: {"burst_jobs": n, "burst_succeeded": n},
    )
    monkeypatch.setattr(mod, "_video_target_checks", lambda *a: ({}, "n/a"))
    monkeypatch.setattr(mod, "block_id", lambda ctx: "id")
    scrapes = iter(["start", "after sequential", "after burst"])
    monkeypatch.setattr(mod, "scrape_video_metrics", lambda base_url: next(scrapes))
    monkeypatch.setattr(
        mod,
        "server_side_video_metrics",
        lambda before, after: {"phase": f"{before} -> {after}"},
    )
    monkeypatch.setenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, "2")
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ctx.base_url = "http://localhost:8000"
    data = mod.run_video_benchmark(ctx).data["Benchmarks"]
    assert data["server_side"] == {"phase": "start -> after sequential"}
    assert data["burst_server_side"] == {"phase": "after sequential -> after burst"}


def test_a_server_without_metrics_adds_nothing_to_the_report(monkeypatch):
    status = mod.VideoGenerationTestStatus(
        status=True, elapsed=80.0, num_inference_steps=40
    )
    monkeypatch.setattr(mod, "require_health", lambda ctx: None)
    monkeypatch.setattr(mod, "get_num_calls", lambda ctx: 1)
    monkeypatch.setattr(mod, "_run_video_generation_benchmark", lambda c, n: [status])
    monkeypatch.setattr(mod, "_video_target_checks", lambda *a: ({}, "n/a"))
    monkeypatch.setattr(mod, "block_id", lambda ctx: "id")
    monkeypatch.setattr(mod, "scrape_video_metrics", lambda base_url: {})
    monkeypatch.delenv(mod.VIDEO_BENCHMARK_BURST_JOBS_ENV, raising=False)
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ctx.base_url = "http://localhost:8000"
    data = mod.run_video_benchmark(ctx).data["Benchmarks"]
    assert "server_side" not in data and "burst_server_side" not in data

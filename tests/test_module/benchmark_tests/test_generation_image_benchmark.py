# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The generations API image benchmark: per-configuration targets, grading and one real job."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from test_module._test_common.image_generation_api import MODELS
from test_module._test_common.target_check import get_performance_targets
from test_module.benchmark_tests import generation_image_benchmark as bench

QWEN = "Qwen/Qwen-Image-Edit-2511"
IDEOGRAM = "ideogram-ai/ideogram-4"


@pytest.mark.parametrize(
    ("num_images", "seconds"), [(1, 11.32), (2, 13.40), (3, 14.06)]
)
def test_each_qwen_input_count_has_the_8x_h100_target(num_images, seconds):
    targets = get_performance_targets(
        QWEN, "blackhole_galaxy", (1024, 1024), None, {"num_images": num_images}
    )
    assert targets.ttft_ms == round(seconds * 1000)


@pytest.mark.parametrize(
    ("preset", "seconds"), [("V4_DEFAULT_20", 50.46), ("V4_QUALITY_48", 119.35)]
)
def test_each_ideogram_preset_has_its_target(preset, seconds):
    targets = get_performance_targets(
        IDEOGRAM, "blackhole_galaxy", (2048, 2048), None, {"preset": preset}
    )
    assert targets.ttft_ms == round(seconds * 1000)


def ctx_for(repo: str, base_url: str = "http://x"):
    return SimpleNamespace(
        base_url=base_url,
        model_spec=SimpleNamespace(
            hf_model_repo=repo, cli_args={"device": "blackhole_galaxy"}, env_vars={}
        ),
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )


def test_every_qwen_configuration_is_graded_against_its_own_target(monkeypatch):
    sent = []

    def run_job(base_url, submit_path, request):
        sent.append((submit_path, len(request.get("images", []))))
        # 1 input meets its 11.32 s target; 2 and 3 inputs take 3x theirs.
        n = len(request["images"])
        return True, {1: 10.0, 2: 40.2, 3: 42.2}[n]

    monkeypatch.setattr(bench, "_run_job", run_job)
    monkeypatch.setattr(bench, "get_num_calls", lambda ctx: 2)
    monkeypatch.setattr(bench, "block_id", lambda ctx: "id")
    block = bench.run_generation_image_benchmark(
        ctx_for(QWEN), "tt-qwen-image-edit-2511"
    )
    assert sent == [
        ("/v1/generations/qwen/qwen-image-edit-2511", n) for n in (1, 1, 2, 2, 3, 3)
    ]
    records = {r["config"]: r for r in block.data["records"]}
    assert records["1_input"]["target_check"].name == "PASS"
    assert records["3_inputs"]["target_check"].name == "FAIL"
    assert block.data["target_check"].name == "FAIL"
    # Each configuration's fields are suffixed, side by side in one block.
    fields = next(iter(block.data["target_checks"].values()))
    assert {"ttft_1_input", "ttft_2_inputs", "ttft_3_inputs"} <= set(fields)
    assert block.data["num_successful"] == 6 and "status" not in block.data


def test_a_failed_job_fails_the_benchmark(monkeypatch):
    monkeypatch.setattr(bench, "_run_job", lambda *a: (False, 1.0))
    monkeypatch.setattr(bench, "get_num_calls", lambda ctx: 1)
    monkeypatch.setattr(bench, "block_id", lambda ctx: "id")
    block = bench.run_generation_image_benchmark(ctx_for(IDEOGRAM), "tt-ideogram-4")
    assert block.data["num_successful"] == 0 and block.data["status"] == "fail"
    assert [r["config"] for r in block.data["records"]] == [
        "V4_DEFAULT_20",
        "V4_QUALITY_48",
    ]


def test_a_job_is_submitted_polled_to_completion_and_deleted(monkeypatch):
    web = pytest.importorskip("aiohttp.web")
    test_utils = pytest.importorskip("aiohttp.test_utils")
    pytest.importorskip("PIL")
    from tests.test_module.fake_image_generation_api import FakeImageServer

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(bench, "POLL_INTERVAL_SECONDS", 0)
    server = FakeImageServer(MODELS["ideogram-ai/ideogram-4"])
    config = bench.BENCHMARKS["tt-ideogram-4"][0]
    request = {**config.request, "seed": 1, "format": "png"}

    async def body():
        app = web.Application()
        app.add_routes(server.routes())
        async with test_utils.TestServer(app) as test_server:
            url = str(test_server.make_url("")).rstrip("/")
            return await asyncio.to_thread(
                bench._run_job, url, "/v1/generations/ideogram-ai/ideogram-4", request
            )

    ok, seconds = asyncio.run(body())
    assert ok and seconds >= 0
    assert server.requests[0]["rendering_speed"] == "default" and not server.jobs


@pytest.mark.parametrize(
    ("repo", "side", "num_images", "seconds"),
    [
        ("black-forest-labs/FLUX.2-klein-4B", 1024, 0, 0.35),
        ("black-forest-labs/FLUX.2-klein-4B", 2048, 1, 2.19),
        ("black-forest-labs/FLUX.2-klein-4B", 2048, 3, 3.48),
        ("black-forest-labs/flux-3-image-klein", 1024, 0, 1.23),
        ("black-forest-labs/flux-3-image-klein", 2048, 1, 2.24),
        ("black-forest-labs/flux-3-image-klein", 4096, 10, 23.30),
    ],
)
def test_klein_and_flux3_have_a_target_per_size_and_input_count(
    repo, side, num_images, seconds
):
    targets = get_performance_targets(
        repo, "blackhole_galaxy", (side, side), None, {"num_images": num_images}
    )
    assert targets.ttft_ms == round(seconds * 1000)


def test_klein_runs_its_release_shapes_with_no_input_and_one():
    configs = bench.BENCHMARKS["tt-flux.2-klein-4b"]
    assert [(c.shape, c.num_images, c.request["size"]) for c in configs] == [
        ((1024, 1024), 0, "1K"),
        ((1024, 1024), 1, "1K"),
        ((2048, 2048), 0, "2K"),
        ((2048, 2048), 1, "2K"),
    ]
    assert all(c.match == {"num_images": c.num_images} for c in configs)

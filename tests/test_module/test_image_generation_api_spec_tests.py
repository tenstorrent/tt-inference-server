# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Image model spec tests (create contract, lifecycle, sweep) against an in-process server."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module._test_common import image_generation_api as api
from test_module.load_param_tests import image_generation_api_contract_test as contract
from test_module.load_param_tests import image_generation_api_param_test as sweep
from test_module.load_param_tests.image_generation_api_test import (
    ImageGenerationAPITest,
    base_request,
    image_problems,
)
from tests.test_module.fake_image_generation_api import (
    FakeImageServer,
    draw,
    validation_errors,
)

web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")
pytest.importorskip("PIL")

MODEL_IDS = sorted(api.MODELS)


@pytest.fixture(params=MODEL_IDS)
def model(request) -> api.ImageModel:
    return api.MODELS[request.param]


def test_each_model_is_found_by_its_api_id_and_repo():
    assert (
        api.model_for("Qwen/Qwen-Image-Edit-2511").model_id
        == "qwen/qwen-image-edit-2511"
    )
    assert (
        api.model_for("black-forest-labs/FLUX.2-klein-4B").model_id
        == "black-forest-labs/flux.2-klein-4b"
    )
    assert (
        api.model_for("ideogram-ai/ideogram-4-fp8").model_id == "ideogram-ai/ideogram-4"
    )
    with pytest.raises(KeyError):
        api.model_for("stabilityai/stable-diffusion-xl-base-1.0")


def test_sizes_follow_each_models_table():
    flux2, flux3 = (
        api.MODELS["black-forest-labs/flux.2-klein-4b"],
        api.MODELS["black-forest-labs/flux-3-image-klein"],
    )
    assert flux2.size_tiers == ("1K", "2K") and flux3.size_tiers == ("1K", "2K", "4K")
    assert flux3.sizes[("16:9", "4K")] == (4096, 2304)
    assert api.MODELS["ideogram-ai/ideogram-4"].sizes[("16:9", "1K")] == (1280, 720)
    qwen = api.MODELS["qwen/qwen-image-edit-2511"]
    assert qwen.size_tiers == () and qwen.sizes[("3:4", None)] == (896, 1184)
    for model in api.MODELS.values():
        assert {r for r, _ in model.sizes} == set(api.ASPECT_RATIOS)
        assert all(w % 16 == 0 and h % 16 == 0 for w, h in model.sizes.values())


@pytest.mark.parametrize(
    "model_id, request_, images, size",
    [
        ("black-forest-labs/flux.2-klein-4b", {}, [], (1024, 1024)),
        ("black-forest-labs/flux.2-klein-4b", {}, [(608, 416)], (1248, 832)),
        ("black-forest-labs/flux-3-image-klein", {}, [], (2048, 2048)),
        (
            "black-forest-labs/flux-3-image-klein",
            {"size": "1K"},
            [(400, 700), (900, 300)],
            (768, 1376),
        ),
        ("qwen/qwen-image-edit-2511", {}, [(400, 700), (900, 300)], (1376, 768)),
        ("ideogram-ai/ideogram-4", {}, [], (2048, 2048)),
        (
            "ideogram-ai/ideogram-4",
            {"aspect_ratio": "3:2", "size": "1K"},
            [],
            (1248, 832),
        ),
    ],
)
def test_defaults_follow_the_spec(model_id, request_, images, size):
    # FLUX: the first image sets the ratio; Qwen: the last; Ideogram: 1:1.
    assert api.requested_size(api.MODELS[model_id], request_, images) == size


def test_base_requests_and_contract_payloads_are_valid(model):
    assert not validation_errors(model, base_request(model))
    assert not validation_errors(model, contract.valid_payload(model))
    assert all(not validation_errors(model, r) for _, r in sweep.variants(model))


def test_case_and_variant_names_are_unique(model):
    names = [case.name for case in contract.cases(model)]
    assert len(names) == len(set(names))
    variants = [name for name, _ in sweep.variants(model)]
    assert len(variants) == len(set(variants))


def test_every_shape_is_swept_unless_a_tier_is_left_out():
    flux3 = api.MODELS["black-forest-labs/flux-3-image-klein"]
    names = [n for n, _ in sweep.variants(flux3) if n.startswith("shape_")]
    assert len(names) == 21
    names = [
        n for n, _ in sweep.variants(flux3, ("1K", "2K")) if n.startswith("shape_")
    ]
    assert len(names) == 14 and not any(n.endswith("_4K") for n in names)


async def _with_server(routes, body):
    # The over-limit image case sends ~45 MB; aiohttp's default is 1 MiB.
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.add_routes(routes)
    async with test_utils.TestServer(app) as server:
        return await body(str(server.make_url("")).rstrip("/"))


def _run_contract(server: FakeImageServer) -> dict:
    async def body(url):
        return await contract.run_create_contract(
            base_url=url, api_key="your-secret-key", model=server.model
        )

    return asyncio.run(_with_server(server.routes(), body))


def test_contract_passes_against_a_server_that_follows_the_spec(model, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    result = _run_contract(FakeImageServer(model))
    failed = [r for r in result["detailed_test_results"] if not r["passed"]]
    assert result["success"], failed


def test_contract_fails_and_cancels_jobs_when_the_server_accepts_everything(
    monkeypatch,
):
    monkeypatch.delenv("API_KEY", raising=False)
    server = FakeImageServer(
        api.MODELS["ideogram-ai/ideogram-4"], accept_everything=True
    )
    result = _run_contract(server)
    failed = {
        r["check"]: r["message"]
        for r in result["detailed_test_results"]
        if not r["passed"]
    }
    assert "caption_photo_and_art_style" in failed and "count_above_1" in failed
    assert "the server created job" in failed["prompt_is_not_a_field"]
    # Every job a refused request created was cancelled.
    assert all(job["status"] == "cancelled" for job in server.jobs.values())


def _run(test_class, server: FakeImageServer, targets: dict | None = None) -> dict:
    async def body(url):
        ctx = SimpleNamespace(
            service_port=0,
            base_url=url,
            model_spec=SimpleNamespace(
                hf_model_repo=server.model.hf_repos[0], env_vars={}
            ),
            device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
            output_path=None,
        )
        test = test_class(
            _TestConfig.create_default(),
            {"poll_interval": 0, "poll_timeout": 30, **(targets or {})},
            ctx=ctx,
        )
        return await test._run_specific_test_async()

    return asyncio.run(_with_server(server.routes(), body))


def test_lifecycle_passes_for_a_good_job(model, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    server = FakeImageServer(model)
    result = _run(ImageGenerationAPITest, server)
    assert result["success"], result["problems"]
    assert result["model"] == model.model_id
    # The job, the queued one and the running one were all cancelled or deleted.
    assert not server.jobs
    assert {r["seed"] for r in server.requests} == {42, 43, 44}


def test_lifecycle_catches_an_image_at_the_wrong_size(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    model = api.MODELS["black-forest-labs/flux.2-klein-4b"]
    result = _run(ImageGenerationAPITest, FakeImageServer(model, wrong_size=True))
    assert "output: width is 1040, expected 1024" in result["problems"]
    assert "image: image is 1040x1024, expected 1024x1024" in result["problems"]


def test_lifecycle_catches_a_prompt_the_model_should_not_rewrite(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    model = api.MODELS["black-forest-labs/flux.2-klein-4b"]
    result = _run(
        ImageGenerationAPITest, FakeImageServer(model, output={"prompt": "rewritten"})
    )
    assert any("prompt is 'rewritten'" in p for p in result["problems"])


def test_lifecycle_wants_ideograms_caption_as_compact_json(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    model = api.MODELS["ideogram-ai/ideogram-4"]
    result = _run(ImageGenerationAPITest, FakeImageServer(model, json_indent=2))
    assert (
        "output: prompt is the structured_prompt, but not as compact JSON"
        in result["problems"]
    )


def test_sweep_passes_for_a_server_that_follows_the_spec(model, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    server = FakeImageServer(model)
    result = _run(sweep.ImageGenerationAPIParamTest, server)
    failed = [r for r in result["detailed_test_results"] if not r["passed"]]
    assert result["success"], failed
    checks = {r["check"] for r in result["detailed_test_results"]}
    assert {
        "same_seed_pixels",
        "other_seed_pixels",
        "default_shape",
        "format_webp",
    } <= checks
    if model.takes_images:
        assert {"edit_one_image", "edit_most_images", "edit_near_limit_image"} <= checks
    # Every job it made was deleted.
    assert not server.jobs


def test_sweep_catches_a_server_that_ignores_the_seed(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    model = api.MODELS["ideogram-ai/ideogram-4"]
    result = _run(
        sweep.ImageGenerationAPIParamTest, FakeImageServer(model, ignore_seed=True)
    )
    failed = {
        r["check"]: r["problems"]
        for r in result["detailed_test_results"]
        if not r["passed"]
    }
    assert set(failed) == {"other_seed_pixels"}


def test_a_blank_image_is_caught(tmp_path):
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(path)
    assert image_problems(str(path), (64, 64)) == ["image is blank (pixel spread 0.00)"]
    good = tmp_path / "good.webp"
    good.write_bytes(draw(64, 32, 1, "webp"))
    assert image_problems(str(good), (64, 32), "webp") == []
    assert image_problems(str(good), (64, 32), "png") == ["image is WEBP, expected PNG"]


def test_output_check_reads_the_spec():
    model = api.MODELS["ideogram-ai/ideogram-4"]
    request = base_request(model)
    good = {
        "id": "j",
        "outputs": [
            {
                "type": "image",
                "url": "/v1/jobs/j/content/0",
                "content_type": "image/png",
                "seed": 42,
                "prompt": json.dumps(
                    request["structured_prompt"], separators=(",", ":")
                ),
                "width": 1024,
                "height": 1024,
            }
        ],
    }
    assert api.image_output_problems(model, good, request) == []
    # Key order does not matter, and ": " inside a value is not a separator.
    caption = dict(reversed(list(request["structured_prompt"].items())))
    caption["high_level_description"] = "Note: a mug"
    request = {**request, "structured_prompt": caption}
    good["outputs"][0]["prompt"] = json.dumps(caption, separators=(",", ":"))
    assert api.image_output_problems(model, good, request) == []

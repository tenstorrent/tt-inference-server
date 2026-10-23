# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Qwen-Image-Edit spec tests (contract, single edit, sweep) against fake servers."""

from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module._test_common.image_output_check import (
    check_image_output,
    mean_image_difference,
    save_image,
)
from test_module.load_param_tests import qwen_image_edit_contract_test as contract
from test_module.load_param_tests import qwen_image_edit_param_test as sweep
from test_module.load_param_tests import qwen_image_edit_test as single

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

MEDIA_SERVER = Path(__file__).resolve().parents[2] / "tt-media-server"
API_KEY = "your-secret-key"


def _encode(image, fmt="JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _picture(offset=0, size=(1024, 1024)) -> str:
    """A smooth RGB pattern; different offsets are clearly different images."""
    w, h = size
    x = (np.arange(w) * 255 // max(w - 1, 1) + offset * 40) % 256
    y = np.arange(h) * 255 // max(h - 1, 1)
    pixels = np.zeros((h, w, 3), np.uint8)
    pixels[..., 0] = x[None, :]
    pixels[..., 1] = y[:, None]
    pixels[..., 2] = (offset * 70) % 256
    return _encode(Image.fromarray(pixels))


# --- the copied server limits ------------------------------------------------


def _source(path: str) -> str:
    return (MEDIA_SERVER / path).read_text()


def test_contract_limits_match_the_server():
    policy = _source("tt_model_runners/qwen_image_edit_policy.py")
    sides = re.search(r"^QWEN_IMAGE_EDIT_SIDES = (.+)$", policy, re.M).group(1)
    pixels = re.search(
        r"^QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS = (.+)$", policy, re.M
    ).group(1)
    assert contract.QWEN_IMAGE_EDIT_SIDES == ast.literal_eval(sides)
    assert contract.QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS == eval(
        pixels, {"__builtins__": {}}
    )
    request = _source("domain/image_generate_request.py")
    minimum = re.search(r"^_DEFAULT_MIN_INFERENCE_STEPS = (\d+)", request, re.M).group(
        1
    )
    maximum = re.search(r"num_inference_steps: .*le=(\d+)", request).group(1)
    assert contract.MIN_INFERENCE_STEPS == sweep.MIN_INFERENCE_STEPS == int(minimum)
    assert contract.MAX_INFERENCE_STEPS == int(maximum)


def test_sweep_large_input_is_at_the_pixel_limit():
    image = Image.open(io.BytesIO(base64.b64decode(sweep.large_input_image())))
    assert image.width * image.height == contract.QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS


def test_over_limit_case_is_one_row_over():
    [case] = [c for c in contract.cases() if c.name == "image_over_pixel_limit"]
    image = Image.open(io.BytesIO(base64.b64decode(case.payload["image"])))
    assert (
        image.width * image.height == contract.QWEN_IMAGE_EDIT_MAX_INPUT_PIXELS + 4096
    )


def test_case_names_are_unique():
    names = [c.name for c in contract.cases()]
    assert len(names) == len(set(names))


# --- the image check ---------------------------------------------------------


def test_image_check_accepts_a_real_image_and_data_url():
    image = _picture()
    assert check_image_output(image, (1024, 1024)) is None
    assert check_image_output("data:image/jpeg;base64," + image, (1024, 1024)) is None


@pytest.mark.parametrize(
    "image,expected",
    [
        (lambda: _encode(Image.new("RGB", (1024, 1024))), "black"),
        (lambda: _encode(Image.new("RGB", (1024, 1024), (90, 90, 90))), "flat"),
        (lambda: _picture(size=(512, 512)), "512x512, expected 1024x1024"),
        (lambda: "bm90IGFuIGltYWdl", "could not be decoded"),
    ],
)
def test_image_check_reports_problems(image, expected):
    assert expected in check_image_output(image(), (1024, 1024))


def test_save_image_handles_data_urls(tmp_path):
    image = _picture(size=(32, 32))
    saved = save_image("data:image/jpeg;base64," + image, tmp_path, "x", "a.jpg")
    assert saved.read_bytes() == base64.b64decode(image)


# --- fake servers ------------------------------------------------------------


def _with_server(handler, body):
    async def run():
        app = web.Application(client_max_size=128 * 1024 * 1024)
        app.add_routes([web.post(contract.EDITS_PATH, handler)])
        async with test_utils.TestServer(app) as server:
            return await body(str(server.make_url("")).rstrip("/"))

    return asyncio.run(run())


def _auth_error(request):
    auth = request.headers.get("Authorization")
    if auth is None:
        return web.json_response({"detail": "Not authenticated"}, status=403)
    if auth != f"Bearer {API_KEY}":
        return web.json_response({"detail": "Invalid or missing API Key"}, status=401)
    return None


def _contract_handler(*, accept_everything=False):
    async def handler(request):
        if (error := _auth_error(request)) is not None:
            return error
        body = await request.json()
        if accept_everything or body == contract.valid_payload():
            return web.json_response({"images": [_picture()]})
        return web.json_response(
            {"detail": [{"msg": "invalid", "input": body}]}, status=422
        )

    return handler


def _run_contract(**kwargs):
    return _with_server(
        _contract_handler(**kwargs),
        lambda url: contract.run_contract(base_url=url, api_key=API_KEY),
    )


def test_contract_passes_against_a_conforming_server():
    result = _run_contract()
    assert result["success"], [
        r for r in result["detailed_test_results"] if not r["passed"]
    ]


def test_contract_reports_requests_that_ran_an_edit():
    checks = {
        r["check"]: r
        for r in _run_contract(accept_everything=True)["detailed_test_results"]
    }
    assert not checks["seed_past_int63"]["passed"]
    assert "ran the edit" in checks["seed_past_int63"]["message"]
    assert checks["invalid_bearer_authentication"]["passed"]


def test_contract_caps_reported_responses():
    result = _run_contract()
    [over] = [
        r
        for r in result["detailed_test_results"]
        if r["check"] == "image_over_pixel_limit"
    ]
    assert len(str(over["response"])) <= contract.MAX_REPORTED_RESPONSE_CHARS + 100


def _ctx(url, output_path=None):
    return SimpleNamespace(service_port=0, base_url=url, output_path=output_path)


def _run_single(reply, output_path=None):
    async def handler(request):
        body = await request.json()
        return web.json_response({"images": reply(body)})

    async def body(url):
        test = single.QwenImageEditTest(
            _TestConfig.create_default(), {}, ctx=_ctx(url, output_path)
        )
        return await test._run_specific_test_async()

    return _with_server(handler, body)


def test_single_edit_passes_on_an_edited_image():
    result = _run_single(lambda body: [_picture(3)])
    assert result["success"], result["problems"]
    assert result["difference_from_input"] > single.EDITED_THRESHOLD


def test_single_edit_fails_when_the_input_comes_back_unchanged(tmp_path):
    unchanged = _encode(
        Image.open(io.BytesIO(base64.b64decode(single.load_input_image())))
        .convert("RGB")
        .resize((1024, 1024))
    )
    result = _run_single(lambda body: [unchanged], output_path=tmp_path)
    assert any("changed nothing" in p for p in result["problems"])
    assert (tmp_path / "images/failed/spec_tests/qwen_image_edit.jpg").exists()


@pytest.mark.parametrize(
    "reply,expected",
    [
        (lambda body: [_encode(Image.new("RGB", (1024, 1024)))], "black"),
        (lambda body: [_picture(1), _picture(2)], "expected 1 image, got 2"),
        (lambda body: [_picture(1, size=(768, 768))], "expected 1024x1024"),
    ],
)
def test_single_edit_fails_on_a_bad_image(reply, expected):
    result = _run_single(reply)
    assert not result["success"]
    assert any(expected in p for p in result["problems"])


# --- the sweep ---------------------------------------------------------------


def _distinct_server(body):
    """Returns an image determined by what the request asked for."""
    offset = body.get("seed", 0) * 3 + (5 if body["prompt"] == sweep.ALT_PROMPT else 0)
    offset += 0 if body.get("num_inference_steps") == 20 else 7
    count = body.get("number_of_images", 1)
    images = [_picture(offset + i) for i in range(count)]
    if body.get("image_return_format") == "PNG":
        images = [
            _encode(Image.open(io.BytesIO(base64.b64decode(i))).convert("RGB"), "PNG")
            for i in images
        ]
    return images


def _run_sweep(reply, output_path=None, delay_per_step=0.0):
    async def handler(request):
        body = await request.json()
        await asyncio.sleep(body.get("num_inference_steps", 20) * delay_per_step)
        return web.json_response({"images": reply(body)})

    async def body(url):
        test = sweep.QwenImageEditParamTest(
            _TestConfig.create_default(), {}, ctx=_ctx(url, output_path)
        )
        return await test._run_specific_test_async()

    return _with_server(handler, body)


def test_sweep_passes_when_every_parameter_changes_the_image(tmp_path):
    result = _run_sweep(_distinct_server, tmp_path, delay_per_step=0.01)
    failed = [c for c in result["checks"] if not c["passed"]]
    assert result["success"] and not failed, failed
    assert not (tmp_path / "images").exists()


def test_sweep_fails_when_the_server_ignores_the_parameters(tmp_path):
    result = _run_sweep(
        lambda body: [_picture()] * body.get("number_of_images", 1), tmp_path
    )
    assert not result["success"]
    failed = {c["check"] for c in result["checks"] if c["gated"] and not c["passed"]}
    assert {
        "seed_1_differs_from_default",
        "different_prompt_differs_from_default",
        "two_images_differ",
        "png_format_is_png",
    } <= failed
    saved = {
        p.name
        for p in (tmp_path / "images/failed/spec_tests/qwen_image_edit_param").iterdir()
    }
    assert {"default_0.jpg", "seed_1_0.jpg"} <= saved
    assert "duplicate_default_0.jpg" not in saved


def test_variants_change_one_thing_each():
    variants = sweep.build_variants("img", 20, "large")
    base = variants["default"]
    assert variants["fewer_steps"]["num_inference_steps"] == sweep.MIN_INFERENCE_STEPS
    for name, payload in variants.items():
        changed = {
            k for k in base.keys() | payload.keys() if base.get(k) != payload.get(k)
        }
        assert len(changed) <= 1, (name, changed)


def test_image_difference_is_zero_for_the_same_image():
    image = _picture()
    assert mean_image_difference(image, image) == 0.0


def test_suite_lists_contract_then_single_then_sweep():
    suites = json.loads(
        (
            Path(__file__).resolve().parents[2] / "test_module/test_suites/image.json"
        ).read_text()
    )
    [qwen] = [m for m in suites["test_matrices"] if m["models"] == ["qwen_image_edit"]]
    assert [c["template"] for c in qwen["test_cases"]] == [
        "QwenImageEditContractTest",
        "QwenImageEditTest",
        "QwenImageEditParamTest",
    ]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the instruction-edit (Qwen-Image-Edit) image eval and benchmark wiring."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest

from test_module.benchmark_tests import image_benchmark_tests as bench
from test_module.eval_tests import image_eval_tests as evals

_RUNNER = "tt-qwen-image-edit"


def _ctx(tmp_path, output_path=None):
    (tmp_path / "image_client_inpainting_payload").write_text(
        json.dumps({"inpaint_image": "aW1n", "inpaint_mask": "bWFzaw=="})
    )
    return SimpleNamespace(
        base_url="http://localhost:8000",
        test_payloads_path=str(tmp_path),
        output_path=output_path,
    )


def _image(shade=0, size=(1024, 1024)):
    """A valid edited image: a gradient, distinct per ``shade``."""
    np = pytest.importorskip("numpy")
    image_module = pytest.importorskip("PIL.Image")
    pixels = np.zeros((size[1], size[0], 3), np.uint8)
    pixels[..., 0] = np.linspace(0, 255, size[0], dtype=np.uint8)[None, :]
    pixels[..., 2] = shade
    buffer = io.BytesIO()
    image_module.fromarray(pixels).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# Edit-only runners: the server registers /v1/images/edits and not /generations, so a
# runner missing from either table falls back to text-to-image and every request 404s.
@pytest.mark.parametrize("runner", [_RUNNER, "tt-qwen-image-edit-2511"])
def test_runner_is_dispatched_to_edit_eval_and_benchmark(runner):
    assert evals.IMAGE_EVAL_DISPATCH[runner] is evals._run_instruction_edit_eval
    assert (
        bench.IMAGE_BENCHMARK_DISPATCH[runner] is bench._run_qwen_image_edit_benchmark
    )


def test_eval_config_exists_for_qwen_image_edit():
    from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

    assert "Qwen/Qwen-Image-Edit" in ALL_EVAL_CONFIGS
    assert "Qwen/Qwen-Image-Edit-2511" in ALL_EVAL_CONFIGS


def test_edit_eval_sends_each_instruction_and_scores_against_captions(
    tmp_path, monkeypatch
):
    sent = []

    images = {
        i: _image(10 * n) for n, (i, _) in enumerate(evals.INSTRUCTION_EDIT_CASES)
    }

    async def _fake_edit(ctx, session, instruction, input_image):
        sent.append((instruction, input_image))
        return True, 2.0, images[instruction]

    monkeypatch.setattr(evals, "_generate_instruction_edit_eval_async", _fake_edit)

    status_list, _total = asyncio.run(
        evals._run_instruction_edit_eval(_ctx(tmp_path), _RUNNER)
    )

    instructions = [i for i, _ in evals.INSTRUCTION_EDIT_CASES]
    captions = [c for _, c in evals.INSTRUCTION_EDIT_CASES]
    assert [i for i, _ in sent] == instructions
    assert all(img == "aW1n" for _, img in sent)
    # CLIP is computed against the expected-result caption, not the instruction.
    assert [s.prompt for s in status_list] == captions
    assert [s.base64image for s in status_list] == [images[i] for i in instructions]


def test_edit_eval_raises_when_an_edit_fails(tmp_path, monkeypatch):
    async def _fake_edit(ctx, session, instruction, input_image):
        return instruction != evals.INSTRUCTION_EDIT_CASES[1][0], 1.0, _image()

    monkeypatch.setattr(evals, "_generate_instruction_edit_eval_async", _fake_edit)

    try:
        asyncio.run(evals._run_instruction_edit_eval(_ctx(tmp_path), _RUNNER))
    except RuntimeError as e:
        assert evals.INSTRUCTION_EDIT_CASES[1][0] in str(e)
    else:
        raise AssertionError("a failed edit must fail the eval")


def test_edit_benchmark_posts_to_edits_without_mask(tmp_path, monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"images": [_image()]}

    def _fake_post(url, json, headers, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return _Resp()

    monkeypatch.setattr(bench.requests, "post", _fake_post)

    ok, _elapsed = bench._generate_image_instruction_edit(_ctx(tmp_path), 20)

    assert ok is True
    assert captured["url"].endswith("/v1/images/edits")
    assert captured["payload"]["image"] == "aW1n"
    assert "mask" not in captured["payload"]
    assert captured["payload"]["num_inference_steps"] == 20
    assert captured["timeout"] == bench.QWEN_IMAGE_EDIT_REQUEST_TIMEOUT_S


def test_edit_eval_keeps_every_edit_with_its_instruction(tmp_path, monkeypatch):
    async def _fake_edit(ctx, session, instruction, input_image):
        return True, 1.0, _image()

    monkeypatch.setattr(evals, "_generate_instruction_edit_eval_async", _fake_edit)
    out = tmp_path / "out"
    asyncio.run(evals._run_instruction_edit_eval(_ctx(tmp_path, str(out)), _RUNNER))

    saved = out / "images" / "evals"
    index = json.loads((saved / "index.json").read_text())
    assert [e["instruction"] for e in index] == [
        i for i, _ in evals.INSTRUCTION_EDIT_CASES
    ]
    assert all((saved / e["file"]).exists() and e["problem"] is None for e in index)
    assert (saved / "input.png").read_bytes() == b"img"


def test_edit_eval_fails_on_a_black_image(tmp_path, monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (1024, 1024)).save(buffer, format="JPEG")
    black = base64.b64encode(buffer.getvalue()).decode("ascii")

    async def _fake_edit(ctx, session, instruction, input_image):
        return True, 1.0, black

    monkeypatch.setattr(evals, "_generate_instruction_edit_eval_async", _fake_edit)
    with pytest.raises(RuntimeError, match="cannot calculate"):
        asyncio.run(evals._run_instruction_edit_eval(_ctx(tmp_path), _RUNNER))


def test_edit_benchmark_fails_on_a_wrong_size_image_and_keeps_it(tmp_path, monkeypatch):
    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"images": [_image(size=(512, 512))]}

    monkeypatch.setattr(bench.requests, "post", lambda *a, **k: _Resp())
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="expected 1024x1024"):
        bench._generate_image_instruction_edit(_ctx(tmp_path, str(out)), 40)
    assert list((out / "images" / "failed" / "benchmarks").iterdir())


def test_edit_benchmark_runs_the_model_card_workload():
    assert bench.QWEN_IMAGE_EDIT_INFERENCE_STEPS == 40
    assert bench.QWEN_IMAGE_EDIT_CANVAS == (1024, 1024)

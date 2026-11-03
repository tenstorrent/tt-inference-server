# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image eval: its data, clients, scorer, gate and report section, on fakes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

from report_module.renderers import render_image_compare  # noqa: E402
from report_module.schema import Block  # noqa: E402
from test_module._test_common import image_compare as scorer  # noqa: E402
from test_module._test_common import image_eval_data as data  # noqa: E402
from test_module._test_common.image_clients import (  # noqa: E402
    GenerationAPIClient,
    LegacyEditsClient,
)
from test_module._test_common.image_generation_api import MODELS  # noqa: E402
from test_module.eval_tests import image_compare_eval as eval_  # noqa: E402
from tests.test_module.fake_image_generation_api import FakeImageServer  # noqa: E402

QWEN = MODELS["qwen/qwen-image-edit-2511"]


def scene(width: int = 128, height: int = 128, shift: int = 0) -> "np.ndarray":
    """A textured scene: a gradient with a checkered square, ``shift`` px to the right."""
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack(
        [x * 255 // width, y * 255 // height, (x + y) * 127 // (width + height)], -1
    )
    image[40:80, 30 + shift : 70 + shift] = (
        (x[40:80, 30:70] // 5 + y[40:80, 30:70] // 5) % 2 * 255
    )[..., None]
    return image.astype(np.uint8)


def edited(image, box=(90, 120, 90, 120), color=(250, 20, 20)):
    out = image.copy()
    out[box[0] : box[1], box[2] : box[3]] = color
    return out


def save(path: Path, image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def test_the_subset_follows_the_runs_eval_scope():
    assert data.eval_subset(None) == "all"
    assert data.eval_subset("ci-nightly") == "ci"
    assert data.eval_subset("smoke-test") == "dev"


def test_select_cases_by_subset_and_input_count():
    manifest = [
        {"case_id": "a", "dev": True, "ci": True, "images": ["x"]},
        {"case_id": "b", "dev": False, "ci": True, "images": ["x", "y"]},
        {"case_id": "c", "dev": False, "ci": False},
    ]
    assert [c["case_id"] for c in data.select_cases(manifest, "ci")] == ["a", "b"]
    assert [c["case_id"] for c in data.select_cases(manifest, "all", max_images=1)] == [
        "a",
        "c",
    ]
    with pytest.raises(ValueError):
        data.select_cases(manifest, "nightly")


def test_letterbox_pads_onto_the_square_like_tt():
    image = Image.fromarray(scene(200, 100))
    square = data.letterbox(image)
    assert square.size == (1024, 1024)
    # Filled with the color at (2, 2), above and below the scaled image.
    assert square.getpixel((512, 5)) == image.getpixel((2, 2))


def test_identical_images_compare_as_identical():
    a = scene()
    assert scorer.psnr(a, a) == 100.0
    assert scorer.psnr(a, np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)) < 100


def test_an_edit_tt_makes_where_the_gpu_does_agrees():
    pytest.importorskip("skimage")
    source = scene()
    gpu = edited(source)
    metrics = scorer.edit_metrics(gpu, gpu, source)
    assert metrics["change_correlation"] == 1.0 and metrics["change_iou"] == 1.0
    elsewhere = edited(source, box=(5, 25, 5, 25))
    metrics = scorer.edit_metrics(elsewhere, gpu, source)
    assert metrics["change_iou"] == 0.0 and metrics["change_correlation"] < 0.5


def test_drift_is_how_far_the_unchanged_scene_moved():
    pytest.importorskip("skimage")
    # A textured scene, so a shift is visible everywhere.
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    source = np.asarray(
        Image.fromarray(blocks).resize((256, 256), Image.Resampling.BICUBIC)
    )
    gpu = edited(source, box=(180, 230, 180, 230))
    # TT made the same edit but slid the whole image 8 px to the right.
    tt = edited(np.roll(source, 8, axis=1), box=(180, 230, 180, 230))
    metrics = scorer.edit_metrics(tt, gpu, source)
    assert metrics["drift_px"]["gpu"] == 0.0
    assert 6 <= metrics["drift_px"]["tt"] <= 10


def test_groups_follow_kind_references_and_size():
    assert scorer.group_of({"kind": "edit", "images": ["a"]}) == "edit"
    assert (
        scorer.group_of({"kind": "multi_ref", "images": ["a", "b", "c"]})
        == "multi_ref_3"
    )
    assert (
        scorer.group_of({"task": "t2i", "benchmark": "longtext", "size": "2K"})
        == "longtext_2K"
    )
    assert (
        scorer.group_of({"task": "multi_ref", "images": ["a", "b"], "size": "1K"})
        == "multi_ref_2"
    )


def make_dataset(root: Path) -> list[dict]:
    """Two single-input edits, a 2-reference case, and each one's square GPU image."""
    cases = []
    for i, (kind, refs) in enumerate((("edit", 1), ("edit", 1), ("multi_ref", 2))):
        case_id = f"case{i}"
        images = [f"inputs/{case_id}_{j}.png" for j in range(refs)]
        for j, path in enumerate(images):
            save(root / path, scene(160, 96, shift=j))
        cases.append(
            {
                "case_id": case_id,
                "kind": kind,
                "category": "c",
                "prompt": f"edit {i}",
                "images": images,
                "aspect_ratio": "16:9",
                "width": 1376,
                "height": 768,
                "seed": 10 + i,
                "dev": True,
                "ci": True,
            }
        )
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / "cases" / "manifest.json").write_text(json.dumps(cases))
    return cases


def gpu_square(root: Path, case: dict):
    """The square GPU image for a case: its letterboxed input, edited."""
    with Image.open(root / case["images"][0]) as image:
        square = np.asarray(data.letterbox(image))
    return edited(square, box=(400, 600, 400, 600))


def legacy_server(root: Path, cases: list[dict], fail: set[str] = frozenset()):
    """Today's /v1/images/edits: answers with the GPU's own square image."""
    by_prompt = {c["prompt"]: c for c in cases}
    seen = []

    async def edits(request):
        body = await request.json()
        seen.append(body)
        case = by_prompt[body["prompt"]]
        if case["case_id"] in fail:
            return web.json_response({"detail": "worker died"}, status=500)
        buf = io.BytesIO()
        Image.fromarray(gpu_square(root, case)).save(buf, format="PNG")
        return web.json_response(
            {"images": [base64.b64encode(buf.getvalue()).decode()]}
        )

    return [web.post("/v1/images/edits", edits)], seen


def run_eval(tmp_path, monkeypatch, fail=frozenset(), compare_in_process=True):
    root = tmp_path / "data"
    cases = make_dataset(root)
    for case in cases:
        if len(case["images"]) == 1:
            save(
                root / "refs" / "r1_square" / f"{case['case_id']}.png",
                gpu_square(root, case),
            )
    monkeypatch.setattr(
        eval_, "load_cases", lambda model_id, subset, reference: (root, cases)
    )
    if compare_in_process:

        def compare(out_dir, data_dir, reference):
            result = scorer.compare(out_dir, data_dir / reference, data_dir, dino=False)
            return result, None

        monkeypatch.setattr(eval_, "run_compare", compare)
    accepted = []
    monkeypatch.setattr(
        "workflow_module.accept_blocks",
        lambda blocks, envelope=None: accepted.extend(blocks),
    )
    routes, seen = legacy_server(root, cases, fail)

    async def body():
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes(routes)
        async with test_utils.TestServer(app) as server:
            ctx = SimpleNamespace(
                base_url=str(server.make_url("")).rstrip("/"),
                device=SimpleNamespace(name="GALAXY"),
                output_path=str(tmp_path / "out"),
                runtime_config=SimpleNamespace(limit_samples_mode="ci-nightly"),
                model_spec=SimpleNamespace(hf_model_repo="Qwen/Qwen-Image-Edit-2511"),
                service_port=0,
            )
            # The eval runs its own event loop, as it does under the workflow.
            return await asyncio.to_thread(eval_.run_image_compare_eval, ctx, QWEN)

    monkeypatch.setattr(eval_, "block_id", lambda ctx: "id")
    monkeypatch.setattr("test_module._test_common.sweep_envelope", lambda ctx: None)
    evals = asyncio.run(body())
    return evals, accepted, seen


def test_the_eval_sends_todays_api_the_square_and_compares_with_the_square_gpu_run(
    tmp_path, monkeypatch
):
    pytest.importorskip("skimage")
    evals, [section], seen = run_eval(tmp_path, monkeypatch)
    # One request per single-input case: PNG, the reference's sampling, the square input.
    assert len(seen) == 2
    assert {
        (b["image_return_format"], b["num_inference_steps"], b["guidance_scale"])
        for b in seen
    } == {("PNG", 40, 4.0)}
    with Image.open(io.BytesIO(base64.b64decode(seen[0]["image"]))) as sent:
        assert sent.size == (1024, 1024)
    d = section.data
    assert (d["api"], d["reference"], d["subset"]) == (
        "legacy_edits",
        "refs/r1_square",
        "ci",
    )
    assert d["num_generated"] == 2 and d["not_served"] == ["case2"]
    assert d["groups"]["edit"]["psnr"] == 100.0
    # TT changed its (square) input exactly where the GPU did.
    assert d["groups"]["edit"]["change_iou"] == 1.0
    # Nothing calibrated: reported, not gated.
    assert d["gate"]["verdict"] == "NA"
    assert evals.kind == "evals" and evals.data["accuracy_check"].name == "NA"


def test_a_case_the_deployment_could_not_make_fails_the_eval(tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    evals, [section], _ = run_eval(tmp_path, monkeypatch, fail={"case1"})
    assert [f["case_id"] for f in section.data["generation_failed"]] == ["case1"]
    assert evals.data["accuracy_check"].name == "FAIL"
    rendered = render_image_compare(section, {})
    assert "❌ case1: not made" in rendered and "| edit " in rendered


def test_the_gate_checks_what_the_config_sets():
    result = {
        "cases": {
            "a": {
                "group": "edit",
                "psnr": 30.0,
                "ssim": 0.9,
                "lpips": 0.1,
                "dino": 0.95,
            },
            "b": {
                "group": "edit",
                "psnr": 20.0,
                "ssim": 0.7,
                "lpips": 0.4,
                "dino": 0.85,
            },
        },
        "groups": {
            "edit": {
                "cases": 2,
                "lpips": 0.25,
                "dino": 0.9,
                "worst": {"case_id": "b", "lpips": 0.4},
            }
        },
    }
    model, ref = "qwen/qwen-image-edit-2511", "refs/r1"
    assert eval_.grade(result, model, ref, {"models": {}})["verdict"] == "NA"
    config = {
        "models": {
            model: {
                ref: {
                    "max_image_lpips": 0.3,
                    "min_mean_dino": 0.8,
                    "mean_margin": {"psnr": 1.0},
                    "calibration": {"a": {"psnr": 30.0}, "b": {"psnr": 22.0}},
                }
            }
        }
    }
    gate = eval_.grade(result, model, ref, config)
    by_check = {c["check"]: c for c in gate["checks"]}
    assert (
        by_check["max_image_lpips"]["status"] == "FAIL"
        and by_check["max_image_lpips"]["detail"] == "b"
    )
    assert by_check["min_mean_dino"]["status"] == "PASS"
    assert by_check["mean_psnr"]["status"] == "PASS"  # 25.0 against 26.0 - 1.0
    assert gate["verdict"] == "FAIL"


def test_the_generation_api_client_gets_the_cases_png(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    server = FakeImageServer(QWEN)
    case = {"prompt": "edit", "aspect_ratio": "16:9", "seed": 5}
    buf = io.BytesIO()
    Image.fromarray(scene()).save(buf, format="PNG")

    async def body():
        import aiohttp

        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes(server.routes())
        async with test_utils.TestServer(app) as test_server:
            client = GenerationAPIClient(
                str(test_server.make_url("")), QWEN, poll_interval=0
            )
            async with aiohttp.ClientSession() as session:
                return await client.generate(session, case, [buf.getvalue()])

    image, sent = asyncio.run(body())
    with Image.open(io.BytesIO(image)) as decoded:
        assert (decoded.format, decoded.size) == ("PNG", (1376, 768))
    assert (
        sent["request"]["images"] == "<1 image(s)>"
        and sent["request"]["format"] == "png"
    )
    # The job was deleted once downloaded.
    assert not server.jobs


def test_only_a_model_with_eval_data_is_compared(monkeypatch):
    assert eval_.eval_model("Qwen/Qwen-Image-Edit-2511") is QWEN
    assert eval_.eval_model("stabilityai/stable-diffusion-xl-base-1.0") is None
    monkeypatch.delenv(eval_.API_ENV, raising=False)
    assert isinstance(eval_.make_client(QWEN, "http://x"), LegacyEditsClient)
    monkeypatch.setenv(eval_.API_ENV, "generation")
    assert isinstance(eval_.make_client(QWEN, "http://x"), GenerationAPIClient)
    assert isinstance(
        eval_.make_client(MODELS["black-forest-labs/flux.2-klein-4b"], "http://x"),
        GenerationAPIClient,
    )


def test_the_section_renders_groups_and_notes():
    block = Block(
        kind="image_compare",
        title="t",
        data={
            "heading": "qwen/qwen-image-edit-2511 on GALAXY",
            "api": "legacy_edits",
            "reference": "refs/r1_square",
            "subset": "ci",
            "num_cases": 3,
            "num_generated": 2,
            "not_served": ["case2"],
            "groups": {
                "edit": {
                    "cases": 2,
                    "psnr": 31.2,
                    "ssim": 0.9,
                    "lpips": 0.08,
                    "change_iou": 0.8,
                    "drift_px_tt": 1.5,
                    "drift_px_gpu": 0.5,
                    "worst": {"case_id": "case1", "lpips": 0.12},
                }
            },
            "gate": {"verdict": "NA", "checks": []},
            "dino_error": "OSError: no network",
        },
    )
    rendered = render_image_compare(block, {})
    assert (
        "2 of 3 ci cases made through legacy_edits, against the GPU run refs/r1_square"
        in rendered
    )
    assert "nothing calibrated yet" in rendered
    assert "| edit  | 2" in rendered and "1.5 / 0.5" in rendered
    assert "Not sent (1)" in rendered and "DINO unavailable" in rendered

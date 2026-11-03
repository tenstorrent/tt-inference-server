# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The image eval: its data, clients, scorer, gate and report section, on fakes."""

from __future__ import annotations

import asyncio
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
    request_schema,
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
    # Sampler tiers past the default are their own groups (Ideogram's 12 and 48 steps).
    t2i = {"task": "t2i", "benchmark": "dpg", "size": "1K"}
    assert scorer.group_of({**t2i, "tier": "default"}) == "dpg"
    assert scorer.group_of({**t2i, "tier": "quality"}) == "dpg_quality"
    assert scorer.group_of({**t2i, "size": "2K", "tier": "turbo"}) == "dpg_2K_turbo"


def test_ideogram_is_asked_for_the_cases_structured_prompt_shape_and_tier():
    client = GenerationAPIClient("http://x", MODELS["ideogram-ai/ideogram-4"])
    caption = {"high_level_description": "a sign"}
    case = {
        "structured_prompt": caption,
        "prompt": "a sign",
        "aspect_ratio": "16:9",
        "size": "1K",
        "tier": "turbo",
        "seed": 3,
    }
    request = client.request(case, [])
    assert request["structured_prompt"] == caption and "prompt" not in request
    assert (request["aspect_ratio"], request["size"], request["rendering_speed"]) == (
        "16:9",
        "1K",
        "turbo",
    )
    # Qwen takes no tier.
    assert "rendering_speed" not in GenerationAPIClient("http://x", QWEN).request(
        {"prompt": "p", "aspect_ratio": "1:1", "seed": 1, "tier": "turbo"}, [b"x"]
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


def gpu_image(root: Path, case: dict):
    """The GPU image for a case: its first input at the output size, edited."""
    with Image.open(root / case["images"][0]) as image:
        scaled = np.asarray(image.resize((case["width"], case["height"])))
    return edited(scaled, box=(300, 500, 400, 700))


class DeploymentServer(FakeImageServer):
    """The generation API, answering each case with the GPU's own image, and
    publishing ``openapi`` if given."""

    def __init__(
        self,
        root: Path,
        cases: list[dict],
        fail: set[str] = frozenset(),
        openapi: dict | None = None,
    ):
        super().__init__(QWEN)
        self.by_seed = {c["seed"]: c for c in cases}
        self.root, self.fail, self.openapi = root, fail, openapi

    def routes(self):
        if self.openapi is None:
            return super().routes()
        return [web.get("/openapi.json", self.publish), *super().routes()]

    async def publish(self, request):
        return web.json_response(self.openapi)

    def _finish(self, job: dict, status: str) -> None:
        case = self.by_seed[job["_request"]["seed"]]
        if status == "completed" and case["case_id"] in self.fail:
            job["error"] = {"code": "internal", "message": "worker died"}
            status = "failed"
        super()._finish(job, status)
        if status == "completed":
            buf = io.BytesIO()
            Image.fromarray(gpu_image(self.root, case)).save(buf, format="PNG")
            job["_image"] = buf.getvalue()


def run_eval(
    tmp_path, monkeypatch, fail=frozenset(), compare_in_process=True, openapi=None
):
    root = tmp_path / "data"
    cases = make_dataset(root)
    for case in cases:
        save(root / "refs" / "r1" / f"{case['case_id']}.png", gpu_image(root, case))
    monkeypatch.setattr(
        eval_, "load_cases", lambda model_id, subset, reference: (root, cases)
    )
    monkeypatch.setattr(
        eval_,
        "make_client",
        lambda model, base_url: GenerationAPIClient(base_url, model, poll_interval=0),
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
    monkeypatch.delenv("API_KEY", raising=False)
    server = DeploymentServer(root, cases, fail, openapi)

    async def body():
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.add_routes(server.routes())
        async with test_utils.TestServer(app) as test_server:
            ctx = SimpleNamespace(
                base_url=str(test_server.make_url("")).rstrip("/"),
                device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
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
    return evals, accepted, server.requests


def test_the_eval_sends_each_cases_inputs_and_compares_with_r1(tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    evals, [section], seen = run_eval(tmp_path, monkeypatch)
    # One job per case, the multi-reference case included, as the API takes it.
    assert len(seen) == 3
    assert [len(b["images"]) for b in seen] == [1, 1, 2]
    assert all(b["images"][0].startswith("data:image/png;base64,") for b in seen)
    assert {(b["aspect_ratio"], b["format"]) for b in seen} == {("16:9", "png")}
    assert [b["seed"] for b in seen] == [10, 11, 12]
    d = section.data
    assert (d["api"], d["reference"], d["subset"]) == (
        "generation_api",
        "refs/r1",
        "ci",
    )
    assert d["num_generated"] == 3 and d["not_served"] == {}
    assert d["groups"]["edit"]["psnr"] == 100.0
    # TT changed its input exactly where the GPU did.
    assert d["groups"]["edit"]["change_iou"] == 1.0
    # Nothing calibrated: reported, not gated.
    assert d["gate"]["verdict"] == "NA"
    assert evals.kind == "evals" and evals.data["accuracy_check"].name == "NA"


def published(images: dict, aspect_ratios: list[str]) -> dict:
    """An OpenAPI document as FastAPI and pydantic write one for Qwen's route."""
    return {
        "paths": {
            QWEN.submit_path: {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Request"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Ratio": {"type": "string", "enum": aspect_ratios},
                "Request": {
                    "type": "object",
                    "required": ["prompt", "images"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "images": images,
                        "aspect_ratio": {
                            "anyOf": [
                                {"$ref": "#/components/schemas/Ratio"},
                                {"type": "null"},
                            ]
                        },
                        "format": {"const": "png", "type": "string"},
                        "seed": {"type": "integer"},
                    },
                },
            }
        },
    }


ONE_IMAGE = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1}


def test_a_case_outside_the_published_schema_is_not_served():
    client = GenerationAPIClient("http://x", QWEN)
    openapi = published(ONE_IMAGE, ["1:1", "16:9"])
    schema = request_schema(openapi, QWEN.submit_path)
    case = {"prompt": "p", "images": ["a.png"], "aspect_ratio": "16:9", "seed": 1}
    assert client.unserved(case, (openapi, schema)) == []
    assert client.unserved({**case, "aspect_ratio": "2:3"}, (openapi, schema)) == [
        "aspect_ratio '2:3' (serves 1:1, 16:9)"
    ]
    assert client.unserved({**case, "images": ["a", "b"]}, (openapi, schema)) == [
        "2 image(s) (serves 1 to 1)"
    ]
    assert client.unserved({**case, "images": []}, (openapi, schema)) == [
        "no images, which it requires"
    ]
    # A field the deployment does not take.
    ideogram = GenerationAPIClient("http://x", MODELS["ideogram-ai/ideogram-4"])
    assert "no size field" in ideogram.unserved(
        {"structured_prompt": {}, "aspect_ratio": "1:1", "size": "1K", "seed": 1},
        (
            {},
            {
                "properties": {
                    "structured_prompt": {},
                    "aspect_ratio": {},
                    "seed": {},
                    "format": {},
                }
            },
        ),
    )
    # No published schema: every case is sent.
    assert client.unserved({**case, "aspect_ratio": "2:3"}, None) == []
    assert request_schema({"paths": {}}, QWEN.submit_path) is None


def test_the_eval_leaves_out_what_the_deployment_does_not_serve(tmp_path, monkeypatch):
    pytest.importorskip("skimage")
    evals, [section], seen = run_eval(
        tmp_path, monkeypatch, openapi=published(ONE_IMAGE, ["16:9"])
    )
    # The 2-reference case is not sent, and is listed, not failed.
    assert [len(b["images"]) for b in seen] == [1, 1]
    assert section.data["not_served"] == {"case2": "2 image(s) (serves 1 to 1)"}
    assert section.data["generation_failed"] == []
    assert evals.data["num_cases"] == 2
    assert "Not sent (1), outside what the deployment serves: case2" in (
        render_image_compare(section, {})
    )


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


def test_only_a_model_with_eval_data_is_compared():
    assert eval_.eval_model("Qwen/Qwen-Image-Edit-2511") is QWEN
    assert eval_.eval_model("stabilityai/stable-diffusion-xl-base-1.0") is None


def test_the_section_renders_groups_and_notes():
    block = Block(
        kind="image_compare",
        title="t",
        data={
            "heading": "qwen/qwen-image-edit-2511 on BLACKHOLE_GALAXY",
            "api": "generation_api",
            "reference": "refs/r1",
            "subset": "ci",
            "num_cases": 3,
            "num_generated": 2,
            "not_served": {"case2": "aspect_ratio '2:3' (serves 1:1)"},
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
        "2 of 3 ci cases made through generation_api, against the GPU run refs/r1"
        in rendered
    )
    assert "nothing calibrated yet" in rendered
    assert "| edit  | 2" in rendered and "1.5 / 0.5" in rendered
    assert "Not sent (1)" in rendered and "DINO unavailable" in rendered

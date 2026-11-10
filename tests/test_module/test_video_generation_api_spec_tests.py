# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Spec tests for the generation API's video models with sound (LTX-2.3 distilled,
MiniMax-H3 FL2VA and Ref2VA) against an in-process server."""

from __future__ import annotations

import asyncio
import importlib
import io
import re
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module._test_common import video_generation_api as api
from test_module._test_common.video_generation_routing import (
    build_video_generation_payload,
    get_video_generation_submit_endpoint,
    get_video_job_endpoint,
)
from test_module.benchmark_tests.video_benchmark_tests import VIDEO_INFERENCE_STEPS
from test_module.load_param_tests import (
    video_generation_api_contract_test as contract,
)
from test_module.load_param_tests import video_generation_api_param_test as sweep
from test_module.load_param_tests import video_generation_api_test as lifecycle
from test_module.test_categorization_system.test_filter import (
    TestFilter as SuiteTestFilter,  # aliased so pytest does not collect it
)
from tests.test_module.fake_generation_api import CHOSEN_SEED
from tests.test_module.fake_video_generation_api import FakeVideoJobsServer
from tests.test_module.video_clips import write_sound_clip

web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

MEDIA_SERVER = Path(__file__).resolve().parents[2] / "tt-media-server"
LTX = "Lightricks/LTX-2.3-distilled"
FL2VA = "MiniMaxAI/MiniMax-H3-FL2VA-LightX2V"
REF2VA = "MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V"
# The base MiniMax-H3 models take the same requests at their own 50 steps.
BASE_FL2VA = "MiniMaxAI/MiniMax-H3-FL2VA"
BASE_REF2VA = "MiniMaxAI/MiniMax-H3-Ref2VA"
DEPLOYMENTS = (LTX, BASE_FL2VA, BASE_REF2VA, FL2VA, REF2VA)


@pytest.fixture(autouse=True)
def _download_dir(monkeypatch, tmp_path):
    # /tmp/videos, the default, can belong to another user on a shared host.
    monkeypatch.setenv("TT_VIDEO_OUTPUT_DIR", str(tmp_path / "downloads"))


# --- The models --------------------------------------------------------------------


def test_every_served_route_has_a_model():
    """Each LTX-2.3 and MiniMax-H3 route the server registers is a model here."""
    domain = "".join(
        (MEDIA_SERVER / "domain" / name).read_text()
        for name in ("ltx23_distilled_request.py", "minimax_h3_request.py")
    )
    served = set(re.findall(r'\bid="([^"]+)"', domain))
    assert served == set(api.MODELS)


@pytest.mark.parametrize("repo", DEPLOYMENTS)
def test_a_deployment_is_found_by_its_repo_and_its_id(repo):
    model = api.model_for(repo)
    assert api.model_for(model.model_id) is model
    assert api.model_for(repo.lower()) is model
    assert model.submit_path == f"/v1/generations/{model.model_id}"
    # A deployment answers another model's route with 404, so it must differ.
    assert model.unserved_model_id != model.model_id


def test_another_model_is_not_one_of_these():
    assert not api.is_video_generation_api_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    with pytest.raises(KeyError):
        api.model_for("Wan-AI/Wan2.2-I2V-A14B-Diffusers")


def test_h3_sizes_are_the_request_schemas_table():
    assert {r: api.h3_canvas(*api.ratio_shape(r)) for r in api.H3_RATIOS} == {
        "21:9": (1536, 672),
        "16:9": (1344, 768),
        "4:3": (1024, 768),
        "1:1": (768, 768),
        "3:4": (768, 1024),
        "9:16": (768, 1344),
    }


def test_frame_counts_are_the_ones_the_models_generate():
    # MiniMax-H3: the 24 fps count, up to the next 17n + 5; LTX-2.3: 8k + 1.
    assert [api.h3_num_frames(d) for d in (4, 5, 15)] == [107, 124, 362]
    assert api.ltx_num_frames(6) == 153
    assert api.model_for(LTX).num_frames() == 153
    assert api.model_for(FL2VA).num_frames() == 124


def test_a_keyframe_request_follows_its_frame():
    model = api.model_for(FL2VA)
    with_frame = {"prompt": "p", "first_frame": "data:image/png;base64,AAAA"}
    assert api.requested_shape(model, with_frame, (500, 375)) == (1024, 768)
    assert api.requested_shape(model, {"prompt": "p"}) == (1344, 768)
    assert api.requested_shape(model, {"prompt": "p", "aspect_ratio": "1:1"}) == (
        768,
        768,
    )
    with pytest.raises(ValueError):
        api.requested_shape(model, with_frame)
    # The references don't set the shape; LTX-2.3 has one.
    assert api.requested_shape(
        api.model_for(REF2VA), {"prompt": "p", "images": ["x"]}
    ) == (1344, 768)
    assert api.requested_shape(api.model_for(LTX), {"prompt": "p"}) == api.LTX_SHAPE


def test_solid_png_and_tone_wav_are_what_they_say():
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.open(io.BytesIO(api.solid_png(640, 256)))
    image.load()
    assert (image.format, image.size) == ("PNG", (640, 256))
    with wave.open(io.BytesIO(api.tone_wav(2.5))) as wav:
        assert wav.getnframes() / wav.getframerate() == 2.5
        assert wav.getnchannels() == 1


# --- The create contract -----------------------------------------------------------


@pytest.mark.parametrize("model", api.MODELS.values(), ids=list(api.MODELS))
def test_case_names_are_unique_and_every_refusal_is_covered(model):
    cases = contract.cases(model)
    names = [case.name for case in cases]
    assert len(names) == len(set(names))
    assert {case.expected_status for case in cases} >= {401, 404, 422}
    takes_media = model.kind != "text"
    assert ({400, 413} <= {case.expected_status for case in cases}) == takes_media
    # Only the route case leaves the model's own path.
    assert [case.name for case in cases if case.path != model.submit_path] == [
        "another_models_route"
    ]


@pytest.mark.parametrize("model", api.MODELS.values(), ids=list(api.MODELS))
def test_the_unfetchable_url_can_point_at_an_allowed_host(model):
    url = "https://allowed.example.com/missing.png"
    unfetchable = [
        case for case in contract.cases(model, url) if case.name.startswith("unfetch")
    ]
    assert len(unfetchable) == (0 if model.kind == "text" else 1)
    for case in unfetchable:
        assert url in str(case.payload)


async def _with_server(routes, body):
    """Run ``body(base_url)`` against an aiohttp app serving ``routes``."""
    # The real server has no body limit; the over-limit image case sends ~45 MB.
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.add_routes(routes)
    async with test_utils.TestServer(app) as server:
        return await body(str(server.make_url("")).rstrip("/"))


def test_contract_fails_a_server_that_accepts_everything_and_cancels_its_jobs():
    pytest.importorskip("PIL.Image")
    model = api.model_for(FL2VA)
    server = FakeVideoJobsServer(model, lambda request: "unused.mp4")
    routes = [
        *server.routes(),
        web.post(model.unserved_submit_path, server.submit),
    ]

    async def body(url):
        return await contract.run_create_contract(
            base_url=url, api_key="test-key", model=model
        )

    result = asyncio.run(_with_server(routes, body))
    assert not result["success"]
    rows = result["parameter_conformance_summary"]
    assert [row["test_case"] for row in rows] == [c.name for c in contract.cases(model)]
    assert {row["status"] for row in rows} == {"FAIL"}
    # A refused request must not leave a job holding the device.
    assert {job["status"] for job in server.jobs.values()} == {"cancelled"}


def test_contract_reports_an_unreachable_server():
    result = asyncio.run(
        contract.run_create_contract(
            base_url="http://127.0.0.1:9",
            api_key="test-key",
            model=api.model_for(LTX),
            request_timeout=2,
        )
    )
    assert not result["success"]
    assert {r["actual_status"] for r in result["detailed_test_results"]} == {
        "request_error"
    }


# --- The lifecycle -----------------------------------------------------------------


def _ctx(url: str, repo: str, output_path=None) -> SimpleNamespace:
    return SimpleNamespace(
        service_port=0,
        base_url=url,
        model_spec=SimpleNamespace(hf_model_repo=repo, env_vars={}),
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
        output_path=output_path,
    )


def _run_lifecycle(server, repo, output_path=None, targets=None) -> dict:
    async def body(url):
        test = lifecycle.VideoGenerationAPITest(
            _TestConfig.create_default(),
            {"poll_interval": 0, "poll_timeout": 30, **(targets or {})},
            ctx=_ctx(url, repo, output_path),
        )
        return await test._run_specific_test_async()

    return asyncio.run(_with_server(server.routes(), body))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """``clip(model, request, hue=0, **kwargs)``: the clip a request asks for, made once."""
    pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("imageio.v3")
    pytest.importorskip("PIL.Image")
    root = tmp_path_factory.mktemp("video_api_clips")
    made: dict[tuple, Path] = {}

    def clip(model, request, hue=0, frames=None, audio=True):
        width, height = api.requested_shape(
            model, request, lifecycle.frame_size(request)
        )
        frames = frames or model.num_frames(request.get("duration"))
        key = (width, height, frames, model.fps, hue, audio)
        if key not in made:
            name = "_".join(str(part) for part in key)
            made[key] = write_sound_clip(
                root / f"{name}.mp4", width, height, frames, model.fps, hue, audio
            )
        return made[key]

    return clip


def _status(result: dict) -> dict[str, str]:
    return {
        row["test_case"]: row["status"]
        for row in result["parameter_conformance_summary"]
    }


@pytest.mark.parametrize("repo", DEPLOYMENTS)
def test_lifecycle_passes_for_a_good_job(clips, repo):
    model = api.model_for(repo)
    server = FakeVideoJobsServer(model, lambda request: clips(model, request))
    result = _run_lifecycle(server, repo)
    assert result["success"], result["problems"]
    assert set(_status(result).values()) == {"PASS"}
    assert result["model"] == model.model_id
    # The job, the queued one and the running one were all cancelled or deleted.
    assert not server.jobs
    [first, queued, running] = server.requests
    assert {first["seed"], queued["seed"], running["seed"]} == {42, 43, 44}
    if model.kind == "keyframes":
        assert first["first_frame"].startswith("data:image/jpeg;base64,")
        assert "aspect_ratio" not in first
    elif model.kind == "references":
        assert len(first["images"]) == 1 and first["aspect_ratio"] == "16:9"
    else:
        assert set(first) == {"prompt", "seed"}


def test_lifecycle_takes_its_duration_from_the_targets(clips):
    model = api.model_for(FL2VA)
    server = FakeVideoJobsServer(model, lambda request: clips(model, request))
    result = _run_lifecycle(server, FL2VA, targets={"duration": 4})
    assert result["success"], result["problems"]
    assert {request["duration"] for request in server.requests} == {4}


def test_lifecycle_wants_the_output_to_describe_its_audio_track(clips):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(
        model, lambda request: clips(model, request), output={"audio": None}
    )
    result = _run_lifecycle(server, LTX)
    assert not result["success"]
    assert _status(result)["output"] == "FAIL"
    assert any("audio is None" in p for p in result["problems"])


def test_lifecycle_wants_an_audio_track_in_the_clip(clips, tmp_path):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(
        model, lambda request: clips(model, request, audio=False)
    )
    result = _run_lifecycle(server, LTX, output_path=tmp_path / "out")
    assert not result["success"]
    status = _status(result)
    assert (status["container"], status["clip"]) == ("FAIL", "PASS")
    assert any("0 audio stream(s), expected one" in p for p in result["problems"])
    # The failing clip is kept with the run.
    assert list((tmp_path / "out" / "videos" / "failed" / "spec_tests").glob("*.mp4"))


def test_lifecycle_wants_every_frame_the_output_reports(clips):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(
        model, lambda request: clips(model, request, frames=145)
    )
    result = _run_lifecycle(server, LTX)
    assert not result["success"]
    assert _status(result)["clip"] == "FAIL"
    assert any("145 frames, expected 153" in p for p in result["problems"])


def test_lifecycle_names_the_step_that_failed_so_it_can_be_waived(clips):
    model = api.model_for(REF2VA)
    server = FakeVideoJobsServer(
        model, lambda request: clips(model, request), cancel_running=False
    )
    result = _run_lifecycle(server, REF2VA)
    assert not result["success"]
    failed = [step for step, status in _status(result).items() if status == "FAIL"]
    assert failed == ["cancel_running_job"]


def test_lifecycle_can_leave_the_running_cancel_out(clips):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(
        model, lambda request: clips(model, request), cancel_running=False
    )
    result = _run_lifecycle(server, LTX, targets={"cancel_running": False})
    assert result["success"], result["problems"]
    assert len(server.requests) == 2


def test_lifecycle_fails_when_the_api_is_not_served():
    result = _run_lifecycle(
        SimpleNamespace(routes=lambda: []), LTX, targets={"poll_timeout": 2}
    )
    assert not result["success"]
    assert result["reason"] == "submit_failed"
    assert _status(result)["submit"] == "FAIL"


# --- The parameter sweep -----------------------------------------------------------


def test_ltx_sweeps_its_one_shape():
    model = api.model_for(LTX)
    requests = sweep.variants(model)
    assert list(requests) == [
        "default",
        "other_seed",
        "no_seed",
        "different_prompt",
        "stated_shape",
    ]
    assert requests["stated_shape"] == {
        **requests["default"],
        "aspect_ratio": "16:9",
        "size": "1080p",
        "duration": 6,
        "fps": 25,
    }
    assert "seed" not in requests["no_seed"]


def test_fl2va_sweeps_text_keyframes_ratios_and_durations():
    pytest.importorskip("PIL.Image")
    model = api.model_for(FL2VA)
    requests = sweep.variants(model)
    assert set(requests) == {
        "default",
        "other_seed",
        "no_seed",
        "different_prompt",
        "text_only",
        "last_frame",
        "first_and_last_frame",
        *(f"shape_{r.replace(':', 'x')}_768p" for r in api.H3_RATIOS),
        "duration_4",
        "duration_15",
    }
    assert "first_frame" in requests["default"]
    assert "first_frame" not in requests["last_frame"]
    assert requests["first_and_last_frame"]["aspect_ratio"] == "adaptive"
    # A listed ratio only applies without a frame.
    for name, request in requests.items():
        if name.startswith(("shape_", "duration_")) or name == "text_only":
            assert not {"first_frame", "last_frame"} & set(request), name


def test_ref2va_sweeps_its_references():
    pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    model = api.model_for(REF2VA)
    requests = sweep.variants(model)
    assert len(requests["two_images"]["images"]) == 2
    assert len(set(requests["most_images"]["images"])) == model.max_images == 9
    assert requests["audio_reference"]["audios"][0].startswith("data:audio/wav;")
    assert requests["video_reference"]["videos"][0].startswith("data:video/mp4;")
    assert "images" not in requests["video_reference"]
    # 16:9 is the base request's own ratio, so it has no variant of its own.
    assert "shape_16x9_768p" not in requests
    assert "shape_9x16_768p" in requests


def test_targets_trim_the_sweep():
    pytest.importorskip("PIL.Image")
    requests = sweep.variants(
        api.model_for(FL2VA),
        aspect_ratios=("1:1",),
        durations=(),
        exclude=("last_frame", "text_only"),
    )
    assert set(requests) == {
        "default",
        "other_seed",
        "no_seed",
        "different_prompt",
        "first_and_last_frame",
        "shape_1x1_768p",
    }


def test_a_reference_video_that_cannot_be_encoded_is_a_skipped_variant(monkeypatch):
    def no_encoder():
        raise ImportError("No module named 'imageio_ffmpeg'")

    monkeypatch.setattr(sweep, "reference_clip", no_encoder)
    assert sweep.variants(api.model_for(REF2VA))["video_reference"] is None


def _run_sweep(server, repo, output_path=None, targets=None) -> dict:
    async def body(url):
        test = sweep.VideoGenerationAPIParamTest(
            _TestConfig.create_default(),
            {"poll_interval": 0, "poll_timeout": 30, **(targets or {})},
            ctx=_ctx(url, repo, output_path),
        )
        return await test._run_specific_test_async()

    return asyncio.run(_with_server(server.routes(), body))


def _working(model, clips):
    """The clip each request asked for, as a working deployment would make it:
    another seed or prompt gives another clip, the same request the same one."""

    def pick(request):
        if request["prompt"] == sweep.ALT_PROMPT:
            hue = 240
        else:
            hue = {42: 0, 142: 120, CHOSEN_SEED: 60}.get(
                request.get("seed", CHOSEN_SEED), 180
            )
        return clips(model, request, hue=hue)

    return pick


def test_ltx_sweep_passes_when_every_parameter_takes_effect(clips):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(model, _working(model, clips))
    result = _run_sweep(server, LTX)
    assert result["success"], result["detailed_test_results"]
    status = _status(result)
    assert set(status.values()) == {"PASS"}
    assert set(status) == {
        "default",
        "other_seed",
        "no_seed",
        "different_prompt",
        "stated_shape",
        "other_seed_differs_from_default",
        "different_prompt_differs_from_default",
    }
    # The seedless request is repeated with the seed its output reported.
    assert server.requests[-1]["seed"] == CHOSEN_SEED
    [replay] = [
        r
        for r in result["detailed_test_results"]
        if r["check"] == "reported_seed_reproduces_clip"
    ]
    assert replay["passed"] and not replay["gated"]
    # Every job was deleted once its clip was downloaded.
    assert not server.jobs


def test_h3_sweep_passes_over_keyframes_and_a_listed_ratio(clips):
    model = api.model_for(FL2VA)
    server = FakeVideoJobsServer(model, _working(model, clips))
    result = _run_sweep(
        server, FL2VA, targets={"aspect_ratios": ["1:1"], "durations": [4]}
    )
    assert result["success"], result["detailed_test_results"]
    status = _status(result)
    assert set(status.values()) == {"PASS"}
    assert {"first_and_last_frame", "shape_1x1_768p", "duration_4"} <= set(status)


def test_sweep_fails_when_the_server_ignores_the_seed_and_the_prompt(clips, tmp_path):
    model = api.model_for(LTX)
    server = FakeVideoJobsServer(model, lambda request: clips(model, request))
    result = _run_sweep(server, LTX, output_path=tmp_path / "out")
    assert not result["success"]
    failed = {step for step, status in _status(result).items() if status == "FAIL"}
    assert failed == {
        "other_seed_differs_from_default",
        "different_prompt_differs_from_default",
    }
    kept = tmp_path / "out" / "videos" / "failed" / "spec_tests" / "param_sweep"
    assert {path.name for path in kept.glob("*.mp4")} == {
        "default.mp4",
        "other_seed.mp4",
        "different_prompt.mp4",
    }


def test_a_variant_the_deployment_refuses_fails_by_name(clips):
    """As a first release that serves fewer shapes does: the row is the waiver's name."""
    model = api.model_for(REF2VA)
    working = FakeVideoJobsServer(model, _working(model, clips))

    async def submit(request):
        body = await request.json()
        if body.get("aspect_ratio") == "1:1":
            return web.json_response(
                {
                    "detail": [
                        {"loc": ["body", "aspect_ratio"], "msg": "x", "type": "y"}
                    ]
                },
                status=422,
            )
        return await working.submit(request)

    server = SimpleNamespace(
        routes=lambda: [web.post(model.submit_path, submit), *working.routes()[1:]]
    )
    result = _run_sweep(
        server,
        REF2VA,
        targets={
            "aspect_ratios": ["1:1"],
            "durations": [],
            "exclude": ["most_images", "two_images", "video_reference"],
        },
    )
    assert not result["success"]
    failed = [step for step, status in _status(result).items() if status == "FAIL"]
    assert failed == ["shape_1x1_768p"]


# --- Enrollment, benchmark and eval routing ----------------------------------------


@pytest.mark.parametrize("repo", DEPLOYMENTS)
def test_a_deployment_on_a_bh_galaxy_runs_the_three_spec_tests(repo):
    suites = (
        SuiteTestFilter()
        .filter_by_model(repo)
        .filter_by_device("blackhole_galaxy")
        .get_tests()
    )
    cases = [case for suite in suites for case in suite["test_cases"]]
    # After the prerequisite tests every suite starts with.
    assert [case["name"] for case in cases][-3:] == [
        "VideoGenerationAPIContractTest",
        "VideoGenerationAPITest",
        "VideoGenerationAPIParamTest",
    ]
    for case in cases[-3:]:
        module = importlib.import_module(case["module"])
        assert hasattr(module, case["name"])


def test_ltx_is_benchmarked_and_evaluated_through_the_generations_api():
    assert get_video_generation_submit_endpoint(LTX) == (
        "v1/generations/lightricks/ltx-2.3-22b-distilled"
    )
    # The benchmark routes by the served model name, the eval by the repo.
    assert get_video_generation_submit_endpoint("LTX-2.3-distilled") == (
        get_video_generation_submit_endpoint(LTX)
    )
    assert get_video_job_endpoint(LTX, "j1") == "v1/jobs/j1"
    payload = build_video_generation_payload(
        prompt="a boat", num_inference_steps=11, model_name=LTX
    )
    # Text to video at its one shape: no step count, no image.
    assert payload == {"prompt": "a boat", "seed": 42}
    # The benchmark skips a model it has no step count for.
    assert set(DEPLOYMENTS) <= set(VIDEO_INFERENCE_STEPS)

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""URL hygiene of the MiniMax-H3 create-contract suite: the target is a validated http(s)
origin plus route constants, and only a well-formed job id is ever placed in a URL. The
checks follow the deployment's task: its own route and body, then the routes it refuses."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from test_module.load_param_tests import minimax_h3_create_contract_test as C


@pytest.mark.parametrize(
    ("given", "root"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("https://h3.example.com/api/", "https://h3.example.com/api"),
        (" http://h3.example.com?x=1#f ", "http://h3.example.com"),
    ],
)
def test_service_root_keeps_origin_and_prefix_only(given, root):
    assert C._service_root(given) == root


@pytest.mark.parametrize(
    "bad", ["file:///etc/passwd", "127.0.0.1:8000", "ftp://x", "", "/v1"]
)
def test_service_root_refuses_anything_but_http(bad):
    with pytest.raises(ValueError):
        C._service_root(bad)


class _RecordingClient:
    """Stands in for MiniMaxH3Client: records what the cancel helper asks for."""

    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def cancel_task(self, task_id):
        _RecordingClient.calls.append((self.kwargs["base_url"], task_id))
        return {"id": task_id, "status": "cancelled"}


@pytest.mark.parametrize(
    "task_id",
    ["../../admin", "abc def", "", "x" * 129, "%0d%0aX: 1", "not-a-uuid", None],
)
def test_cancel_refuses_a_job_id_that_is_not_a_uuid(task_id, monkeypatch):
    def no_client(**kwargs):
        raise AssertionError(f"no request expected for {task_id!r}: {kwargs}")

    monkeypatch.setattr(C, "MiniMaxH3Client", no_client)
    assert C._job_uuid(task_id) is None
    result = asyncio.run(
        C._cancel_created_job(
            base_url="http://127.0.0.1:9", api_key="k", task_id=task_id
        )
    )
    assert result is None


def test_cancel_uses_the_canonical_uuid_and_the_service_root(monkeypatch):
    monkeypatch.setattr(C, "MiniMaxH3Client", _RecordingClient)
    _RecordingClient.calls.clear()
    result = asyncio.run(
        C._cancel_created_job(
            base_url="http://127.0.0.1:8000/",
            api_key="k",
            task_id="6F9619FF-8B86-D011-B42D-00C04FC964FF",
        )
    )
    assert result == {
        "id": "6f9619ff-8b86-d011-b42d-00c04fc964ff",
        "status": "cancelled",
    }
    assert _RecordingClient.calls == [
        ("http://127.0.0.1:8000", "6f9619ff-8b86-d011-b42d-00c04fc964ff")
    ]


def test_job_uuid_is_the_canonical_form():
    assert (
        C._job_uuid("6F9619FF-8B86-D011-B42D-00C04FC964FF")
        == "6f9619ff-8b86-d011-b42d-00c04fc964ff"
    )


# --- task awareness -------------------------------------------------------------------

FIELD_CHECKS = [
    "missing_bearer_authentication",
    "invalid_bearer_authentication",
    "missing_prompt",
    "provider_model_field_is_rejected",
    "provider_resolution_field_is_rejected",
    "provider_ratio_field_is_rejected",
    "provider_duration_field_is_rejected",
    "unsupported_aspect_ratio",
    "duration_below_minimum",
    "duration_above_maximum",
    "explicit_num_inference_steps_is_rejected",
]


@pytest.mark.parametrize(
    ("task", "route_task", "refused"),
    [
        ("t2va", "t2va", ["fl2va", "ref2va"]),
        # FL2VA serves text-only /generations: the t2va checks pass there as-is.
        ("fl2va", "t2va", ["ref2va"]),
        ("ref2va", "ref2va", ["t2va", "fl2va"]),
    ],
)
def test_checks_follow_the_deployment_task(task, route_task, refused):
    cases = C._cases("validation", task)
    field_cases = cases[: len(FIELD_CHECKS)]
    assert [case.name for case in field_cases] == FIELD_CHECKS
    assert {case.route_task for case in field_cases} == {route_task}
    assert not any(case.expects_refusal for case in field_cases)

    refusals = cases[len(FIELD_CHECKS) :]
    assert [case.name for case in refusals] == [
        f"{t}_route_is_refused" for t in refused
    ]
    assert [case.route_task for case in refusals] == refused
    assert all(
        case.expects_refusal and case.expected_status == 422 for case in refusals
    )
    # An empty body: a served route answers it with field errors, never with a job.
    assert all(case.payload == {} for case in refusals)


def test_ref2va_field_checks_carry_the_reference_image():
    cases = {case.name: case for case in C._cases("validation", "ref2va")}
    for name in FIELD_CHECKS:
        assert cases[name].payload["references"]["images"][0]["b64"]
    assert cases["unsupported_aspect_ratio"].payload["aspect_ratio"] == "2:1"
    assert "prompt" not in cases["missing_prompt"].payload
    assert (
        cases["explicit_num_inference_steps_is_rejected"].payload["num_inference_steps"]
        == 50
    )


def test_t2va_field_checks_send_no_media():
    for case in C._cases("validation", "t2va")[: len(FIELD_CHECKS)]:
        assert not {"references", "image_prompts"} & set(case.payload)


@pytest.mark.parametrize(
    ("task", "name", "route_task"),
    [
        ("t2va", "valid_t2va_job", "t2va"),
        ("fl2va", "valid_t2va_job", "t2va"),
        ("ref2va", "valid_ref2va_job", "ref2va"),
    ],
)
def test_smoke_job_uses_the_request_shape(task, name, route_task):
    smoke = C._cases("smoke", task)[-1]
    assert (smoke.name, smoke.route_task, smoke.expected_status) == (
        name,
        route_task,
        202,
    )
    assert smoke.requires_job_id


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="task must be one of"):
        asyncio.run(
            C.run_create_contract(
                base_url="http://127.0.0.1:9", api_key="k", task="i2v"
            )
        )


class _Response:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class _Session:
    """Answers every POST with one canned response and records the URLs."""

    def __init__(self, status, body):
        self.status, self.body, self.urls = status, body, []

    def post(self, url, headers=None, json=None):
        self.urls.append(url)
        return _Response(self.status, self.body)


def _refusal_case(route_task="ref2va"):
    return C._RequestCase(
        f"{route_task}_route_is_refused",
        {},
        422,
        route_task=route_task,
        expects_refusal=True,
    )


def _no_client(**kwargs):
    raise AssertionError(f"no cancel expected: {kwargs}")


def test_route_refusal_passes_on_a_string_detail(monkeypatch):
    monkeypatch.setattr(C, "MiniMaxH3Client", _no_client)
    session = _Session(422, '{"detail": "This deployment does not serve Ref2VA."}')
    result = asyncio.run(
        C._run_case(
            session, base_url="http://h3:8000", api_key="k", case=_refusal_case()
        )
    )
    assert result["passed"] is True
    assert session.urls == ["http://h3:8000/v1/videos/generations/ref2va"]
    assert result["endpoint_url"] == session.urls[0]


def test_route_refusal_fails_on_field_validation():
    # A list detail means the route served the request and only the body was refused.
    session = _Session(422, '{"detail": [{"loc": ["body", "prompt"], "msg": "x"}]}')
    result = asyncio.run(
        C._run_case(
            session, base_url="http://h3:8000", api_key="k", case=_refusal_case()
        )
    )
    assert result["passed"] is False
    assert result["message"] == "422 was field validation, not a route refusal"


def test_field_check_is_posted_to_its_route():
    case = C._cases("validation", "ref2va")[2]  # missing_prompt
    session = _Session(422, '{"detail": [{"loc": ["body", "prompt"], "msg": "x"}]}')
    result = asyncio.run(
        C._run_case(session, base_url="http://h3:8000", api_key="k", case=case)
    )
    assert result["passed"] is True
    assert session.urls == ["http://h3:8000/v1/videos/generations/ref2va"]


JOB_ID = "6f9619ff-8b86-d011-b42d-00c04fc964ff"


@pytest.mark.parametrize(
    "case",
    [
        _refusal_case("fl2va"),
        C._cases("validation", "ref2va")[3],  # provider_model_field_is_rejected
    ],
    ids=["route_refusal", "field_check"],
)
def test_an_accepted_negative_check_fails_and_its_job_is_cancelled(case, monkeypatch):
    # A regressed gate or validator queues a real job; it must not outlive the suite.
    monkeypatch.setattr(C, "MiniMaxH3Client", _RecordingClient)
    _RecordingClient.calls.clear()
    session = _Session(202, f'{{"id": "{JOB_ID}", "status": "queued"}}')
    result = asyncio.run(
        C._run_case(session, base_url="http://h3:8000", api_key="k", case=case)
    )
    assert result["passed"] is False
    assert result["actual_status"] == 202
    assert result["task_id"] == JOB_ID
    assert result["cancellation"] == {"id": JOB_ID, "status": "cancelled"}
    assert result["message"] == "request was accepted; its job was cancelled"
    assert _RecordingClient.calls == [("http://h3:8000", JOB_ID)]


def test_an_accepted_negative_check_reports_a_failed_cancel(monkeypatch):
    async def no_cancel(**kwargs):
        return None

    monkeypatch.setattr(C, "_cancel_created_job", no_cancel)
    session = _Session(202, f'{{"id": "{JOB_ID}"}}')
    result = asyncio.run(
        C._run_case(
            session, base_url="http://h3:8000", api_key="k", case=_refusal_case()
        )
    )
    assert result["passed"] is False
    assert result["task_id"] == JOB_ID and result["cancellation"] is None
    assert result["message"] == "request was accepted; its job could not be cancelled"


def test_a_refused_negative_check_cancels_nothing(monkeypatch):
    monkeypatch.setattr(C, "MiniMaxH3Client", _no_client)
    case = C._cases("validation", "t2va")[3]  # provider_model_field_is_rejected
    session = _Session(422, f'{{"id": "{JOB_ID}", "detail": [{{"msg": "x"}}]}}')
    result = asyncio.run(
        C._run_case(session, base_url="http://h3:8000", api_key="k", case=case)
    )
    assert result["passed"] is True
    assert result["task_id"] is None and result["cancellation"] is None


def test_the_smoke_job_is_cancelled(monkeypatch):
    monkeypatch.setattr(C, "MiniMaxH3Client", _RecordingClient)
    _RecordingClient.calls.clear()
    smoke = C._cases("smoke", "ref2va")[-1]
    session = _Session(202, f'{{"id": "{JOB_ID}", "status": "queued"}}')
    result = asyncio.run(
        C._run_case(session, base_url="http://h3:8000", api_key="k", case=smoke)
    )
    assert result["passed"] is True
    assert result["message"] == ""
    assert _RecordingClient.calls == [("http://h3:8000", JOB_ID)]
    assert session.urls == ["http://h3:8000/v1/videos/generations/ref2va"]


def test_workflow_test_resolves_the_task_from_the_spec(monkeypatch):
    seen = {}

    async def fake_run_create_contract(**kwargs):
        seen.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(C, "run_create_contract", fake_run_create_contract)
    ctx = SimpleNamespace(
        service_port=8000,
        base_url="http://127.0.0.1:8000",
        model_spec=SimpleNamespace(env_vars={"MODEL_RUNNER": "tt-minimax-h3-ref2va"}),
    )
    test = C.MiniMaxH3CreateContractTest(
        C.TestConfig({"timeout": 5, "retry_attempts": 0, "retry_delay": 0}), {}, ctx=ctx
    )
    asyncio.run(test._run_specific_test_async())
    assert seen["task"] == "ref2va"
    assert seen["base_url"] == "http://127.0.0.1:8000"

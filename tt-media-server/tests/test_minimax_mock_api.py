"""Contract tests for MiniMax mock authentication and request validation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from minimax_mock.app import create_app
from minimax_mock.download_signer import DownloadSigner
from minimax_mock.fixture_resolver import FixtureCatalog
from minimax_mock.schemas import VideoGenerationRequest
from minimax_mock.task_store import TaskStore

API_KEY = "minimax-mock-test-key"


@dataclass
class _FakeClock:
    wall: float = 1_700_000_000.0
    monotonic: float = 100.0

    def wall_time(self) -> float:
        return self.wall

    def monotonic_time(self) -> float:
        return self.monotonic

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.monotonic += seconds


def _minimal_payload() -> dict:
    return {
        "model": "MiniMax-H3",
        "content": [
            {
                "type": "text",
                "text": "A paper airplane glides through a sunlit library.",
            }
        ],
        "resolution": "2K",
        "duration": 5,
        "ratio": "16:9",
    }


def _headers(content_type: str = "application/json") -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": content_type,
    }


def _task_store(clock: _FakeClock, task_ids: tuple[str, ...]) -> TaskStore:
    ids = iter(task_ids)
    return TaskStore(
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        task_id_factory=lambda: next(ids),
    )


@pytest.fixture
def client():
    with TestClient(create_app(API_KEY)) as test_client:
        yield test_client


def test_create_requires_bearer_authentication(client):
    missing = client.post("/v2/video_generation", json=_minimal_payload())
    invalid = client.post(
        "/v2/video_generation",
        json=_minimal_payload(),
        headers={"Authorization": "Bearer wrong-key"},
    )

    for response in (missing, invalid):
        assert response.status_code == 401
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authorized_error"
        assert body["error"]["http_code"] == "401"
        assert body["error"]["message"].endswith("(1004)")
        assert len(body["request_id"]) == 32


def test_create_requires_json_content_type(client):
    response = client.post(
        "/v2/video_generation",
        content=json.dumps(_minimal_payload()),
        headers=_headers("text/plain"),
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "bad_request_error"
    assert response.json()["error"]["http_code"] == "400"


def test_create_accepts_json_content_type_with_charset(client):
    response = client.post(
        "/v2/video_generation",
        content=json.dumps(_minimal_payload()),
        headers=_headers("application/json; charset=utf-8"),
    )

    assert response.status_code == 200


def test_create_persists_request_and_selected_fixture():
    task_store = TaskStore(task_id_factory=lambda: "100000000000001")
    with TestClient(create_app(API_KEY, task_store=task_store)) as test_client:
        response = test_client.post(
            "/v2/video_generation",
            json=_minimal_payload(),
            headers=_headers(),
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "100000000000001"}
    task = task_store.get("100000000000001")
    assert task is not None
    assert task.request.content[0].text == _minimal_payload()["content"][0]["text"]
    assert task.fixture.manifest.name == "t2v-success"


def test_query_transitions_task_and_downloads_fixture():
    task_id = "100000000000001"
    clock = _FakeClock()
    task_store = _task_store(clock, (task_id,))
    download_signer = DownloadSigner(
        API_KEY,
        ttl_seconds=5,
        clock=clock.wall_time,
    )
    application = create_app(
        API_KEY,
        task_store=task_store,
        download_signer=download_signer,
    )

    with TestClient(application) as test_client:
        created = test_client.post(
            "/v2/video_generation",
            json=_minimal_payload(),
            headers=_headers(),
        )
        assert created.json() == {"task_id": task_id}

        queued = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        assert queued.status_code == 200
        assert queued.json()["task"]["status"] == "queued"
        assert "content" not in queued.json()["task"]
        assert queued.json()["task"]["usage"]["total_seconds"] == 0

        stored_task = task_store.get(task_id)
        clock.advance(stored_task.fixture.manifest.queued_for_ms / 1000)
        running = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        assert running.json()["task"]["status"] == "running"

        clock.advance(stored_task.fixture.manifest.running_for_ms / 1000)
        succeeded = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        task = succeeded.json()["task"]
        assert task["status"] == "succeeded"
        assert task["model"] == "MiniMax-H3"
        assert task["resolution"] == "2K"
        assert task["duration"] == 5
        assert task["ratio"] == "16:9"
        assert task["task_type"] == "generation"
        assert task["modality"] == "video"
        assert task["usage"] == {
            "total_seconds": 5,
            "input_seconds": 0,
            "output_seconds": 5,
            "input_image_count": 0,
        }

        content_url = task["content"]["url"]
        downloaded = test_client.get(content_url)
        assert downloaded.status_code == 200
        assert downloaded.headers["content-type"].startswith("video/mp4")
        assert b"ftyp" in downloaded.content[:32]

        tampered = test_client.get(
            content_url.replace("signature=", "signature=invalid")
        )
        assert tampered.status_code == 403

        clock.advance(6)
        expired = test_client.get(content_url)
        assert expired.status_code == 403


def test_query_rejects_unknown_task_and_requires_authentication(client):
    missing_auth = client.get("/v2/query/video_generation/100000000000001")
    assert missing_auth.status_code == 401

    unknown = client.get(
        "/v2/query/video_generation/100000000000001",
        headers=_headers(),
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["message"] == "invalid task_id (2013)"


def test_query_returns_fixture_error_for_failed_task():
    task_id = "100000000000001"
    clock = _FakeClock()
    task_store = _task_store(clock, (task_id,))
    request = VideoGenerationRequest.model_validate(_minimal_payload())
    fixture = FixtureCatalog().resolve(request, scenario_name="generation-failed")
    task_store.create(request, fixture)
    clock.advance(
        (fixture.manifest.queued_for_ms + fixture.manifest.running_for_ms) / 1000
    )

    with TestClient(create_app(API_KEY, task_store=task_store)) as test_client:
        response = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )

    task = response.json()["task"]
    assert task["status"] == "failed"
    assert task["error"] == {
        "code": "1026",
        "message": "video description contains sensitive content",
    }
    assert "content" not in task
    assert task["usage"]["total_seconds"] == 0


def test_delete_endpoint_cancels_queued_task():
    task_id = "100000000000001"
    clock = _FakeClock()
    task_store = _task_store(clock, (task_id,))

    with TestClient(create_app(API_KEY, task_store=task_store)) as test_client:
        test_client.post(
            "/v2/video_generation",
            json=_minimal_payload(),
            headers=_headers(),
        )

        missing_auth = test_client.delete(f"/v2/video_generation/{task_id}")
        assert missing_auth.status_code == 401

        cancelled = test_client.delete(
            f"/v2/video_generation/{task_id}",
            headers=_headers(),
        )
        assert cancelled.status_code == 200
        assert cancelled.json() == {
            "task_id": task_id,
            "action": "cancelled",
            "status": "cancelled",
        }

        queried = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        assert queried.json()["task"]["status"] == "cancelled"

        repeated = test_client.delete(
            f"/v2/video_generation/{task_id}",
            headers=_headers(),
        )
        assert repeated.status_code == 400
        assert "status is cancelled" in repeated.json()["error"]["message"]


def test_delete_endpoint_rejects_running_task():
    task_id = "100000000000001"
    clock = _FakeClock()
    task_store = _task_store(clock, (task_id,))

    with TestClient(create_app(API_KEY, task_store=task_store)) as test_client:
        test_client.post(
            "/v2/video_generation",
            json=_minimal_payload(),
            headers=_headers(),
        )
        fixture = task_store.get(task_id).fixture
        clock.advance(fixture.manifest.queued_for_ms / 1000)

        response = test_client.delete(
            f"/v2/video_generation/{task_id}",
            headers=_headers(),
        )

        assert response.status_code == 400
        assert "status is running" in response.json()["error"]["message"]
        assert task_store.get(task_id).status.value == "running"


def test_delete_endpoint_removes_succeeded_task_and_invalidates_download():
    task_id = "100000000000001"
    clock = _FakeClock()
    task_store = _task_store(clock, (task_id,))
    download_signer = DownloadSigner(
        API_KEY,
        ttl_seconds=60,
        clock=clock.wall_time,
    )

    with TestClient(
        create_app(
            API_KEY,
            task_store=task_store,
            download_signer=download_signer,
        )
    ) as test_client:
        test_client.post(
            "/v2/video_generation",
            json=_minimal_payload(),
            headers=_headers(),
        )
        fixture = task_store.get(task_id).fixture
        clock.advance(
            (fixture.manifest.queued_for_ms + fixture.manifest.running_for_ms) / 1000
        )
        succeeded = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        content_url = succeeded.json()["task"]["content"]["url"]

        deleted = test_client.delete(
            f"/v2/video_generation/{task_id}",
            headers=_headers(),
        )
        assert deleted.json() == {
            "task_id": task_id,
            "action": "deleted",
            "status": "deleted",
        }

        missing = test_client.get(
            f"/v2/query/video_generation/{task_id}",
            headers=_headers(),
        )
        assert missing.status_code == 400

        download = test_client.get(content_url)
        assert download.status_code == 404


def test_delete_endpoint_rejects_unknown_task(client):
    response = client.delete(
        "/v2/video_generation/100000000000001",
        headers=_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "invalid task_id (2013)"


def test_list_tasks_supports_pagination_and_filters():
    task_ids = (
        "100000000000001",
        "100000000000002",
        "100000000000003",
    )
    clock = _FakeClock()
    task_store = _task_store(clock, task_ids)

    with TestClient(create_app(API_KEY, task_store=task_store)) as test_client:
        fixture = FixtureCatalog().resolve(
            VideoGenerationRequest.model_validate(_minimal_payload())
        )
        advances_between_tasks = (
            fixture.manifest.running_for_ms / 1000,
            fixture.manifest.queued_for_ms / 1000,
        )
        for index in range(3):
            response = test_client.post(
                "/v2/video_generation",
                json=_minimal_payload(),
                headers=_headers(),
            )
            assert response.json()["task_id"] == task_ids[index]
            if index < len(advances_between_tasks):
                clock.advance(advances_between_tasks[index])

        listed = test_client.get(
            "/v2/query/video_generation",
            headers=_headers(),
        )
        assert listed.status_code == 200
        assert listed.json()["total"] == 3
        assert [(task["id"], task["status"]) for task in listed.json()["items"]] == [
            (task_ids[2], "queued"),
            (task_ids[1], "running"),
            (task_ids[0], "succeeded"),
        ]

        second_page = test_client.get(
            "/v2/query/video_generation",
            params={"page_num": 2, "page_size": 2},
            headers=_headers(),
        )
        assert second_page.json()["total"] == 3
        assert [task["id"] for task in second_page.json()["items"]] == [task_ids[0]]

        succeeded = test_client.get(
            "/v2/query/video_generation",
            params={"filter.status": "succeeded"},
            headers=_headers(),
        )
        assert succeeded.json()["total"] == 1
        assert succeeded.json()["items"][0]["id"] == task_ids[0]
        assert "content" in succeeded.json()["items"][0]

        selected = test_client.get(
            "/v2/query/video_generation",
            params=[
                ("filter.task_ids", task_ids[0]),
                ("filter.task_ids", task_ids[2]),
            ],
            headers=_headers(),
        )
        assert [task["id"] for task in selected.json()["items"]] == [
            task_ids[2],
            task_ids[0],
        ]

        other_task_type = test_client.get(
            "/v2/query/video_generation",
            params={"filter.task_type": "h3_context_ir"},
            headers=_headers(),
        )
        assert other_task_type.json() == {"items": [], "total": 0}


def test_list_tasks_validates_pagination_and_requires_authentication(client):
    missing_auth = client.get("/v2/query/video_generation")
    assert missing_auth.status_code == 401

    invalid_page = client.get(
        "/v2/query/video_generation",
        params={"page_num": 0},
        headers=_headers(),
    )
    _assert_bad_request(invalid_page)


@pytest.mark.parametrize(
    "content,ratio",
    [
        (
            [{"type": "text", "text": "A lighthouse during a storm."}],
            "21:9",
        ),
        (
            [
                {"type": "text", "text": "Animate the opening frame."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/first.png"},
                },
            ],
            None,
        ),
        (
            [
                {"type": "text", "text": "Arrive at the final frame."},
                {
                    "type": "image_url",
                    "image_url": {"url": "mm_file://last-frame-id"},
                    "role": "last_frame",
                },
            ],
            "adaptive",
        ),
        (
            [
                {"type": "text", "text": "Move between these frames."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                    "role": "first_frame",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/last.webp"},
                    "role": "last_frame",
                },
            ],
            "16:9",
        ),
        (
            [
                {"type": "text", "text": "Follow the references."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/reference.jpg"},
                    "role": "reference_image",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/reference.mp4"},
                    "role": "reference_video",
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": "data:audio/mp3;base64,aGVsbG8="},
                    "role": "reference_audio",
                },
            ],
            None,
        ),
    ],
)
def test_create_accepts_documented_content_combinations(client, content, ratio):
    payload = _minimal_payload()
    payload["content"] = content
    if ratio is None:
        payload.pop("ratio", None)
    else:
        payload["ratio"] = ratio

    response = client.post("/v2/video_generation", json=payload, headers=_headers())

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"task_id"}
    assert response.json()["task_id"].isdigit()
    assert len(response.json()["task_id"]) == 15


@pytest.mark.parametrize("field", ["model", "content", "resolution", "duration"])
def test_create_rejects_missing_required_fields(client, field):
    payload = _minimal_payload()
    del payload[field]

    response = client.post("/v2/video_generation", json=payload, headers=_headers())

    _assert_bad_request(response)


def _invalid_payloads():
    wrong_model = _minimal_payload()
    wrong_model["model"] = "Wan2.2"

    no_text = _minimal_payload()
    no_text["content"] = [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/frame.png"},
        }
    ]

    empty_text = _minimal_payload()
    empty_text["content"][0]["text"] = "   "

    missing_ratio = _minimal_payload()
    del missing_ratio["ratio"]

    adaptive_t2v = _minimal_payload()
    adaptive_t2v["ratio"] = "adaptive"

    mixed_modes = _minimal_payload()
    mixed_modes["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/first.png"},
                "role": "first_frame",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/reference.mp3"},
                "role": "reference_audio",
            },
        ]
    )

    missing_video_role = _minimal_payload()
    missing_video_role["content"].append(
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/reference.mp4"},
        }
    )

    invalid_media_url = _minimal_payload()
    invalid_media_url["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": "file:///tmp/frame.png"},
            "role": "first_frame",
        }
    )

    float_duration = _minimal_payload()
    float_duration["duration"] = 5.0

    invalid_callback = _minimal_payload()
    invalid_callback["callback_url"] = "not-a-url"

    unknown_field = _minimal_payload()
    unknown_field["seed"] = 42

    return [
        wrong_model,
        no_text,
        empty_text,
        missing_ratio,
        adaptive_t2v,
        mixed_modes,
        missing_video_role,
        invalid_media_url,
        float_duration,
        invalid_callback,
        unknown_field,
    ]


@pytest.mark.parametrize("payload", _invalid_payloads())
def test_create_rejects_invalid_requests(client, payload):
    response = client.post(
        "/v2/video_generation",
        json=deepcopy(payload),
        headers=_headers(),
    )

    _assert_bad_request(response)


def test_app_requires_mock_api_key_at_startup(monkeypatch):
    monkeypatch.delenv("MINIMAX_MOCK_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError, match="MINIMAX_MOCK_API_KEY must be configured"
    ), TestClient(create_app()):
        pass


def _assert_bad_request(response) -> None:
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "bad_request_error"
    assert body["error"]["http_code"] == "400"
    assert body["error"]["message"].startswith("invalid params,")
    assert body["error"]["message"].endswith("(2013)")
    assert len(body["request_id"]) == 32

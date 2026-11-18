# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def weight_update_api(monkeypatch):
    class Router:
        @staticmethod
        def post(*args, **kwargs):
            return lambda function: function

        @staticmethod
        def get(*args, **kwargs):
            return lambda function: function

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class BaseModel:
        def __init__(self, **kwargs):
            for cls in reversed(type(self).mro()):
                for name in getattr(cls, "__annotations__", {}):
                    if name in kwargs:
                        value = kwargs[name]
                    elif hasattr(type(self), name):
                        value = getattr(type(self), name)
                    else:
                        raise TypeError(f"missing required field: {name}")
                    setattr(self, name, value)

    class JSONResponse:
        def __init__(self, content, status_code, headers):
            self.content = content
            self.status_code = status_code
            self.headers = headers

        async def __call__(self, scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

    fastapi = types.ModuleType("fastapi")
    fastapi.APIRouter = lambda *args, **kwargs: Router()
    fastapi.Depends = lambda dependency: dependency
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = BaseModel
    pydantic.Field = lambda default, **kwargs: default
    monkeypatch.setitem(sys.modules, "pydantic", pydantic)

    starlette = types.ModuleType("starlette")
    responses = types.ModuleType("starlette.responses")
    responses.JSONResponse = JSONResponse
    starlette_types = types.ModuleType("starlette.types")
    starlette_types.ASGIApp = object
    starlette_types.Receive = object
    starlette_types.Scope = dict
    starlette_types.Send = object
    monkeypatch.setitem(sys.modules, "starlette", starlette)
    monkeypatch.setitem(sys.modules, "starlette.responses", responses)
    monkeypatch.setitem(sys.modules, "starlette.types", starlette_types)

    path = Path(__file__).parents[1] / "vllm-tt-metal" / "src" / "weight_update_api.py"
    spec = importlib.util.spec_from_file_location("weight_update_api_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


async def _receive():
    return {"type": "http.request", "body": b"", "more_body": False}


def _scope(path: str):
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "query_string": b"",
    }


def test_admission_gate_rejects_requests_during_update(weight_update_api):
    async def scenario():
        async def downstream(scope, receive, send):
            raise AssertionError("rejected request reached downstream app")

        gate = weight_update_api._AdmissionGate()
        gate.in_progress = True
        middleware = weight_update_api._AdmissionGateMiddleware(downstream, gate)
        sent = []

        async def send(message):
            sent.append(message)

        await middleware(_scope("/v1/completions"), _receive, send)

        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 503

    asyncio.run(scenario())


def test_weight_update_waits_for_previously_admitted_request(weight_update_api):
    async def scenario():
        admitted = asyncio.Event()
        release = asyncio.Event()

        async def downstream(scope, receive, send):
            admitted.set()
            await release.wait()

        gate = weight_update_api._AdmissionGate()
        middleware = weight_update_api._AdmissionGateMiddleware(downstream, gate)
        request_task = asyncio.create_task(
            middleware(_scope("/v1/completions"), _receive, lambda message: None)
        )
        await admitted.wait()
        assert gate.inflight == 1

        calls = []

        class OutputProcessor:
            @staticmethod
            def has_unfinished_requests():
                return False

        class EngineClient:
            output_processor = OutputProcessor()

            async def collective_rpc(self, method, kwargs=None):
                calls.append((method, kwargs))
                if method == "get_weight_transfer_status":
                    return [{"owns_model": True, "initialized": True}]
                return [{"updated": True, "version": 1}]

        app = SimpleNamespace(
            state=SimpleNamespace(
                engine_client=EngineClient(),
                _tt_weight_update_gate=gate,
                _tt_weight_update_lock=asyncio.Lock(),
            )
        )
        request = SimpleNamespace(app=app)
        update_task = asyncio.create_task(
            weight_update_api.update_weights(
                weight_update_api.WeightUpdateRequest(), request
            )
        )

        await asyncio.sleep(0)
        assert gate.in_progress
        assert calls == []

        release.set()
        await request_task
        response = await update_task

        assert response.version == 1
        assert [method for method, _ in calls] == [
            "get_weight_transfer_status",
            "update_weights",
        ]
        assert gate.inflight == 0
        assert not gate.in_progress

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "case,status",
    [
        ("uninitialized", 409),
        ("update_failure", 500),
        ("status_failure", 500),
        ("no_owner", 500),
        ("ready", 200),
    ],
)
def test_weight_update_initialization_error_and_gate_cleanup(
    weight_update_api, case, status
):
    async def scenario():
        calls = []
        version = 7

        class Engine:
            output_processor = SimpleNamespace(has_unfinished_requests=lambda: False)

            async def collective_rpc(self, method, kwargs=None):
                nonlocal version
                calls.append(method)
                if method == "get_weight_transfer_status":
                    if case == "status_failure":
                        raise RuntimeError("RPC failed")
                    return [
                        {
                            "owns_model": case != "no_owner",
                            "initialized": case != "uninitialized",
                        },
                        {"owns_model": False, "initialized": False},
                    ]
                if case == "update_failure":
                    raise RuntimeError("Device transfer failed")
                version += 1
                return [{"updated": True, "version": version}]

            async def reset_prefix_cache(self):
                calls.append("reset_prefix_cache")

        gate = weight_update_api._AdmissionGate()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    engine_client=Engine(),
                    _tt_weight_update_gate=gate,
                    _tt_weight_update_lock=asyncio.Lock(),
                )
            )
        )
        if status == 200:
            response = await weight_update_api.update_weights(
                weight_update_api.WeightUpdateRequest(), request
            )
            assert response.version == 8
            assert calls == [
                "get_weight_transfer_status",
                "update_weights",
                "reset_prefix_cache",
            ]
        else:
            with pytest.raises(weight_update_api.HTTPException) as error:
                await weight_update_api.update_weights(
                    weight_update_api.WeightUpdateRequest(), request
                )
            assert error.value.status_code == status
            assert version == 7
            assert "reset_prefix_cache" not in calls
            if status == 409:
                assert error.value.detail["code"] == "weight_transfer_not_initialized"
                assert calls == ["get_weight_transfer_status"]
        assert not gate.in_progress
        assert not request.app.state._tt_weight_update_lock.locked()

    asyncio.run(scenario())

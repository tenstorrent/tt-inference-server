# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "path",
    [
        "/pause",
        "/resume",
        "/update_weights",
        "/init_weight_transfer_engine",
        "/get_world_size",
        "/is_paused",
    ],
)
@pytest.mark.parametrize(
    "keys,env_key,header,status",
    [
        (None, "secret", None, 401),
        (None, "secret", "Bearer wrong", 401),
        (None, "secret", "Bearer secrex", 401),
        (None, "secret", "Bearer secret-extra", 401),
        (None, "secret", "Bearer ", 401),
        (None, "secret", "Basic secret", 401),
        (None, "secret", "Bearer secret", 200),
        (None, "secret", "bEaReR secret", 200),
        (["first", "second"], "env", "Bearer first", 200),
        (["first", "second"], "env", "Bearer second", 200),
        (["cli"], "env", "Bearer env", 401),
        (None, None, None, 200),
    ],
)
def test_native_routes_follow_server_auth_policy(
    monkeypatch, path, keys, env_key, header, status
):
    if env_key is None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VLLM_API_KEY", env_key)
    source = Path(__file__).parents[1] / "vllm-tt-metal/src/weight_update_api.py"
    spec = importlib.util.spec_from_file_location(
        "weight_update_auth_under_test", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = ModuleType("vllm.entrypoints.serve.rlhf.api_router")
    native.router = APIRouter()
    calls = []

    async def handler():
        calls.append(path)
        return {"status": "ok"}

    method = "GET" if path in ("/get_world_size", "/is_paused") else "POST"
    native.router.add_api_route(path, handler, methods=[method])
    monkeypatch.setitem(sys.modules, native.__name__, native)
    app = FastAPI()
    app.state.args = SimpleNamespace(api_key=keys)
    module._mount_native_rl_routes(app)
    with TestClient(app) as client:
        response = client.request(
            method, path, headers={"Authorization": header} if header else {}
        )
    assert response.status_code == status
    assert calls == ([path] if status == 200 else [])

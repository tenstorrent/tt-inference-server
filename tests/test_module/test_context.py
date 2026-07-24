# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``test_module.context``: URL derivation, token counting,
metadata helpers and the health-check wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module import context as ctx_mod
from test_module._test_common import HardwareRequirement
from test_module.context import (
    MediaContext,
    common_eval_metadata,
    common_report_metadata,
    count_tokens,
    get_health,
    require_health,
)


def _ctx(
    server_url=None,
    service_port=8000,
    *,
    model="m",
    device="N300",
    max_concurrency=4,
    all_params=None,
) -> MediaContext:
    return MediaContext(
        all_params=all_params if all_params is not None else [],
        model_spec=SimpleNamespace(
            model_name=model,
            device_model_spec=SimpleNamespace(max_concurrency=max_concurrency),
        ),
        device=SimpleNamespace(name=device),
        output_path="/tmp/out",
        service_port=service_port,
        server_url=server_url,
    )


class TestMediaContextUrls:
    def test_default_deploy_url_uses_localhost(self):
        c = _ctx()
        assert c.base_url == "http://127.0.0.1:8000"
        assert c.server_host == "http://127.0.0.1"
        assert c.server_port == 8000

    def test_explicit_server_url_port_overrides_service_port(self):
        c = _ctx(server_url="http://myhost:9000", service_port=8000)
        assert c.base_url == "http://myhost:9000"
        assert c.server_host == "http://myhost"
        assert c.server_port == 9000

    def test_server_url_without_port_falls_back_to_service_port(self):
        c = _ctx(server_url="http://1.2.3.4", service_port=8001)
        assert c.base_url == "http://1.2.3.4:8001"
        assert c.server_port == 8001

    def test_test_payloads_path(self):
        assert _ctx().test_payloads_path == "utils/test_payloads"


class TestCountTokens:
    def test_blank_text_is_zero(self, monkeypatch):
        # Returns before consulting any tokenizer.
        monkeypatch.setattr(
            ctx_mod, "get_tokenizer", lambda repo: pytest.fail("unused")
        )
        assert count_tokens("any/repo", "   ") == 0

    def test_word_count_fallback_when_no_tokenizer(self, monkeypatch):
        monkeypatch.setattr(ctx_mod, "get_tokenizer", lambda repo: None)
        assert count_tokens("any/repo", "one two three") == 3

    def test_uses_tokenizer_when_available(self, monkeypatch):
        fake = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [1, 2, 3, 4]
        )
        monkeypatch.setattr(ctx_mod, "get_tokenizer", lambda repo: fake)
        assert count_tokens("any/repo", "whatever") == 4

    def test_tokenizer_error_falls_back_to_word_count(self, monkeypatch):
        def _boom(text, add_special_tokens=False):
            raise RuntimeError("encode failed")

        monkeypatch.setattr(
            ctx_mod, "get_tokenizer", lambda repo: SimpleNamespace(encode=_boom)
        )
        assert count_tokens("any/repo", "a b") == 2


class TestMetadataHelpers:
    def test_common_report_metadata(self):
        meta = common_report_metadata(
            _ctx(model="my-model", device="N300"), "benchmark"
        )
        assert meta["model"] == "my-model"
        assert meta["device"] == "N300"
        assert meta["task_type"] == "benchmark"
        assert "timestamp" in meta

    def test_common_eval_metadata_lowercases_device_and_pulls_task(self):
        params = SimpleNamespace(
            tasks=[
                SimpleNamespace(task_name="mmlu", score=SimpleNamespace(tolerance=0.05))
            ]
        )
        meta = common_eval_metadata(
            _ctx(device="N300", all_params=params), "evaluation"
        )
        assert meta["device"] == "n300"
        assert meta["task_name"] == "mmlu"
        assert meta["tolerance"] == 0.05


class TestHealth:
    def _liveness_block(self, **data):
        return SimpleNamespace(data=data)

    def test_get_health_full_board_requires_all_workers(self, monkeypatch):
        captured = {}

        def fake_liveness(ctx, min_required):
            captured["min_required"] = min_required
            return self._liveness_block(
                success=True, runner_in_use="tt-metal", ready_count=4, attempts=1
            )

        monkeypatch.setattr(ctx_mod, "run_device_liveness", fake_liveness)
        ok, runner = get_health(_ctx(max_concurrency=4), HardwareRequirement.FULL_BOARD)
        assert ok is True and runner == "tt-metal"
        assert captured["min_required"] == 4  # full board

    def test_get_health_any_chip_needs_only_one(self, monkeypatch):
        captured = {}

        def fake_liveness(ctx, min_required):
            captured["min_required"] = min_required
            return self._liveness_block(success=True, runner_in_use="x", ready_count=1)

        monkeypatch.setattr(ctx_mod, "run_device_liveness", fake_liveness)
        get_health(_ctx(max_concurrency=8), HardwareRequirement.ANY_CHIP)
        assert captured["min_required"] == 1

    def test_get_health_failure_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            ctx_mod,
            "run_device_liveness",
            lambda ctx, n: self._liveness_block(success=False),
        )
        assert get_health(_ctx()) == (False, None)

    def test_get_health_swallows_exceptions(self, monkeypatch):
        def _raise(ctx, n):
            raise RuntimeError("boom")

        monkeypatch.setattr(ctx_mod, "run_device_liveness", _raise)
        assert get_health(_ctx()) == (False, None)

    def test_require_health_returns_runner_on_success(self, monkeypatch):
        monkeypatch.setattr(
            ctx_mod,
            "run_device_liveness",
            lambda ctx, n: self._liveness_block(success=True, runner_in_use="tt-metal"),
        )
        assert require_health(_ctx()) == "tt-metal"

    def test_require_health_raises_on_failure(self, monkeypatch):
        monkeypatch.setattr(
            ctx_mod,
            "run_device_liveness",
            lambda ctx, n: self._liveness_block(success=False),
        )
        with pytest.raises(RuntimeError, match="Health check failed"):
            require_health(_ctx())

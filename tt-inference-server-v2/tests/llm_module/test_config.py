# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``llm_module.config`` dataclasses + URL derivation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from llm_module.config import DriverContext, LLMRunConfig, ServerConnection


class TestLLMRunConfig:
    def test_is_frozen(self):
        cfg = LLMRunConfig(isl=128, osl=64, max_concurrency=4, num_prompts=10)
        with pytest.raises(FrozenInstanceError):
            cfg.isl = 256  # type: ignore[misc]


class TestServerConnection:
    def test_tokenizer_defaults_to_model(self):
        conn = ServerConnection(base_url="localhost", service_port=8000, model="m")
        assert conn.tokenizer == "m"

    def test_explicit_tokenizer_is_kept(self):
        conn = ServerConnection(
            base_url="localhost", service_port=8000, model="m", tokenizer="tok"
        )
        assert conn.tokenizer == "tok"

    def test_url_with_port_adds_scheme_and_port(self):
        conn = ServerConnection(base_url="localhost", service_port=8000, model="m")
        assert conn.url_with_port == "http://localhost:8000"

    def test_url_with_port_keeps_existing_scheme(self):
        conn = ServerConnection(base_url="http://1.2.3.4", service_port=9000, model="m")
        assert conn.url_with_port == "http://1.2.3.4:9000"

    def test_url_with_port_strips_trailing_slash(self):
        conn = ServerConnection(base_url="http://host/", service_port=8000, model="m")
        assert conn.url_with_port == "http://host:8000"

    def test_url_with_port_keeps_embedded_port(self):
        conn = ServerConnection(base_url="http://host:9000", service_port=8000, model="m")
        assert conn.url_with_port == "http://host:9000"

    def test_remote_url_preserves_full_url(self):
        conn = ServerConnection(
            base_url="https://console.tenstorrent.com/openai",
            service_port=8000,
            model="m",
            is_remote=True,
        )
        assert conn.url_with_port == "https://console.tenstorrent.com/openai"

    def test_host_strips_scheme_and_embedded_port(self):
        conn = ServerConnection(
            base_url="http://host:1234", service_port=8000, model="m"
        )
        assert conn.host == "host"

    def test_host_from_bare_hostname(self):
        conn = ServerConnection(base_url="myhost", service_port=8000, model="m")
        assert conn.host == "myhost"


class TestDriverContext:
    def test_independent_extra_env_per_instance(self):
        a = DriverContext(output_dir=Path("/a"))
        b = DriverContext(output_dir=Path("/b"))
        a.extra_env["X"] = "1"
        assert b.extra_env == {}

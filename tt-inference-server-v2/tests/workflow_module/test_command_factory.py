# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.command_factory`` arg-translation helpers.

Covers the pure, side-effect-light helpers that turn argparse Namespaces
into orchestrator options; the model-spec-dependent ``_build_context`` is
out of scope here.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from workflow_module import command_factory as cf
from workflow_module.execution import PrefixCacheOptions, ServingBenchOptions


class TestResolveServerMode:
    def test_falls_back_to_cli_flag_without_runtime_config(self):
        assert cf._resolve_server_mode(Namespace(docker_server=True), None) == "docker"
        assert cf._resolve_server_mode(Namespace(docker_server=False), None) == "API"

    def test_runtime_config_takes_precedence(self):
        rc = Namespace(docker_server=True)
        # args say API, runtime_config says docker -> runtime_config wins.
        assert cf._resolve_server_mode(Namespace(docker_server=False), rc) == "docker"


class TestResolveRunCommand:
    def test_prefixes_python_and_quotes_from_argv(self, monkeypatch):
        monkeypatch.delenv(cf._V1_RUN_COMMAND_ENV, raising=False)
        monkeypatch.setattr(cf.sys, "argv", ["run.py", "--model", "my model"])
        out = cf._resolve_run_command()
        assert out == "python run.py --model 'my model'"

    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv(cf._V1_RUN_COMMAND_ENV, "python run.py --from env")
        monkeypatch.setattr(cf.sys, "argv", ["ignored.py"])
        assert cf._resolve_run_command() == "python run.py --from env"


class TestLoadRuntimeConfig:
    def test_none_path_returns_none(self):
        assert cf._load_runtime_config(None) is None

    def test_missing_file_returns_none(self):
        assert cf._load_runtime_config("/no/such/file.json") is None


class TestServingBenchOptions:
    def test_none_for_non_serving_bench_workflow(self):
        assert cf._build_serving_bench_options(Namespace(workflow="benchmarks")) is None

    def test_built_for_serving_bench_workflow(self):
        args = Namespace(workflow="serving_bench", serving_bench_suites="a,b")
        opts = cf._build_serving_bench_options(args)
        assert isinstance(opts, ServingBenchOptions)
        assert opts.suites == "a,b"


class TestPrefixCacheOptions:
    def test_none_when_flag_absent(self):
        assert cf._build_prefix_cache_options(Namespace()) is None

    def test_none_when_flag_false(self):
        assert cf._build_prefix_cache_options(Namespace(prefix_cache=False)) is None

    def test_built_from_flags(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = Namespace(
            prefix_cache=True,
            prefix_cache_preset="full",
            prefix_cache_scenarios="s1",
            prefix_cache_arrival="poisson",
            prefix_cache_request_rate=4.0,
            prefix_cache_scenarios_json=None,
            prefix_cache_trace=None,
            jwt_secret=None,
        )
        opts = cf._build_prefix_cache_options(args)
        assert isinstance(opts, PrefixCacheOptions)
        assert opts.preset == "full"
        assert opts.scenarios == "s1"
        assert opts.arrival_pattern == "poisson"
        assert opts.request_rate == 4.0
        assert opts.auth_token == ""  # no secret -> auth disabled


class TestMintJwt:
    def test_no_secret_returns_empty(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cf._mint_jwt_if_secret(None) == ""

    def test_literal_api_key_used_when_no_jwt_secret(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.setenv("API_KEY", "literal-token")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cf._mint_jwt_if_secret(None) == "literal-token"
        import os

        assert os.environ["OPENAI_API_KEY"] == "literal-token"

    def test_secret_mints_token_and_exports_env(self, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("OPENAI_API_KEY", "")  # tracked for restore
        token = cf._mint_jwt_if_secret("super-secret-key-of-sufficient-length-1234")
        assert token  # non-empty JWT
        import os

        assert os.environ["OPENAI_API_KEY"] == token

    def test_secret_read_from_env_when_arg_absent(self, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("JWT_SECRET", "env-secret-key-of-sufficient-length-12345")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert cf._mint_jwt_if_secret(None)

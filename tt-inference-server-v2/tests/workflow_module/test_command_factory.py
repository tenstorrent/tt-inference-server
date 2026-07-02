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
from types import SimpleNamespace

import pytest

from workflows.workflow_types import InferenceEngine

from workflow_module import command_factory as cf
from workflow_module.execution import (
    LLMBenchOptions,
    LLMEvalOptions,
    PrefixCacheOptions,
    ServingBenchOptions,
)


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


class TestLLMEvalOptions:
    def test_none_for_non_eval_workflow(self):
        assert cf._build_llm_eval_options(Namespace(workflow="benchmarks")) is None

    def test_built_for_evals(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        opts = cf._build_llm_eval_options(Namespace(workflow="evals", jwt_secret=None))
        assert isinstance(opts, LLMEvalOptions)
        assert opts.auth_token == ""

    def test_built_for_release_threads_minted_token(self, monkeypatch):
        monkeypatch.setattr(cf, "_mint_jwt_if_secret", lambda secret: f"tok:{secret}")
        opts = cf._build_llm_eval_options(
            Namespace(workflow="release", jwt_secret="sek")
        )
        assert isinstance(opts, LLMEvalOptions)
        assert opts.auth_token == "tok:sek"


class TestLLMBenchOptions:
    def test_none_for_non_bench_workflow(self):
        assert cf._build_llm_bench_options(Namespace(workflow="evals")) is None

    def test_benchmarks_has_no_venv_python(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        opts = cf._build_llm_bench_options(
            Namespace(
                workflow="benchmarks",
                tools="aiperf",
                jwt_secret=None,
                prefix_cache=False,
                spec_decode=False,
            )
        )
        assert isinstance(opts, LLMBenchOptions)
        assert opts.tools == "aiperf"
        assert opts.venv_python is None

    def test_release_pins_tool_venv_python(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        opts = cf._build_llm_bench_options(
            Namespace(
                workflow="release",
                tools=None,
                jwt_secret=None,
                prefix_cache=False,
                spec_decode=False,
            )
        )
        assert opts.tools == "vllm"
        assert opts.venv_python is not None
        assert "vllm" in opts.venv_python.lower()

    def test_prefix_cache_defers(self):
        args = Namespace(workflow="benchmarks", prefix_cache=True, spec_decode=False)
        assert cf._build_llm_bench_options(args) is None


class TestMintJwt:
    def test_no_secret_returns_empty(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        assert cf._mint_jwt_if_secret(None) == ""

    def test_secret_mints_token_and_exports_env(self, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        token = cf._mint_jwt_if_secret("super-secret-key-of-sufficient-length-1234")
        assert token
        import os

        assert os.environ["OPENAI_API_KEY"] == token

    def test_secret_read_from_env_when_arg_absent(self, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("JWT_SECRET", "env-secret-key-of-sufficient-length-12345")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert cf._mint_jwt_if_secret(None)


class TestResolveAuthToken:
    """Engine-aware bearer-token selection.

    Forge/media servers (tt-media-server) validate a *literal* ``Bearer
    $API_KEY``; only the vLLM (tt-metal) server decodes a JWT. Sending a
    minted JWT to a forge/media server is what caused the 401 storm this
    fix addresses.
    """

    def _args(self, **kw):
        base = dict(model="m", device="p150", jwt_secret=None)
        base.update(kw)
        return Namespace(**base)

    def _patch_engine(self, monkeypatch, engine):
        monkeypatch.setattr(
            cf,
            "get_runtime_model_spec",
            lambda model, device: (
                SimpleNamespace(inference_engine=engine),
                None,
                None,
            ),
        )

    def test_forge_uses_literal_default_not_jwt(self, monkeypatch):
        # Even with JWT_SECRET set, a forge server must get the literal key.
        monkeypatch.setenv("JWT_SECRET", "secret-of-sufficient-length-123456")
        for var in ("VLLM_API_KEY", "API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        self._patch_engine(monkeypatch, InferenceEngine.FORGE)
        assert cf._resolve_auth_token(self._args()) == "your-secret-key"

    def test_forge_accepts_raw_string_engine(self, monkeypatch):
        # model_spec may carry the enum's str value rather than the enum.
        for var in ("VLLM_API_KEY", "API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        self._patch_engine(monkeypatch, InferenceEngine.FORGE.value)
        assert cf._resolve_auth_token(self._args()) == "your-secret-key"

    def test_media_prefers_explicit_vllm_api_key(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET", "secret-of-sufficient-length-123456")
        monkeypatch.setenv("VLLM_API_KEY", "explicit-key")
        self._patch_engine(monkeypatch, InferenceEngine.MEDIA)
        assert cf._resolve_auth_token(self._args()) == "explicit-key"

    def test_vllm_engine_mints_jwt(self, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("JWT_SECRET", "secret-of-sufficient-length-123456")
        self._patch_engine(monkeypatch, InferenceEngine.VLLM)
        token = cf._resolve_auth_token(self._args())
        assert token != "your-secret-key"
        assert token.count(".") == 2  # header.payload.signature

    def test_vllm_engine_without_secret_returns_empty(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        self._patch_engine(monkeypatch, InferenceEngine.VLLM)
        assert cf._resolve_auth_token(self._args()) == ""

    def test_unresolvable_spec_falls_back_to_jwt_path(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)

        def boom(model, device):
            raise RuntimeError("no spec")

        monkeypatch.setattr(cf, "get_runtime_model_spec", boom)
        assert cf._resolve_auth_token(self._args()) == ""

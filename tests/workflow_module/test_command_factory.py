# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.command_factory`` arg-translation helpers.

Covers the pure, side-effect-light helpers that turn argparse Namespaces
into orchestrator options; the model-spec-dependent ``_build_context`` is
out of scope here.
"""

from __future__ import annotations

import json
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
    SpecDecodeOptions,
)


class TestServerLaunchPrepend:
    """``CommandFactory.build`` prepends a ServerCommand only when a launch
    spec is supplied; the workflow-only list is otherwise unchanged. The
    model-spec-dependent ``_workflow_commands`` is stubbed out here.
    """

    def test_no_server_launch_leaves_workflow_only_list(self, monkeypatch):
        wf = object()
        monkeypatch.setattr(
            cf.CommandFactory,
            "_workflow_commands",
            staticmethod(lambda args: [wf]),
        )
        assert cf.CommandFactory.build(Namespace()) == [wf]

    def test_server_launch_prepends_server_command(self, monkeypatch):
        from workflow_module.commands import ServerCommand, ServerLaunchSpec

        wf = object()
        monkeypatch.setattr(
            cf.CommandFactory,
            "_workflow_commands",
            staticmethod(lambda args: [wf]),
        )
        spec = ServerLaunchSpec(
            mode="docker",
            model_spec=None,
            runtime_config=None,
            setup_config=None,
        )
        out = cf.CommandFactory.build(Namespace(), server_launch=spec)
        assert len(out) == 2
        assert isinstance(out[0], ServerCommand)
        assert out[0].launch is spec
        assert out[1] is wf


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
            prefix_cache_metrics_url=["worker-a:9000", "worker-b:9000/metrics"],
            jwt_secret=None,
        )
        opts = cf._build_prefix_cache_options(args)
        assert isinstance(opts, PrefixCacheOptions)
        assert opts.preset == "full"
        assert opts.scenarios == "s1"
        assert opts.arrival_pattern == "poisson"
        assert opts.request_rate == 4.0
        assert opts.auth_token == ""  # no secret -> auth disabled
        # Repeatable --prefix-cache-metrics-url -> tuple, forwarded verbatim
        # (normalization happens later in the driver).
        assert opts.metrics_urls == ("worker-a:9000", "worker-b:9000/metrics")

    def test_metrics_urls_default_empty_when_flag_absent(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # No prefix_cache_metrics_url attr at all (image-model entry path).
        args = Namespace(
            prefix_cache=True,
            prefix_cache_preset="ci",
            prefix_cache_scenarios=None,
            prefix_cache_arrival=None,
            prefix_cache_request_rate=None,
            prefix_cache_scenarios_json=None,
            prefix_cache_trace=None,
            jwt_secret=None,
        )
        opts = cf._build_prefix_cache_options(args)
        assert opts is not None
        assert opts.metrics_urls == ()

    def test_release_pins_tool_venv_python(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = Namespace(
            workflow="release",
            prefix_cache=True,
            prefix_cache_preset="ci",
            prefix_cache_scenarios=None,
            prefix_cache_arrival=None,
            prefix_cache_request_rate=None,
            prefix_cache_scenarios_json=None,
            prefix_cache_trace=None,
            jwt_secret=None,
        )
        opts = cf._build_prefix_cache_options(args)
        assert isinstance(opts, PrefixCacheOptions)
        assert opts.venv_python is not None
        assert "prefix" in opts.venv_python.lower()


class TestSpecDecodeOptions:
    def test_none_when_flag_absent(self):
        assert cf._build_spec_decode_options(Namespace()) is None

    def test_built_from_flags(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = Namespace(
            spec_decode=True,
            spec_decode_preset="ci",
            spec_decode_warmup_requests=2,
            jwt_secret=None,
        )
        opts = cf._build_spec_decode_options(args)
        assert isinstance(opts, SpecDecodeOptions)
        assert opts.preset == "ci"
        assert opts.warmup_requests == 2
        assert opts.auth_token == ""

    def test_release_pins_tool_venv_python(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        args = Namespace(
            workflow="release",
            spec_decode=True,
            spec_decode_preset="ci",
            spec_decode_warmup_requests=4,
            jwt_secret=None,
        )
        opts = cf._build_spec_decode_options(args)
        assert isinstance(opts, SpecDecodeOptions)
        assert opts.venv_python is not None
        assert "spec" in opts.venv_python.lower()


class TestLLMEvalOptions:
    def test_none_for_non_eval_workflow(self):
        assert cf._build_llm_eval_options(Namespace(workflow="benchmarks")) is None

    def test_built_for_evals(self, monkeypatch):
        # _mint_jwt_if_secret falls back to literal API_KEY/OPENAI_API_KEY for
        # remote console endpoints, so clear them to assert the no-secret case.
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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

    def test_release_with_prefix_cache_still_builds_bench_options(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        opts = cf._build_llm_bench_options(
            Namespace(
                workflow="release",
                tools=None,
                jwt_secret=None,
                prefix_cache=True,
                spec_decode=False,
            )
        )
        assert isinstance(opts, LLMBenchOptions)
        assert opts.tools == "vllm"
        assert opts.venv_python is not None

    def test_release_with_spec_decode_still_builds_bench_options(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        opts = cf._build_llm_bench_options(
            Namespace(
                workflow="release",
                tools=None,
                jwt_secret=None,
                prefix_cache=False,
                spec_decode=True,
            )
        )
        assert isinstance(opts, LLMBenchOptions)
        assert opts.tools == "vllm"
        assert opts.venv_python is not None


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
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        self._patch_engine(monkeypatch, InferenceEngine.VLLM)
        assert cf._resolve_auth_token(self._args()) == ""

    def test_unresolvable_spec_falls_back_to_jwt_path(self, monkeypatch):
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        def boom(model, device):
            raise RuntimeError("no spec")

        monkeypatch.setattr(cf, "get_runtime_model_spec", boom)
        assert cf._resolve_auth_token(self._args()) == ""

    # --- runtime_model_spec_json precedence (dual-catalog models) -------------

    def _spec_json(self, tmp_path, engine_name):
        p = tmp_path / "runtime_model_spec.json"
        p.write_text(
            json.dumps({"runtime_model_spec": {"inference_engine": engine_name}})
        )
        return str(p)

    def test_runtime_spec_json_forge_overrides_catalog_default(
        self, tmp_path, monkeypatch
    ):
        # Dual-catalog model (e.g. Llama-3.1-8B-Instruct): the catalog default
        # resolves vLLM, but the runtime spec JSON v1 handed us says forge — the
        # forge server needs the literal key, not a JWT. The JSON serializes the
        # enum *value* ("forge"), which is what v1 actually writes.
        monkeypatch.setenv("JWT_SECRET", "secret-of-sufficient-length-123456")
        for var in ("VLLM_API_KEY", "API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        self._patch_engine(monkeypatch, InferenceEngine.VLLM)  # catalog = wrong
        args = self._args(runtime_model_spec_json=self._spec_json(tmp_path, "forge"))
        assert cf._resolve_auth_token(args) == "your-secret-key"

    def test_runtime_spec_json_vllm_mints_jwt_over_catalog(self, tmp_path, monkeypatch):
        pytest.importorskip("jwt")
        monkeypatch.setenv("JWT_SECRET", "secret-of-sufficient-length-123456")
        self._patch_engine(monkeypatch, InferenceEngine.FORGE)  # catalog = wrong
        args = self._args(runtime_model_spec_json=self._spec_json(tmp_path, "vLLM"))
        assert cf._resolve_auth_token(args).count(".") == 2  # a JWT

    def test_engine_from_runtime_spec_json(self, tmp_path):
        # Real serialization is the enum value ("forge"); tolerate the name form.
        assert (
            cf._engine_from_runtime_spec_json(self._spec_json(tmp_path, "forge"))
            == "forge"
        )
        assert (
            cf._engine_from_runtime_spec_json(self._spec_json(tmp_path, "FORGE"))
            == "forge"
        )
        assert (
            cf._engine_from_runtime_spec_json(self._spec_json(tmp_path, "vLLM"))
            == "vLLM"
        )
        assert cf._engine_from_runtime_spec_json(None) is None
        assert cf._engine_from_runtime_spec_json("/no/such/file.json") is None

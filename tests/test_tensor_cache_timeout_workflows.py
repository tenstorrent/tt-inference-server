#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

import benchmarking.run_benchmarks as run_benchmarks_module
import evals.run_evals as run_evals_module
from workflows.workflow_types import DeviceTypes, ModelType


class _FakeCacheMonitor:
    def __init__(self, timeout):
        self.timeout = timeout
        self.get_tensor_cache_timeout_calls = 0

    def get_tensor_cache_timeout(self):
        self.get_tensor_cache_timeout_calls += 1
        return self.timeout


class _FakePromptClient:
    created_instances = []

    def __init__(self, env_config, model_spec=None):
        self.env_config = env_config
        self.model_spec = model_spec
        self.cache_monitor = _FakeCacheMonitor(
            model_spec.device_model_spec.tensor_cache_timeout
        )
        self.wait_for_healthy_calls = 0
        self.capture_traces_calls = 0
        self.get_health_calls = 0
        self.__class__.created_instances.append(self)

    def wait_for_healthy(self):
        self.wait_for_healthy_calls += 1
        return True

    def capture_traces(self, *args, **kwargs):
        self.capture_traces_calls += 1

    def get_health(self):
        self.get_health_calls += 1
        return SimpleNamespace(status_code=200)


def _build_model_spec():
    return SimpleNamespace(
        model_name="TestModel-7B",
        model_id="id_tt-transformers_TestModel-7B_n150",
        hf_model_repo="test/TestModel-7B",
        model_type=ModelType.LLM,
        device_type=DeviceTypes.N150,
        supported_modalities=["text"],
        has_builtin_warmup=False,
        device_model_spec=SimpleNamespace(
            tensor_cache_timeout=3600.0,
            max_context=4096,
            max_tokens_all_users=4096,
            max_concurrency=1,
        ),
    )


def test_run_evals_uses_prompt_client_with_model_spec_tensor_cache_timeout(
    monkeypatch, tmp_path
):
    _FakePromptClient.created_instances.clear()
    model_spec = _build_model_spec()
    runtime_config = SimpleNamespace(
        device="n150",
        disable_trace_capture=True,
        service_port="8000",
    )
    args = SimpleNamespace(
        runtime_model_spec_json=str(tmp_path / "runtime.json"),
        output_path=str(tmp_path),
        jwt_secret="",
        hf_token="",
    )

    monkeypatch.setattr(run_evals_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_evals_module.ModelSpec, "from_json", lambda _path: model_spec
    )
    monkeypatch.setattr(
        run_evals_module.RuntimeConfig, "from_json", lambda _path: runtime_config
    )
    monkeypatch.setattr(run_evals_module, "PromptClient", _FakePromptClient)
    monkeypatch.setattr(
        run_evals_module,
        "EVAL_CONFIGS",
        {model_spec.model_name: SimpleNamespace(tasks=[])},
    )
    monkeypatch.setattr(
        run_evals_module, "setup_workflow_script_logger", lambda _logger: None
    )

    assert run_evals_module.main() == 0
    prompt_client = _FakePromptClient.created_instances[0]
    assert prompt_client.model_spec is model_spec
    assert prompt_client.cache_monitor.get_tensor_cache_timeout_calls == 1
    assert prompt_client.wait_for_healthy_calls == 1


def test_run_benchmarks_uses_prompt_client_with_model_spec_tensor_cache_timeout(
    monkeypatch, tmp_path
):
    _FakePromptClient.created_instances.clear()
    model_spec = _build_model_spec()
    runtime_config = SimpleNamespace(
        device="n150",
        disable_trace_capture=True,
        service_port="8000",
        tools=None,
        concurrency_sweeps=False,
    )
    args = SimpleNamespace(
        runtime_model_spec_json=str(tmp_path / "runtime.json"),
        output_path=str(tmp_path),
        jwt_secret="",
        hf_token="",
        concurrency_sweeps=False,
    )
    benchmark_param = SimpleNamespace(
        task_type="text",
        isl=32,
        osl=4,
        max_concurrency=1,
        num_prompts=1,
    )
    workflow_venv_type = next(iter(run_benchmarks_module.VENV_CONFIGS))
    benchmark_task = SimpleNamespace(
        workflow_venv_type=workflow_venv_type,
        param_map={DeviceTypes.N150: [benchmark_param]},
    )
    benchmark_config = SimpleNamespace(tasks=[benchmark_task])

    monkeypatch.setattr(run_benchmarks_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_benchmarks_module.ModelSpec, "from_json", lambda _path: model_spec
    )
    monkeypatch.setattr(
        run_benchmarks_module.RuntimeConfig, "from_json", lambda _path: runtime_config
    )
    monkeypatch.setattr(run_benchmarks_module, "PromptClient", _FakePromptClient)
    monkeypatch.setattr(
        run_benchmarks_module,
        "BENCHMARK_CONFIGS",
        {model_spec.model_id: benchmark_config},
    )
    monkeypatch.setattr(
        run_benchmarks_module, "setup_workflow_script_logger", lambda _logger: None
    )
    monkeypatch.setattr(run_benchmarks_module, "run_command", lambda **kwargs: 0)
    monkeypatch.setattr(run_benchmarks_module.time, "sleep", lambda _seconds: None)

    assert run_benchmarks_module.main() == 0
    prompt_client = _FakePromptClient.created_instances[0]
    assert prompt_client.model_spec is model_spec
    assert prompt_client.cache_monitor.get_tensor_cache_timeout_calls == 1
    assert prompt_client.wait_for_healthy_calls == 1

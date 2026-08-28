from types import SimpleNamespace

import pytest

from workflows.model_spec import get_runtime_model_spec, load_templates_from_yaml
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.utils import get_repo_root_path


MODELS = {
    "Qwen3.6-27B": (8192, "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "gemma-4-31B-it": (4096, "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "gpt-oss-120b": (1024, "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
}


def _dev_quetzal_spec(model):
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    template = next(
        item
        for item in templates
        if item.impl.impl_id == "quetzal"
        and item.weights
        == [
            {
                "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
                "gemma-4-31B-it": "google/gemma-4-31B-it",
                "gpt-oss-120b": "openai/gpt-oss-120b",
            }[model]
        ]
    )
    return template.expand_to_specs()[0]


@pytest.mark.parametrize("model", MODELS)
def test_dev_quetzal_specs_are_nondefault_and_revision_pinned(model):
    spec = _dev_quetzal_spec(model)
    max_context, revision = MODELS[model]
    assert spec.impl.impl_name == "quetzal"
    assert spec.inference_engine == "vLLM"
    assert spec.device_model_spec.default_impl is False
    assert spec.device_model_spec.max_concurrency == 1
    assert spec.device_model_spec.max_context == max_context
    assert spec.device_model_spec.vllm_args["revision"] == revision
    assert spec.device_model_spec.vllm_args["tokenizer_revision"] == revision
    assert spec.env_vars["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert spec.env_vars["TT_VLLM_BUILTIN_MODELS"] == "0"

    native, native_impl, _ = get_runtime_model_spec(model=model, device="p300x2")
    assert native_impl != "quetzal"
    assert native.device_model_spec.default_impl is True


def test_docker_command_mounts_quetzal_root_readonly_and_forwards_impl(tmp_path):
    root = tmp_path / "sha256-artifact-root"
    root.mkdir()
    (root / "qualification_manifest.yaml").write_text("models: []\n")
    spec = _dev_quetzal_spec("Qwen3.6-27B")
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        quetzal_models_root=str(root),
    )

    command, _ = generate_docker_run_command(spec, runtime)

    mount = (
        f"type=bind,src={root.resolve()},"
        "dst=/home/container_app_user/quetzal_models,readonly"
    )
    assert mount in command
    assert command.count("--impl") == 1
    assert command[command.index("--impl") + 1] == "quetzal"
    assert "QZ_MODELS_ROOT=/home/container_app_user/quetzal_models" in command
    assert "VLLM_PLUGINS=quetzal_model_registry,tt" in command
    assert "TT_VLLM_BUILTIN_MODELS=0" in command


def test_docker_command_rejects_missing_or_misapplied_quetzal_root(tmp_path):
    qz_spec = _dev_quetzal_spec("Qwen3.6-27B")
    qz_runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
    )
    with pytest.raises(ValueError, match="requires --quetzal-models-root"):
        generate_docker_run_command(qz_spec, qz_runtime)

    native_spec, native_impl, engine = get_runtime_model_spec(
        model="Qwen3.6-27B", device="p300x2"
    )
    native_runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl=native_impl,
        engine=engine,
        quetzal_models_root=str(tmp_path),
    )
    with pytest.raises(ValueError, match="only valid with --impl quetzal"):
        generate_docker_run_command(native_spec, native_runtime)


def test_quetzal_docker_hook_builds_one_wheel_from_commit_only():
    dockerfile = (
        get_repo_root_path() / "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile"
    ).read_text()
    assert 'ARG TT_QUETZAL_COMMIT_SHA=""' in dockerfile
    assert "^[0-9a-f]{40}$" in dockerfile
    assert "uv build --wheel" in dockerfile
    assert 'uv pip install --no-cache-dir --no-deps "${quetzal_wheel}"' in dockerfile
    assert (
        "uv pip install --no-cache-dir --no-deps --no-build-isolation ."
        not in dockerfile
    )
    assert "TT_QUETZAL_COMMIT_SHA_OR_TAG" not in dockerfile

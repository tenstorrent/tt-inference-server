# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from workflows.helm_generator.cli import (
    assert_single_default_impl,
    assert_unique,
    compute_default_engine_per_model,
    filter_specs,
    generate,
)
from workflows.helm_generator.yaml_io import load_values


def test_inserts_new_model_and_idempotent_rerun(tmp_path, fixtures_dir, vllm_spec):
    work = tmp_path / "values.yaml"
    work.write_text((fixtures_dir / "test_values.yaml").read_text())

    changed = generate(values_path=work, specs=[vllm_spec])
    assert changed == 1

    doc = load_values(work)
    galaxy = doc["models"]["Llama-3.1-8B-Instruct"]["vllm"]["galaxy"]
    assert galaxy["defaultImpl"] == "test_impl"
    assert galaxy["impls"]["test_impl"]["image"]["tag"] == "0.11.1-bac8b34"
    assert doc["models"]["Llama-3.1-8B-Instruct"]["defaultEngine"] == "vllm"

    after_first = work.read_text()
    changed_again = generate(values_path=work, specs=[vllm_spec])
    assert changed_again == 0
    assert work.read_text() == after_first


def test_multi_impl_same_device(tmp_path, fixtures_dir, sample_impl):
    """Two impls under one (model, engine, device) -> one defaultImpl chosen."""
    from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpecTemplate
    from workflows.workflow_types import DeviceTypes, InferenceEngine

    work = tmp_path / "values.yaml"
    work.write_text((fixtures_dir / "test_values.yaml").read_text())

    other = ImplSpec(
        impl_id="other_impl",
        impl_name="other",
        repo_url="https://example.com",
        code_path="x",
    )

    def make(impl, default_impl: bool):
        return ModelSpecTemplate(
            weights=["acme/MultiImpl-7B"],
            impl=impl,
            tt_metal_commit="aaa",
            vllm_commit="bbb",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.GALAXY,
                    max_concurrency=1,
                    max_context=1024,
                    default_impl=default_impl,
                )
            ],
            docker_image=f"ghcr.io/acme/img-{impl.impl_id}:1.0",
        )

    a = make(sample_impl, default_impl=True).expand_to_specs()[0]
    b = make(other, default_impl=False).expand_to_specs()[0]
    changed = generate(values_path=work, specs=[a, b])
    assert changed == 2

    doc = load_values(work)
    galaxy = doc["models"]["MultiImpl-7B"]["vllm"]["galaxy"]
    assert galaxy["defaultImpl"] == "test_impl"
    assert set(galaxy["impls"].keys()) == {"test_impl", "other_impl"}


def test_multi_engine_same_model(tmp_path, fixtures_dir, sample_impl):
    """Two engines on the same model -> defaultEngine picked by precedence."""
    from workflows.model_spec import DeviceModelSpec, ModelSpecTemplate
    from workflows.workflow_types import DeviceTypes, InferenceEngine

    work = tmp_path / "values.yaml"
    work.write_text((fixtures_dir / "test_values.yaml").read_text())

    def make(engine):
        return ModelSpecTemplate(
            weights=["acme/MultiEngine-7B"],
            impl=sample_impl,
            tt_metal_commit="aaa",
            vllm_commit="bbb" if engine == InferenceEngine.VLLM.value else None,
            inference_engine=engine,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.GALAXY,
                    max_concurrency=1,
                    max_context=1024,
                    default_impl=True,
                )
            ],
            docker_image=f"ghcr.io/acme/img-{engine}:1.0",
            min_ram_gb=6,
        )

    a = make(InferenceEngine.MEDIA.value).expand_to_specs()[0]
    b = make(InferenceEngine.VLLM.value).expand_to_specs()[0]
    generate(values_path=work, specs=[a, b])

    doc = load_values(work)
    entry = doc["models"]["MultiEngine-7B"]
    assert entry["defaultEngine"] == "vllm"  # precedence wins
    assert set(entry.keys()) >= {"vllm", "media", "defaultEngine"}


def test_dry_run_does_not_modify_file(tmp_path, fixtures_dir, vllm_spec):
    work = tmp_path / "values.yaml"
    work.write_text((fixtures_dir / "test_values.yaml").read_text())
    before = work.read_text()

    changed = generate(values_path=work, specs=[vllm_spec], dry_run=True)
    assert changed == 1
    assert work.read_text() == before


def test_real_model_specs_passes_uniqueness_and_default_checks():
    """Auditing the real catalog: (model, device, engine, impl) is unique and
    each (model, device, engine) has exactly one default_impl=True."""
    from workflows.model_spec import MODEL_SPECS

    specs = filter_specs(MODEL_SPECS.values(), include_multihost=False)
    assert_unique(specs)
    assert_single_default_impl(specs)
    defaults = compute_default_engine_per_model(specs)
    assert all(v in ("vllm", "media", "forge") for v in defaults.values())

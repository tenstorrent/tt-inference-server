# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev.render_gpt120_f0b_behavioral_runtime_spec import (
    AUXILIARY_NAME,
    CHECKPOINT,
    EXPECTED_FILES,
    MODEL_ID,
    QUETZAL_COMMIT,
    ContractError,
    render,
)
from workflows.model_spec import ModelSpec


def _candidate(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "gpt120-multiprefill-s128-s1024-c8192-f0b"
    for relative in EXPECTED_FILES.values():
        member = package / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(b"fixture")
    auxiliary = tmp_path / "gpt120-b5c939de"
    auxiliary.mkdir()
    return package, auxiliary


def test_behavioral_spec_is_normal_quetzal_model_spec_but_never_certifiable(tmp_path):
    package, auxiliary = _candidate(tmp_path)
    document = render(package, auxiliary)
    spec = document["runtime_model_spec"]
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(document))
    parsed = ModelSpec.from_json(path)

    assert document["official_models_ci"] is False
    assert document["certification_eligible"] is False
    assert document["package_trust_expected"] == "fail_closed"
    assert parsed.impl.impl_id == "quetzal"
    assert parsed.hf_model_repo == MODEL_ID
    assert parsed.device_model_spec.max_context == 8192
    assert parsed.device_model_spec.max_concurrency == 1
    assert parsed.device_model_spec.default_impl is False
    env = parsed.env_vars
    assert env["QUETZAL_REQUIRED_SOURCE_REVISION"] == QUETZAL_COMMIT
    assert env["QUETZAL_HF_REVISION"] == CHECKPOINT
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["QUETZAL_REQUIRED_AUXILIARY_NAMES"] == AUXILIARY_NAME
    assert env["QUETZAL_BUNDLE_MANIFEST_SHA256"] == "0" * 64
    assert parsed.metadata["local_shadow"] is True


def test_behavioral_spec_refuses_missing_exact_f0b_members(tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    auxiliary = tmp_path / "auxiliary"
    auxiliary.mkdir()
    with pytest.raises(ContractError, match="exact f0b member is missing"):
        render(package, auxiliary)


def test_behavioral_spec_refuses_official_namespace(monkeypatch, tmp_path):
    package, auxiliary = _candidate(tmp_path)
    module = __import__(
        "scripts.dev.render_gpt120_f0b_behavioral_runtime_spec",
        fromlist=["OFFICIAL_PACKAGE_PREFIX"],
    )
    monkeypatch.setattr(module, "OFFICIAL_PACKAGE_PREFIX", tmp_path)
    with pytest.raises(ContractError, match="refuses the official"):
        render(package, auxiliary)


def test_document_round_trips_through_runtime_model_spec_json(tmp_path):
    package, auxiliary = _candidate(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(render(package, auxiliary)))
    parsed = ModelSpec.from_json(path)
    assert parsed.model_name == "gpt-oss-120b"
    assert parsed.impl.impl_name == "quetzal"

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import io
import json
import zipfile

import pytest

from scripts.release.release_scope import (
    extract_bundle_identity,
    identity_from_model_spec_document,
)


IDENTITY = ("org/model", "N150", "vLLM", "tt_transformers")


def _model_spec():
    return {
        "hf_model_repo": IDENTITY[0],
        "device_type": IDENTITY[1],
        "inference_engine": IDENTITY[2],
        "impl": {"impl_id": IDENTITY[3], "impl_name": "tt-transformers"},
    }


def _bundle_bytes(document=None, *, path="runtime_model_specs/spec.json"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if document is not None:
            archive.writestr(path, json.dumps(document))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "document",
    [
        {"runtime_model_spec": _model_spec(), "runtime_config": {}},
        {"model_spec": _model_spec()},
        _model_spec(),
    ],
)
def test_identity_from_modern_and_legacy_runtime_specs(document):
    assert identity_from_model_spec_document(document) == IDENTITY


@pytest.mark.parametrize(
    "field,value",
    [
        ("hf_model_repo", None),
        ("hf_model_repo", "   "),
        ("device_type", 123),
        ("inference_engine", None),
    ],
)
def test_runtime_identity_rejects_non_string_or_empty_fields(field, value):
    document = _model_spec()
    document[field] = value
    with pytest.raises(ValueError, match="invalid exact identity fields"):
        identity_from_model_spec_document(document)

    document = _model_spec()
    document["impl"]["impl_id"] = value
    with pytest.raises(ValueError, match="invalid exact identity fields"):
        identity_from_model_spec_document(document)


def test_extract_bundle_identity_accepts_legacy_run_specs():
    bundle = _bundle_bytes({"model_spec": _model_spec()}, path="run_specs/old.json")

    assert extract_bundle_identity(bundle) == IDENTITY


def test_extract_bundle_identity_rejects_missing_and_conflicting_specs():
    with pytest.raises(ValueError, match="no runtime model spec"):
        extract_bundle_identity(_bundle_bytes())

    other = {**_model_spec(), "hf_model_repo": "org/other"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("runtime_model_specs/a.json", json.dumps(_model_spec()))
        archive.writestr("runtime_model_specs/b.json", json.dumps(other))

    with pytest.raises(ValueError, match="conflicting runtime identities"):
        extract_bundle_identity(buffer.getvalue())


def test_extract_bundle_identity_rejects_duplicate_runtime_spec_path():
    buffer = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "runtime_model_specs/spec.json",
                json.dumps(_model_spec()),
            )
            archive.writestr(
                "runtime_model_specs/spec.json",
                json.dumps({**_model_spec(), "hf_model_repo": "org/other"}),
            )

    with pytest.raises(ValueError, match="duplicate runtime model spec path"):
        extract_bundle_identity(buffer.getvalue())

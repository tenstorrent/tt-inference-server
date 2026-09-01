from pathlib import Path

import pytest

from utils.prompt_client import resolve_trace_tokenizer_model


def test_trace_tokenizer_uses_admitted_local_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(snapshot))

    assert resolve_trace_tokenizer_model("org/public-model") == str(snapshot.resolve())


def test_trace_tokenizer_keeps_public_identity_without_local_boundary(monkeypatch):
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)

    assert resolve_trace_tokenizer_model("org/public-model") == "org/public-model"


@pytest.mark.parametrize("value", ["relative/model", "/missing/model-snapshot"])
def test_trace_tokenizer_local_boundary_fails_closed(monkeypatch, value):
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", value)

    with pytest.raises(RuntimeError, match="MODEL_WEIGHTS_DIR"):
        resolve_trace_tokenizer_model("org/public-model")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import hashlib
import json
from types import SimpleNamespace

import pytest

from llm_module.config import LLMRunConfig
from llm_module.fixed_workload_protocol import (
    resolve_tokenizer,
    validate_fixed_workload,
)


def valid_raw():
    return {
        "completed": 2,
        "failed": 0,
        "num_prompts": 2,
        "max_concurrency": 1,
        "input_lens": [128, 128],
        "output_lens": [4, 4],
        "errors": ["", ""],
        "total_input_tokens": 256,
        "total_output_tokens": 8,
        "ttfts": [0.02, 0.04],
        "itls": [[0.01, 0.01, 0.01], [0.02, 0.01]],
        "duration": 0.13,
        "mean_ttft_ms": 30.0,
        "mean_tpot_ms": 10.0,
        "output_throughput": 8 / 0.13,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"completed": 1, "failed": 1},
        {"num_prompts": 3},
        {"max_concurrency": 2},
        {"input_lens": [127, 128]},
        {"output_lens": [4, 3]},
        {"errors": ["", "error"]},
        {"total_input_tokens": 254},
        {"total_output_tokens": 7},
        {"ttfts": [0.02]},
        {"itls": [[], [0.01]]},
        {"ttfts": [float("nan"), 0.04]},
        {"duration": 0},
        {"mean_tpot_ms": 9},
        {"mean_ttft_ms": 29},
        {"output_throughput": 99},
    ],
)
def test_incomplete_or_inconsistent_evidence_fails(change):
    cfg = LLMRunConfig(128, 4, 1, 2)
    raw = valid_raw()
    validate_fixed_workload(raw, cfg)  # Multiple tokens can share one content event.
    raw.update(change)
    with pytest.raises(ValueError):
        validate_fixed_workload(raw, cfg)


def test_tokenizer_revision_and_hashes_are_enforced(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text(name)
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    revision = "a" * 40
    spec = SimpleNamespace(
        hf_model_repo="org/model",
        device_model_spec=SimpleNamespace(vllm_args={"tokenizer_revision": revision}),
        metadata={"benchmark_tokenizer_sha256": hashes},
    )
    evidence = tmp_path / "results"
    assert resolve_tokenizer(spec, evidence) == str(snapshot)
    assert calls[0]["revision"] == revision
    assert (
        json.loads((evidence / "tokenizer_identity.json").read_text())["sha256"]
        == hashes
    )
    (snapshot / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="differ from the frozen reference"):
        resolve_tokenizer(spec, evidence)
    spec.device_model_spec.vllm_args["tokenizer_revision"] = "main"
    with pytest.raises(ValueError, match="pinned tokenizer"):
        resolve_tokenizer(spec, evidence)

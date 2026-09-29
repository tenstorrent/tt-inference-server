# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared pinning and LongBench contracts must not depend on an implementation ID."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import EvalTask


def spec(revision=None, tokenizer=None, required=False, context=131072):
    args = (
        {}
        if revision is None and tokenizer is None
        else {"revision": revision, "tokenizer_revision": tokenizer}
    )
    return SimpleNamespace(
        model_id="example",
        hf_model_repo="org/example",
        metadata={"pinned_checkpoint": required},
        impl=SimpleNamespace(impl_id="another_implementation"),
        device_model_spec=SimpleNamespace(
            max_context=context, max_concurrency=32, vllm_args=args
        ),
    )


def model_args(command):
    return dict(
        s.split("=", 1) for s in command[command.index("--model_args") + 1].split(",")
    )


@pytest.mark.parametrize("context", [8192, 131072])
def test_longbench_defaults_to_device_context_for_any_implementation(tmp_path, context):
    task = EvalTask(task_name="longbench_single_e", model_kwargs={})
    command = build_eval_command(task, spec(context=context), None, tmp_path, 8000)
    assert model_args(command)["max_length"] == str(context)
    assert task.model_kwargs == {}


def test_longbench_preserves_explicit_task_context(tmp_path):
    task = EvalTask(task_name="longbench_single_e", model_kwargs={"max_length": 4096})
    command = build_eval_command(task, spec(), None, tmp_path, 8000)
    assert model_args(command)["max_length"] == "4096"


def test_command_builder_uses_prepared_tokenizer_without_network(tmp_path):
    task = EvalTask(task_name="longbench_single_e", model_kwargs={})
    with patch(
        "huggingface_hub.snapshot_download", side_effect=AssertionError("network")
    ):
        command = build_eval_command(
            task,
            spec("a" * 40, "a" * 40),
            None,
            tmp_path,
            8000,
            tokenizer_path="/verified/tokenizer",
        )
    assert model_args(command)["tokenizer"] == "/verified/tokenizer"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("as_dict", [False, True])
def test_pin_validation_supports_runtime_objects_and_container_dicts(as_dict):
    from utils.pinned_artifacts import get_pinned_revision

    candidate = spec("a" * 40, "a" * 40)
    if as_dict:
        candidate = {
            "device_model_spec": {"vllm_args": candidate.device_model_spec.vllm_args}
        }
    assert get_pinned_revision(candidate) == "a" * 40
    assert get_pinned_revision(spec()) is None


@pytest.mark.parametrize(
    "revision,tokenizer",
    [("main", "main"), (True, True), ("a" * 40, "b" * 40), (None, None)],
)
def test_required_pin_fails_closed(revision, tokenizer):
    from utils.pinned_artifacts import get_pinned_revision

    with pytest.raises(ValueError, match="revision"):
        get_pinned_revision(spec(revision, tokenizer, required=True))


def test_pinned_eval_records_tokenizer_identity_without_benchmark_hashes(tmp_path):
    from utils.pinned_artifacts import resolve_tokenizer

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (snapshot / name).write_text(name)
    candidate = spec("a" * 40, "a" * 40)
    with patch(
        "huggingface_hub.snapshot_download", return_value=str(snapshot)
    ) as download:
        assert resolve_tokenizer(candidate, tmp_path / "evidence") == str(snapshot)
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert (tmp_path / "evidence" / "tokenizer_identity.json").is_file()
    with patch(
        "huggingface_hub.snapshot_download", side_effect=AssertionError("network")
    ):
        with pytest.raises(ValueError, match="hashes"):
            resolve_tokenizer(candidate, tmp_path / "benchmark", require_hashes=True)

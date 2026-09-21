# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1] / "tt-vllm-plugin/tt_vllm_plugin"


def load_function(path, name, namespace, class_name=None):
    # Exercise the actual CPU helper without importing ttnn or a GPU vLLM build.
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(
            node
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    function = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace
    )
    return namespace[name]


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("logprobs", [None, 0, 5, -1])
def test_validation_supports_both_pinned_call_conventions(
    monkeypatch, legacy, logprobs
):
    sampling = ModuleType("vllm.sampling_params")

    class SamplingParams(SimpleNamespace):
        pass

    sampling.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    validate = load_function("platform.py", "validate_request", {}, "TTPlatform")
    platform = SimpleNamespace(device_name="tt", process_data_parallel_size=1)
    params = SamplingParams(n=1, logprobs=logprobs, prompt_logprobs=None)

    def call():
        if legacy:
            validate(platform, prompt="hello", params=params, processed_inputs={})
        else:
            validate(platform, {}, params)

    if logprobs == -1:
        with pytest.raises(ValueError, match="Full-vocabulary"):
            call()
    else:
        call()


def test_sampled_logprobs_match_vllm_ranks_and_raw_probabilities():
    torch = pytest.importorskip("torch")
    compute = load_function(
        "v1/worker/tt_model_runner.py",
        "compute_sampled_logprobs",
        {"torch": torch, "LogprobsTensors": SimpleNamespace},
    )
    logits = torch.tensor([[3.0, 2.0, 1.0], [3.0, 3.0, 1.0], [3.0, 2.0, 1.0]])
    result = compute(logits, torch.tensor([0, 0, 2]), 2)
    assert result.selected_token_ranks.tolist() == [1, 2, 3]
    assert result.logprob_token_ids[:, 0].tolist() == [0, 0, 2]
    torch.testing.assert_close(
        result.logprobs[:, 0],
        torch.log_softmax(logits, dim=-1)[torch.arange(3), torch.tensor([0, 0, 2])],
    )


@pytest.mark.parametrize("mode", ["all", "decode_only"])
@pytest.mark.parametrize("logprobs", [None, 0, 5])
def test_device_sampling_rejects_logprobs(monkeypatch, mode, logprobs):
    sampling = ModuleType("vllm.sampling_params")

    class SamplingParams(SimpleNamespace):
        pass

    sampling.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    validate = load_function("platform.py", "validate_request", {}, "TTPlatform")
    platform = SimpleNamespace(
        device_name="tt", process_data_parallel_size=1, sample_on_device_mode=mode
    )
    params = SamplingParams(n=1, logprobs=logprobs, prompt_logprobs=None)
    if logprobs is None:
        validate(platform, {}, params)
    else:
        with pytest.raises(ValueError, match="host-side sampling"):
            validate(platform, {}, params)


@pytest.mark.parametrize("release", ["preempted", "finished", "unscheduled"])
def test_replica_pins_release_only_requests_without_live_kv(release):
    execute = load_function(
        "v1/worker/tt_model_runner.py",
        "_execute_model_inprocess_dp",
        {},
        "TTModelRunner",
    )

    class GroupingComplete(Exception):
        pass

    class BlockTables:
        def __getitem__(self, index):
            # Stop before device execution, after actual replica placement.
            raise GroupingComplete

    replacing = release != "unscheduled"
    ids = ["a", "c"] if replacing else ["a"]
    runner = SimpleNamespace(
        _req_to_mesh={"a": 0, "b": 1},
        _tt_dp=2,
        _per_mesh_max_seqs=1,
        _update_states=lambda _: None,
        input_batch=SimpleNamespace(
            num_reqs=len(ids),
            req_ids=ids,
            num_computed_tokens_cpu=[1, 0],
            num_prompt_tokens=[1, 1],
            block_table=BlockTables(),
        ),
    )
    output = SimpleNamespace(
        finished_req_ids={"b"} if release == "finished" else set(),
        preempted_req_ids={"b"} if release == "preempted" else set(),
        total_num_scheduled_tokens=len(ids),
        scheduled_new_reqs=[SimpleNamespace(req_id="c")] if replacing else [],
        num_scheduled_tokens={key: 1 for key in ids},
    )
    with pytest.raises(GroupingComplete):
        execute(runner, output)
    assert runner._req_to_mesh == ({"a": 0, "c": 1} if replacing else {"a": 0, "b": 1})

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import json

import pytest

from diagnostics.longbench_context import (
    MODEL,
    REVISION,
    IMAGE,
    build_payload,
    validate_response,
    validate_runtime,
    replay,
)


def test_pair_changes_only_prompt_and_retains_async_stop():
    a = build_payload(
        [1, 2],
        {
            "max_gen_toks": 512,
            "temperature": 0,
            "do_sample": False,
            "until": [],
            "stream": False,
            "seed": 42,
        },
    )
    b = build_payload(
        [0, 1, 2],
        {
            "max_gen_toks": 512,
            "temperature": 0,
            "do_sample": False,
            "until": [],
            "stream": False,
            "seed": 42,
        },
    )
    assert a.pop("prompt") == [[1, 2]]
    assert b.pop("prompt") == [[0, 1, 2]]
    assert (
        a
        == b
        == {
            "model": MODEL,
            "max_tokens": 512,
            "temperature": 0,
            "stop": [],
            "seed": 42,
            "stream": False,
        }
    )


def response(n=2):
    return {
        "choices": [{"index": 0, "text": "answer", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["usage"].update(prompt_tokens=3),
        lambda x: x["usage"].update(completion_tokens=513),
        lambda x: x["usage"].update(total_tokens=100),
        lambda x: x["choices"][0].update(finish_reason=None),
        lambda x: x.update(choices=[]),
        lambda x: x.update(error="device failure"),
    ],
)
def test_invalid_response_fails(mutation):
    data = response()
    mutation(data)
    with pytest.raises(ValueError):
        validate_response(data, 2)


def test_good_response():
    validate_response(response(), 2)


def test_failure_is_saved_and_stops_later_requests(tmp_path):
    rows = [
        {
            "case_id": str(i),
            "arm": "original_suffix",
            "input_tokens": 2,
            "payload": {"prompt": [[1, 2]]},
        }
        for i in range(3)
    ]
    calls = []

    def transport(payload):
        calls.append(payload)
        return response(2 if len(calls) == 1 else 3)

    with pytest.raises(ValueError):
        replay(rows, tmp_path, transport)
    saved = [
        json.loads(x) for x in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    assert len(calls) == len(saved) == 2
    assert saved[-1]["response"]["usage"]["prompt_tokens"] == 3
    assert not (tmp_path / "COMPLETE.json").exists()


def runtime():
    return {
        "image": IMAGE,
        "weights_path": "/weights/snapshots/" + REVISION,
        "log": "models.demos.llama31_8b_qb2.tt.generator_vllm.LlamaForCausalLM gu4_head8_lm_head_hifi2 --max_model_len 131072 --max_num_seqs 32 --seed 0 --revision "
        + REVISION
        + " --tokenizer_revision "
        + REVISION,
    }


def test_runtime_expected():
    validate_runtime(runtime())


@pytest.mark.parametrize(
    "key,value",
    [
        ("image", "mutable:latest"),
        ("weights_path", "/weights/main"),
        ("log", "other model"),
    ],
)
def test_runtime_mismatch_fails(key, value):
    value_map = runtime()
    value_map[key] = value
    with pytest.raises(ValueError):
        validate_runtime(value_map)

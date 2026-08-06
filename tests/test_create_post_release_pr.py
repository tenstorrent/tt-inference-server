# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from scripts.release.create_post_release_pr import (
    build_rows,
    collect_release_combos,
    find_block,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine

# Two prod blocks whose basenames collide across orgs, so a basename-keyed
# lookup cannot tell them apart.
QWEN_BLOCK = {
    "impl": "tt-transformers",
    "inference_engine": "vLLM",
    "weights": ["Qwen/Qwen3-32B"],
    "device_model_specs": [{"device": "GALAXY"}, {"device": "P150"}],
    "tt_metal_commit": "aaa1111",
    "status": "ready",
}
LLAMA_BLOCK = {
    "impl": "tt-transformers",
    "inference_engine": "vLLM",
    "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
    "device_model_specs": [{"device": "GALAXY"}],
    "tt_metal_commit": "bbb2222",
    "status": "ready",
}
CNN_BLOCK = {
    "impl": "forge",
    "inference_engine": "FORGE",
    "weights": ["resnet-50"],
    "device_model_specs": [{"device": "P150"}],
    "tt_metal_commit": "ccc3333",
    "status": "ready",
}
BLOCKS = [QWEN_BLOCK, LLAMA_BLOCK, CNN_BLOCK]


def test_collect_release_combos_yields_config_keys_verbatim():
    """The combo model_name is the models-ci-config.json key, not a basename.

    find_block compares against it directly, so this is the contract the
    lookup depends on.
    """
    ci_config = {
        "models": {
            "Qwen/Qwen3-32B": {
                "inference_engine": "vLLM",
                "ci": {"release": {"devices": ["GALAXY"]}},
            }
        }
    }
    assert collect_release_combos(ci_config) == [
        ("Qwen/Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY)
    ]


def test_find_block_matches_prefixed_config_key():
    block = find_block(
        BLOCKS, "Qwen/Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY
    )
    assert block is QWEN_BLOCK


def test_find_block_matches_bare_config_key():
    """The eight Forge-CNN entries key on a bare name and their weights are
    bare too, so the same equality holds without a basename fallback."""
    block = find_block(BLOCKS, "resnet-50", InferenceEngine.FORGE, DeviceTypes.P150)
    assert block is CNN_BLOCK


def test_find_block_rejects_basename_of_a_prefixed_weight():
    assert (
        find_block(BLOCKS, "Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY)
        is None
    )


def test_build_rows_keeps_distinct_models_on_shared_engine_and_device():
    """Regression: when find_block missed, every row carried empty weights and
    the (weights, engine, device) dedup key collapsed unrelated models into one.
    """
    combos = [
        ("Qwen/Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ("meta-llama/Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
    ]
    rows = build_rows(BLOCKS, [], combos, jobs=None, tt_shield_repo="", run_id=None)

    assert [r["model_arch"] for r in rows] == [
        "Qwen/Qwen3-32B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    assert [r["weights"] for r in rows] == [
        ["Qwen/Qwen3-32B"],
        ["meta-llama/Llama-3.1-8B-Instruct"],
    ]
    assert [r["tt_after"] for r in rows] == ["aaa1111", "bbb2222"]


def test_build_rows_keeps_distinct_models_when_no_block_matches():
    """The case the test above does not reach: `find_block` misses entirely.

    Every blockless combo used to key on the same empty weights tuple, so all
    but the first on a given (engine, device) were dropped -- silently, exit 0.
    A stale weights spelling in prod, or a release cut before a spec landed, is
    enough to trigger it.
    """
    combos = [
        ("Qwen/Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ("meta-llama/Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ("openai/gpt-oss-20b", InferenceEngine.VLLM, DeviceTypes.GALAXY),
    ]
    rows = build_rows([], [], combos, jobs=None, tt_shield_repo="", run_id=None)

    assert [r["model_arch"] for r in rows] == [c[0] for c in combos]
    # No block, so nothing is known about them -- but they must still be listed.
    assert all(r["weights"] == [] for r in rows)


def test_build_rows_still_dedups_blockless_repeats():
    """The dedup itself must survive: a combo repeated verbatim is one row."""
    combo = ("Qwen/Qwen3-32B", InferenceEngine.VLLM, DeviceTypes.GALAXY)
    rows = build_rows([], [], [combo, combo], jobs=None, tt_shield_repo="", run_id=None)
    assert len(rows) == 1


def test_build_rows_dedups_a_block_bundling_several_weights():
    multi = {
        "impl": "tt-transformers",
        "inference_engine": "vLLM",
        "weights": ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo"],
        "device_model_specs": [{"device": "P150"}],
        "tt_metal_commit": "ddd4444",
        "status": "ready",
    }
    combos = [
        ("openai/whisper-large-v3", InferenceEngine.VLLM, DeviceTypes.P150),
        ("openai/whisper-large-v3-turbo", InferenceEngine.VLLM, DeviceTypes.P150),
    ]
    rows = build_rows([multi], [], combos, jobs=None, tt_shield_repo="", run_id=None)
    assert len(rows) == 1

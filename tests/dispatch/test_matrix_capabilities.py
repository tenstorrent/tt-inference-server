# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device schema/capability tests for model_matrix.toml (issues #3, #46).

Loads the matrix via the dispatcher's loader and asserts:
  - schema_version matches the dispatcher,
  - every listed entry's GQA divisibility holds,
  - behavioral fields + DERIVED capabilities (fast_path / lm_head_ondevice) match the
    expected per-model table,
  - Granite resolves its four architectural scaling multipliers (#43).

NOTE (novel-safety): every assertion is POSITIVE — "Granite resolves X",
"GQA holds for listed entries". Nothing here asserts a model must be listed to run;
the novel auto-derive path is the universal floor and is exercised on-card by the
#47 novel gate, not here.
"""

import pytest

from tt_inference_server.dispatch.dispatcher import (
    DISPATCHER_SCHEMA_VERSION,
    _DEFAULT_MATRIX_PATH,
    _load_and_validate_matrix,
)

# name -> (norm_type, activation, rotary_pct, attn_bias, parallel_residual,
#          embed_scale, expected_fast_path, expected_lm_head_ondevice)
EXPECTED = {
    "meta-llama/Llama-3-8B-Instruct":            ("rmsnorm",   "silu",      1.0,  False, False, "none",        True,  True),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":  ("rmsnorm",   "silu",      1.0,  False, False, "none",        True,  True),
    "Qwen/Qwen2.5-7B-Instruct":                  ("rmsnorm",   "silu",      1.0,  True,  False, "none",        False, True),
    "mistralai/Mistral-7B-v0.3":                 ("rmsnorm",   "silu",      1.0,  False, False, "none",        True,  True),
    "google/gemma-2-2b-it":                      ("gemma_rms", "gelu_tanh", 1.0,  False, False, "sqrt_hidden", False, True),
    "microsoft/Phi-3.5-mini-instruct":           ("rmsnorm",   "silu",      1.0,  False, False, "none",        True,  True),
    "allenai/OLMo-1B-hf":                        ("rmsnorm",   "silu",      1.0,  False, False, "none",        True,  True),
    "EleutherAI/pythia-2.8b":                    ("layernorm", "gelu",      0.25, True,  True,  "none",        False, False),
}


@pytest.fixture(scope="module")
def matrix():
    entries, index = _load_and_validate_matrix(_DEFAULT_MATRIX_PATH)
    return entries, index


def test_schema_version_loads(matrix):
    entries, _index = matrix
    assert entries, "matrix loaded zero entries"
    # _load_and_validate_matrix raises SchemaVersionError on mismatch, so reaching here
    # means the file's schema_version equals DISPATCHER_SCHEMA_VERSION.
    assert DISPATCHER_SCHEMA_VERSION >= 1


def test_gqa_divisibility_holds_for_all_listed(matrix):
    entries, _index = matrix
    for e in entries:
        assert e.n_kv_heads >= 1 and e.n_heads >= 1, e.name
        assert e.n_heads % e.n_kv_heads == 0, (
            f"{e.name}: n_heads={e.n_heads} not divisible by n_kv_heads={e.n_kv_heads}")


@pytest.mark.parametrize("name,expected", EXPECTED.items(), ids=list(EXPECTED))
def test_capabilities_match_expected(matrix, name, expected):
    _entries, index = matrix
    assert name in index, f"{name} MISSING from matrix"
    e = index[name]
    cap = e.capabilities()
    got = (e.norm_type, e.activation, e.rotary_pct, e.attn_bias,
           e.parallel_residual, e.embed_scale, cap.fast_path, cap.lm_head_ondevice)
    assert got == expected, f"{name}: expected {expected}, got {got}"


def test_granite_multipliers(matrix):
    """Granite (#43) needs four architectural scaling factors plain Llama lacks; the
    matrix must resolve them (12.0 / 0.22 / 0.015625 / 8.0)."""
    _entries, index = matrix
    name = "ibm-granite/granite-3.1-2b-instruct"
    assert name in index, f"{name} MISSING from matrix"
    e = index[name]
    assert e.embedding_multiplier == 12.0
    assert e.residual_multiplier == 0.22
    assert e.attention_multiplier == 0.015625
    assert e.logits_scaling == 8.0

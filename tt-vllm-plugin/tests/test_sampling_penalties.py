# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-reference parity tests for TS-10 host-side sampling penalties.

These exercise the pure-torch sampler utilities in
``tt_vllm_plugin.worker.tt_model_runner`` with no device / vLLM dependency, so
they run anywhere torch + transformers are installed::

    pytest tt-vllm-plugin/tests/test_sampling_penalties.py
"""

import torch

from tt_vllm_plugin.worker.tt_model_runner import (
    TTSamplingParams,
    apply_penalties,
    sample_tokens,
)


def _ref_apply_penalties(
    logits, prompt_rows, output_rows, presence, frequency, repetition
):
    """Independent reference (vLLM semantics) for apply_penalties."""
    logits = logits.clone()
    num_seqs, vocab = logits.shape
    for i in range(num_seqs):
        prompt_set = set(int(t) for t in prompt_rows[i])
        out_counts = {}
        for t in output_rows[i]:
            out_counts[int(t)] = out_counts.get(int(t), 0) + 1
        seen = prompt_set | set(out_counts.keys())
        rp = float(repetition[i])
        for tok in seen:
            if logits[i, tok] > 0:
                logits[i, tok] = logits[i, tok] / rp
            else:
                logits[i, tok] = logits[i, tok] * rp
        for tok, cnt in out_counts.items():
            logits[i, tok] -= float(frequency[i]) * cnt
            logits[i, tok] -= float(presence[i])
    return logits


def test_apply_penalties_matches_reference():
    torch.manual_seed(0)
    num_seqs, vocab = 4, 50
    logits = torch.randn(num_seqs, vocab)
    prompt_rows = [[1, 2, 3], [10], [], [5, 5, 6]]
    output_rows = [[2, 2, 4], [], [7, 8], [6]]
    presence = torch.tensor([0.5, 0.0, 1.0, 0.2])
    frequency = torch.tensor([0.3, 0.0, 0.0, 0.7])
    repetition = torch.tensor([1.2, 1.0, 1.5, 1.1])

    got = apply_penalties(
        logits, prompt_rows, output_rows, presence, frequency, repetition
    )
    ref = _ref_apply_penalties(
        logits, prompt_rows, output_rows, presence, frequency, repetition
    )
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_greedy_no_penalty_is_plain_argmax():
    """No penalties + no context must be byte-for-byte the legacy greedy path."""
    torch.manual_seed(1)
    logits = torch.randn(8, 200)
    params = TTSamplingParams(temperature=0, top_k=-1, top_p=1.0)
    out = sample_tokens(logits, params)
    assert torch.equal(out, torch.argmax(logits, dim=-1))
    # Supplying context but no penalties must also be a no-op.
    out_ctx = sample_tokens(
        logits,
        params,
        prompt_token_ids=[[1, 2]] * 8,
        output_token_ids=[[3]] * 8,
    )
    assert torch.equal(out_ctx, torch.argmax(logits, dim=-1))


def test_repetition_penalty_changes_greedy_choice():
    """A strong repetition penalty must steer greedy away from a repeated token."""
    vocab = 10
    logits = torch.full((1, vocab), -5.0)
    logits[0, 4] = 10.0  # would be the greedy pick
    logits[0, 7] = 9.0  # runner-up
    # No penalty -> token 4.
    base = TTSamplingParams(temperature=0, top_k=-1, top_p=1.0)
    assert int(sample_tokens(logits, base)[0]) == 4
    # token 4 already generated + strong repetition penalty -> token 7 wins.
    penal = TTSamplingParams(
        temperature=0, top_k=-1, top_p=1.0, repetition_penalty=2.0
    )
    out = sample_tokens(
        logits, penal, prompt_token_ids=[[]], output_token_ids=[[4]]
    )
    assert int(out[0]) == 7


def test_frequency_penalty_scales_with_count():
    vocab = 6
    logits = torch.zeros(1, vocab)
    logits[0, 2] = 1.0
    params = TTSamplingParams(
        temperature=0, top_k=-1, top_p=1.0, frequency_penalty=0.5
    )
    # token 2 generated 4 times -> 1.0 - 0.5*4 = -1.0, so another token wins.
    out = sample_tokens(
        logits, params, prompt_token_ids=[[]], output_token_ids=[[2, 2, 2, 2]]
    )
    assert int(out[0]) != 2


def test_min_p_masks_low_prob_tokens():
    from tt_vllm_plugin.worker.tt_model_runner import apply_min_p

    logits = torch.tensor([[5.0, 4.9, -10.0, -12.0]])
    masked = apply_min_p(logits, torch.tensor([0.5]))
    # The two clearly-low-prob tokens must be -inf; the top two survive.
    assert torch.isneginf(masked[0, 2]) and torch.isneginf(masked[0, 3])
    assert not torch.isneginf(masked[0, 0]) and not torch.isneginf(masked[0, 1])

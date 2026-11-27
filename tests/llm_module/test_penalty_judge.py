# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host tests for ``judge_penalty_runs``, the verdict of the chat-completions suite's
``test_penalties`` (llm_module/test_vllm_chat_completions.py), without a server."""

import pytest

from llm_module.test_vllm_chat_completions import judge_penalty_runs

SEEDS = (1234, 1, 2, 3, 4)
# 12 words, 3 distinct (unique ratio 0.25), top word 4 times (share 1/3).
REPETITIVE = "the cat sat. the cat sat. the cat sat. the cat sat."
# 14 words, 13 distinct, top word ("the") twice.
VARIED = "a quick brown fox jumps over the lazy dog near the old mill today."


def _runs(base_texts, test_texts):
    return list(zip(SEEDS, base_texts, test_texts))


def _has_seeds(failure, seeds):
    return all(f"seed {seed}:" in failure for seed in seeds)


def test_one_unchanged_seed_passes():
    base = [REPETITIVE] * 5
    test = [VARIED] * 4 + [REPETITIVE]
    assert (
        judge_penalty_runs("repeat_trap", "frequency_penalty", _runs(base, test))
        is None
    )


@pytest.mark.parametrize(
    "penalty_param", ["presence_penalty", "frequency_penalty", "repetition_penalty"]
)
def test_an_ignored_penalty_fails_with_every_seed(penalty_param):
    texts = [VARIED, REPETITIVE, VARIED, REPETITIVE, VARIED]
    failure = judge_penalty_runs(
        "natural_repetition", penalty_param, _runs(texts, texts)
    )
    assert failure is not None and failure.startswith("the penalty had no effect")
    assert _has_seeds(failure, SEEDS)


@pytest.mark.parametrize(
    "prompt_name, penalty_param",
    [
        # the exemption from the diversity rule does not let an empty reply through
        ("repeat_trap", "presence_penalty"),
        ("natural_repetition", "repetition_penalty"),
    ],
)
@pytest.mark.parametrize("empty", ["", " \n\t", None])
def test_an_empty_penalized_reply_fails_with_its_seed(
    prompt_name, penalty_param, empty
):
    base = [REPETITIVE] * 5
    test = [VARIED, VARIED, VARIED, empty, VARIED]  # seed 3
    failure = judge_penalty_runs(prompt_name, penalty_param, _runs(base, test))
    assert failure is not None and failure.startswith("empty reply")
    assert "seed 3:" in failure
    assert not _has_seeds(failure, [1234])


def test_presence_on_the_repeat_trap_keeps_its_diversity_exemption():
    base = [VARIED] * 5
    test = [REPETITIVE] * 5
    assert (
        judge_penalty_runs("repeat_trap", "presence_penalty", _runs(base, test)) is None
    )


def test_reduced_diversity_fails_with_every_seed():
    base = [VARIED] * 5
    test = [REPETITIVE] * 5
    failure = judge_penalty_runs(
        "natural_repetition", "frequency_penalty", _runs(base, test)
    )
    assert failure is not None and failure.startswith("the penalty reduced diversity")
    assert _has_seeds(failure, SEEDS)


def test_repetition_is_judged_by_the_top_word_share_not_its_count():
    # Each penalized reply is three times longer and repeats its top word more often
    # (6 times against 4) while the word's share falls (6/36 against 4/12).
    words = []
    for i in range(30):
        if i % 5 == 0:
            words.append("the")
        words.append(f"word{i}")
    longer = " ".join(words)
    base = [REPETITIVE] * 5
    test = [longer] * 5
    for penalty_param in ("frequency_penalty", "repetition_penalty"):
        assert (
            judge_penalty_runs("repeat_trap", penalty_param, _runs(base, test)) is None
        )


def test_a_growing_top_word_share_fails_with_every_seed():
    # Same unique-word ratio (18/20), top word's share 2/20 -> 3/20.
    base_words = [f"w{i}" for i in range(18)] + ["x", "y"]
    base_words[0], base_words[1] = "x", "y"
    test_words = [f"v{i}" for i in range(17)] + ["x", "x", "x"]
    base = [" ".join(base_words)] * 5
    test = [" ".join(test_words)] * 5
    failure = judge_penalty_runs("repeat_trap", "frequency_penalty", _runs(base, test))
    assert failure is not None
    assert failure.startswith("the penalty did not reduce repetition")
    assert _has_seeds(failure, SEEDS)


def test_repetition_penalty_must_raise_diversity():
    # A different reply with the same unique-word ratio (13/14): changed, not more diverse.
    base = [VARIED] * 5
    test = [VARIED.replace("today", "tonight")] * 5
    failure = judge_penalty_runs(
        "natural_repetition", "repetition_penalty", _runs(base, test)
    )
    assert failure is not None
    assert failure.startswith("repetition_penalty did not increase diversity")
    assert _has_seeds(failure, SEEDS)
    assert (
        judge_penalty_runs("natural_repetition", "frequency_penalty", _runs(base, test))
        is None
    )

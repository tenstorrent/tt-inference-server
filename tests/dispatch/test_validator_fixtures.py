# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device unit test for the shared output validator (issue #46).

Asserts the tuned cascade classifies all 12 labeled samples (captured on card
2026-06-11; see ~/dispatch/tests/diagnostic/tune_validator.py) correctly, and that
each garbage sample fails for the EXPECTED reason category. Frozen reason prefixes
are a contract for the sweep/diff tooling, so we assert on them here too.
"""

import pytest

from tt_inference_server.dispatch.output_validator import (
    ValidatorThresholds,
    validate_output,
)

# (name, expect_good, text, reason_category_when_garbage)
# reason_category is one of: "repetition", "exotic", "punct", or None for good samples.
SAMPLES = [
    (
        "llama3-8b",
        True,
        " Paris, which is located in the north-central part of the country. Paris is the most populous city in France and",
        None,
    ),
    (
        "deepseek",
        True,
        " its official language is French. The currency is the Euro, and the country is part of the European Union.",
        None,
    ),
    (
        "qwen2.5-7b",
        True,
        " The theory of relativity, developeded by Albert Einstein, consists of two parts: Special Relativity and General Relativity.\n\n1. **Special Relativity (1905)**: This part deals with objects moving at constant speeds, espe",
        None,
    ),
    (
        "olmo-1b",
        True,
        " the city of Paris. It is the largest city in France and the second largest in Europe.",
        None,
    ),
    (
        "stablelm-2",
        True,
        " \n\nThe theory of relativity, proposed by Albert Einstein in 1905, is a fundamental principle in physics that describes the relationship between space and time. It is based on two key postulates:\n\n1. The laws of physics are the same for all ",
        None,
    ),
    (
        "phi-3.5",
        True,
        " Paris. It is the most populous city in France and serves as the country's political, economic, and cultural center.",
        None,
    ),
    (
        "gemma2-2b",
        True,
        "\n\n**Special Relativity**\n\n* **The speed of light is constant:** No matter how fast you're moving, light always travels at the same speed (approximately 299,792,458 meters per second).\n* **Time and space are relative:** Time and space are no",
        None,
    ),
    (
        "qwen3-4b",
        True,
        " The theory of relativity is a fundamental concept in physics that describes how gravity and space-time are interconnected. It was developed by Albert Einstein in the early 20th century and consists of two main parts: special relativity and",
        None,
    ),
    (
        "bloom-3b",
        False,
        "\n plume todo br R retr錄》Berَحَ principal all开展的活动 exbras zel 70 At bel de banyakmansगानिस्तान a general aوفيما开展的活动并注意到并注意到helgraf whole ar hamee ris pr 1 total aterior de》一卷 or;plhar b la le;;dos note des specificr cheg",
        "exotic",
    ),
    (
        "pythia-2.8b",
        False,
        " **less** (neTkl. DISTH-\n.P. P. THEILOG==========================) L. IDRIG-j I. HARDERAR) P. 1care' '\n\n                                                                        .equal- _rof_ ) + B _cons** - >_.\n.aken.er_ ",
        "punct",
    ),
    (
        "mistral-7b",
        False,
        "Albert Einstein developed the theory of of relativity in simple terms:\n\n1 bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan bekan be",
        "repetition",
    ),
    (
        "starcoder2-3b",
        False,
        "_ CELLSPACING SizedBoxGf_ LGPL_ Franc_ Franc_ Franc_ Franc_ Franc_ Franc_ eros segurança_ Franc_ eros ÷luş_ eros armazenamento_ eros ÷Hh_ Chen_ Franc_ eros segurança_ Franc_ eros segurança_ Franc_ eros segurança_ eros armazenamento_ eros se",
        "repetition",
    ),
]

# Token ids of plausible length with no single-token loop, so the TEXT checks decide
# (the captured samples are 80-token continuations).
_NO_LOOP_IDS = list(range(80))


def _category(reason: str) -> str:
    if reason.startswith("repetition loop"):
        return "repetition"
    if "non-Latin" in reason:
        return "exotic"
    if "punctuation/symbols" in reason:
        return "punct"
    return "other"


@pytest.mark.parametrize(
    "name,expect_good,text,category", SAMPLES, ids=[s[0] for s in SAMPLES]
)
def test_validator_classifies_sample(name, expect_good, text, category):
    is_valid, reason = validate_output(text, _NO_LOOP_IDS)
    assert is_valid == expect_good, (
        f"{name}: expected good={expect_good}, got {reason!r}"
    )
    if not expect_good:
        assert _category(reason) == category, (
            f"{name}: expected {category} reason, got {reason!r}"
        )


def test_all_twelve_classified():
    """The headline guarantee: 12/12 correct."""
    wrong = [s[0] for s in SAMPLES if validate_output(s[2], _NO_LOOP_IDS)[0] != s[1]]
    assert not wrong, f"misclassified: {wrong}"


def test_empty_and_too_short_reasons():
    assert validate_output("", _NO_LOOP_IDS) == (False, "empty output")
    assert validate_output("   ", _NO_LOOP_IDS)[0] is False
    ok, reason = validate_output("hello world", [1, 2, 3])
    assert ok is False and reason.startswith("too short:")


def test_single_token_repetition_caught_via_ids():
    """The check tune_validator dropped: a token-level loop the text n-gram check could
    miss (diverse-looking text but one token id dominating)."""
    ids = [42] * 60 + list(range(20))  # 60/80 = 75% one id
    ok, reason = validate_output("varied looking words here " * 5, ids)
    assert ok is False and reason == "repetition loop (token)"


def test_text_only_mode_skips_length_checks():
    """token_ids=None => length + token-rep checks skipped (legacy bench.py path)."""
    ok, reason = validate_output("Paris is the capital of France.", None)
    assert ok is True and reason == "ok"


def test_thresholds_override_relaxes_exotic():
    """A multilingual/code novel model can relax the English-smoke heuristics."""
    text = "Paris 東京 北京 ソウル"  # several non-Latin letters
    assert validate_output(text, None)[0] is False
    relaxed = ValidatorThresholds(exotic_char_limit=100, punct_fraction=0.9)
    assert validate_output(text, None, thresholds=relaxed)[0] is True

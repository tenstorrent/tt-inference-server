# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Noise-aware acceptance rule for eval scores, with no third-party imports so
that tools outside the eval venvs (scripts/gpu_reference_colab) grade exactly
as test_module.llm_tests.llm_eval_tests does, which re-exports these names."""

from __future__ import annotations

import math
from typing import Optional, Tuple

# --- noise-aware acceptance for low-baseline accuracy tasks ------------------
#
# The ratio rule (score / reference >= 1 - tolerance, tolerance 0.05) is a
# relative bar. For a task whose reference is in the single digits, 5% of the
# reference is smaller than the sampling noise of the score itself: Llama-3.2-3B
# leaderboard_ifeval scored 9.43 then 8.32 on the same 541 prompts (about 1 SE)
# and the ratio went 1.02 -> 0.90. So a task also PASSES when its shortfall is
# within NOISE_Z binomial standard errors of the reference:
#
#     reference - score <= NOISE_Z * 100 * sqrt(p * (1 - p) / n),
#     p = reference / 100
#
# NOISE_Z is fixed in advance at 1.96 (two-sided 95%, one-sided 97.5%) and is
# never fitted to observed scores. Only the reference's own sampling error is
# counted, not the run's, so the bar is the stricter of the two natural
# choices (a two-sample test would widen it by sqrt(2)). For a high-baseline
# task the term is small next to 5% of the reference, so the ratio rule still
# decides there.
#
# The binomial model is valid only where every sample scores 0 or 1, so the
# rule applies only when:
#   * the unit is percent and every configured metric is one of
#     BINOMIAL_METRICS (per-sample correct/incorrect), and
#   * n is known: the scored entry's effective sample count (a group's count is
#     the sum of its leaves), or, for an unweighted mean of k subtask
#     proportions (score_multilevel_keys_mean, e.g. leaderboard_math_hard), the
#     effective n of that mean, k**2 / sum(1 / n_i), with p taken at the
#     reference for every subtask.
# Everything else is graded by the ratio rule alone. That excludes continuous
# per-sample metrics (LongBench F1 / ROUGE / "score", ANLS, WER, relaxed
# accuracy) and instruction-level IFEval accuracy, whose unit is the
# instruction while n counts prompts. For a [0, 1]-bounded per-sample metric
# the binomial SE is only an upper bound on the true SE (Bhatia-Davis), so
# using it there would loosen the bar by an unknown amount.
NOISE_Z = 1.96

BINOMIAL_METRICS = frozenset(
    {
        "acc",
        "acc_norm",
        "exact_match",
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "pass_at_1",
        "pass@1",
    }
)

RULE_RATIO = "ratio"
RULE_WITHIN_NOISE = "within_noise"
RULE_FAIL = "fail"
RULE_NA = "na"


def _metric_name(key) -> str:
    return str(key).split(",", 1)[0]


def binomial_noise_n(
    result_keys, t_key: str, sample_counts: Optional[dict]
) -> Optional[float]:
    """The n of a binomial-eligible score, or None when the rule does not apply.

    ``result_keys`` are the score function's configured keys: one plain metric
    key on ``t_key`` (n is ``t_key``'s count), or (subtask, metric) paths whose
    unweighted mean is the score (n is k**2 / sum(1 / n_i)). Any other shape --
    a mean of several metrics on one entry, a non-binomial metric, an unknown
    count -- returns None.
    """
    counts = sample_counts or {}
    keys = list(result_keys or [])
    if not keys:
        return None

    def known(count) -> bool:
        return isinstance(count, int) and not isinstance(count, bool) and count > 0

    if all(isinstance(k, str) for k in keys):
        if len(keys) != 1 or _metric_name(keys[0]) not in BINOMIAL_METRICS:
            return None
        count = counts.get(t_key)
        return float(count) if known(count) else None
    if all(isinstance(k, tuple) and len(k) == 2 for k in keys):
        if any(_metric_name(metric) not in BINOMIAL_METRICS for _, metric in keys):
            return None
        sizes = [counts.get(name) for name, _ in keys]
        if not all(known(size) for size in sizes):
            return None
        return len(sizes) ** 2 / sum(1.0 / size for size in sizes)
    return None


def noise_aware_check(
    ratio_pass: Optional[bool],
    score: float,
    reference: Optional[float],
    n: Optional[float],
    z: float = NOISE_Z,
) -> Tuple[Optional[bool], str, Optional[float]]:
    """Apply the noise-aware rule on top of the ratio rule's verdict.

    Returns ``(passed, rule, se)``: ``rule`` is ``ratio`` when the ratio rule
    passed, ``within_noise`` when only the noise rule passed, ``fail`` when
    neither did, and ``na`` when there is no reference to grade against.
    ``se`` is the binomial SE in percentage points at the reference, or None
    when the rule does not apply (no n, or a reference outside (0, 100]).
    """
    if ratio_pass is None:
        return None, RULE_NA, None
    se = None
    if n and reference is not None and 0 < reference <= 100:
        p = reference / 100.0
        se = 100.0 * math.sqrt(p * (1.0 - p) / n)
    if ratio_pass:
        return True, RULE_RATIO, se
    if se is not None and reference - score <= z * se:
        return True, RULE_WITHIN_NOISE, se
    return False, RULE_FAIL, se

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Milestone-0 rubric scoring.

Turns a submission into the completed scorecard of RFP Part K: every line's
qualifying and excellence values, measured figure, fraction, weight and score,
then group subtotals, the core total, the bonus and the overall.

This ranks competing commercial bids, so the arithmetic has to be reproducible
and auditable rather than merely correct once. Three things follow from that:

* **Nothing is authored per point.** Each graded point's qualifying and
  excellence values are derived from that point's own target using the published
  multipliers (Appendix B.5). A Partner can recompute any value themselves, and
  there is no second set of numbers to drift.
* **Every intermediate is kept.** :class:`LineScore` carries the inputs to its
  own fraction, and per-point lines keep every point's contribution, so a
  disputed score can be walked back to the measurement it came from.
* **An unscoreable line is recorded, not silently zeroed.** It still contributes
  zero — the rubric has no other option — but it is flagged, because "measured
  exactly at qualifying" and "never measured" are different facts about a
  submission and must not look identical on a scorecard.

Scoring is linear between two absolute values in both directions (requirements
K.2)::

    fraction = clamp((measured - qualifying) / (excellence - qualifying), 0, 1)

Lower-is-better lines need no special case: their excellence value is below their
qualifying value, so both differences flip sign and the expression is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The published rubric (Appendix B.5). Changing anything here changes published
# terms, so the values live in one place and are named for what they are.
# ---------------------------------------------------------------------------

#: Qualifying value = that point's mean time-to-first-token target x this.
#:
#: The gate targets the mean while the rubric scores percentiles, and a percentile
#: of a latency distribution sits above its mean. Each factor is set to the ratio
#: measured on Tenstorrent hardware (gpt-oss-120b release run, 16 points: median
#: p50/mean 1.000, p99/mean 1.192) so that meeting the mean target scores zero,
#: which is what requirements K.2 says every line must do.
#:
#: p99 is set at 1.35 rather than the observed 1.19 on purpose: tails widen under
#: load and on a first-of-kind bring-up, and a qualifying value at the median
#: observation would put the rubric's heaviest line out of reach of a system with
#: an ordinary tail. p90 was not captured in that run and is interpolated.
#:
#: These must match RFP Appendix B.5. A Partner is told they can compute their own
#: score from the published values; if these drift, that stops being true.
TTFT_QUALIFYING_FACTORS = {"p50": 1.00, "p90": 1.12, "p99": 1.35}

#: Excellence = qualifying x this. Full marks require twice the required
#: performance, in whichever direction "twice" means for that line.
LATENCY_EXCELLENCE_FACTOR = 0.50
THROUGHPUT_EXCELLENCE_FACTOR = 2.00

#: Concurrency-1 versus maximum-concurrency share of every per-point line.
#: Prefill under load is what production depends on; the idle corner is retained
#: because it is the only place isolated prefill compute is visible, and the
#: scaling fit needs it as a clean baseline.
CONCURRENCY_SHARES = (0.25, 0.75)

LINE_WEIGHTS: Dict[str, float] = {
    # Prefill — 55
    "ttft_p99": 22,
    "ttft_p90": 11,
    "ttft_p50": 5,
    "prefill_throughput": 10,
    "tail_discipline": 4,
    "scaling_quality": 3,
    # Decode — 13
    "tput_user_median": 7,
    "decode_throughput": 6,
    # Model quality — 22
    "agentic_eval": 12,
    "standard_eval": 10,
    # Engineering quality — 10
    "reproduced_first_attempt": 3,
    "run_to_run_cov": 2,
    "contribution_quality": 3,
    "technical_assistance": 2,
    # Bonus — 20
    "prefix_cache_hit_rate": 12,
    "ttft_uplift": 8,
}

GROUPS: Dict[str, Tuple[str, ...]] = {
    "prefill": (
        "ttft_p99",
        "ttft_p90",
        "ttft_p50",
        "prefill_throughput",
        "tail_discipline",
        "scaling_quality",
    ),
    "decode": ("tput_user_median", "decode_throughput"),
    "quality": ("agentic_eval", "standard_eval"),
    "engineering": (
        "reproduced_first_attempt",
        "run_to_run_cov",
        "contribution_quality",
        "technical_assistance",
    ),
    "bonus": ("prefix_cache_hit_rate", "ttft_uplift"),
}

CORE_GROUPS = ("prefill", "decode", "quality", "engineering")

LINE_TITLES: Dict[str, str] = {
    "ttft_p99": "TTFT p99",
    "ttft_p90": "TTFT p90",
    "ttft_p50": "TTFT median",
    "prefill_throughput": "Prefill throughput",
    "tail_discipline": "Tail discipline",
    "scaling_quality": "Scaling quality",
    "tput_user_median": "Tokens/s/user median",
    "decode_throughput": "Decode throughput",
    "agentic_eval": "Agentic eval margin",
    "standard_eval": "Standard eval margin",
    "reproduced_first_attempt": "Reproduced first attempt",
    "run_to_run_cov": "Coefficient of variation",
    "contribution_quality": "Contribution quality",
    "technical_assistance": "Technical assistance",
    "prefix_cache_hit_rate": "Prefix cache hit rate",
    "ttft_uplift": "TTFT uplift",
}

#: Lines carrying explicit values rather than per-point derived ones.
FIXED_RANGES: Dict[str, Tuple[float, float]] = {
    # Measured p99/p50 on real hardware runs ~1.19; 4.0 was a round guess that
    # every system cleared outright. See Appendix B.5.
    "tail_discipline": (1.25, 1.05),
    "scaling_quality": (2.0, 1.0),
    "agentic_eval": (1.00, 1.15),
    "standard_eval": (1.00, 1.15),
    "run_to_run_cov": (0.15, 0.02),
    "contribution_quality": (0.0, 3.0),
    "technical_assistance": (12.0, 0.0),
    "prefix_cache_hit_rate": (0.70, 0.82),
    "ttft_uplift": (0.0, 60.0),
}


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------


def fraction(
    measured: Optional[float], qualifying: float, excellence: float
) -> Optional[float]:
    """Where ``measured`` sits between qualifying and excellence, clamped to 0-1.

    Returns None when the line cannot be scored at all: no measurement, or a
    degenerate range where the two values coincide and there is no scale to
    place anything on. None is not 0 — 0 means "measured, and at the bar".
    """
    if measured is None or not isinstance(measured, (int, float)):
        return None
    if isinstance(measured, bool):
        return None
    span = excellence - qualifying
    if span == 0:
        return None
    return max(0.0, min(1.0, (float(measured) - qualifying) / span))


@dataclass
class PointContribution:
    """One graded point's part of one per-point line."""

    concurrency: int
    input_length: int
    qualifying: Optional[float]
    excellence: Optional[float]
    measured: Optional[float]
    fraction: Optional[float]
    weight: float

    @property
    def contribution(self) -> float:
        return (self.fraction or 0.0) * self.weight


@dataclass
class LineScore:
    key: str
    weight: float
    fraction: Optional[float]
    qualifying: Optional[float] = None
    excellence: Optional[float] = None
    measured: Optional[float] = None
    #: Per-point detail, for lines aggregated across the sweep.
    points: List[PointContribution] = field(default_factory=list)
    #: Why the line scores what it does, when that is not simply the arithmetic.
    note: str = ""

    @property
    def score(self) -> float:
        return (self.fraction or 0.0) * self.weight

    @property
    def scoreable(self) -> bool:
        return self.fraction is not None

    @property
    def title(self) -> str:
        return LINE_TITLES.get(self.key, self.key)


@dataclass
class Scorecard:
    partner: str
    model: str
    lines: Dict[str, LineScore]
    notes: List[str] = field(default_factory=list)

    def group_score(self, group: str) -> float:
        return sum(self.lines[k].score for k in GROUPS[group] if k in self.lines)

    def group_weight(self, group: str) -> float:
        return sum(LINE_WEIGHTS[k] for k in GROUPS[group])

    @property
    def core_total(self) -> float:
        return sum(self.group_score(g) for g in CORE_GROUPS)

    @property
    def bonus_total(self) -> float:
        return self.group_score("bonus")

    @property
    def overall(self) -> float:
        return self.core_total + self.bonus_total

    @property
    def unscoreable(self) -> List[str]:
        return [k for k, line in self.lines.items() if not line.scoreable]


# ---------------------------------------------------------------------------
# Per-point weights
# ---------------------------------------------------------------------------


def point_weights(
    points: Sequence[Tuple[int, int]],
    shares: Tuple[float, float] = CONCURRENCY_SHARES,
) -> Dict[Tuple[int, int], float]:
    """Weight for every ``(concurrency, input_length)`` point. Sums to 1.0.

    Two factors combine, per Appendix B.5. Within a concurrency level, weight is
    proportional to ``log2(input_length)`` so long inputs matter more without the
    longest point dominating. The level is then scaled by its share.

    Raises when the sweep does not have exactly two concurrency levels. The
    published shares describe an idle corner and a loaded corner; silently
    spreading them over three levels would produce weights nobody agreed to and
    a score no Partner could reproduce.
    """
    levels = sorted({c for c, _ in points})
    if len(levels) != 2:
        raise ValueError(
            f"Per-point weights need exactly two concurrency levels "
            f"(an idle corner and a loaded one); this sweep has {len(levels)}: "
            f"{levels}. Appendix B.5 publishes shares for two."
        )
    level_share = dict(zip(levels, shares))

    weights: Dict[Tuple[int, int], float] = {}
    for level in levels:
        lengths = [isl for c, isl in points if c == level]
        logs = {isl: math.log2(isl) for isl in lengths}
        total = sum(logs.values())
        if total <= 0:
            raise ValueError(f"No usable input lengths at concurrency {level}.")
        for isl in lengths:
            weights[(level, isl)] = logs[isl] / total * level_share[level]
    return weights


# ---------------------------------------------------------------------------
# The submission
# ---------------------------------------------------------------------------


@dataclass
class GradedPoint:
    """One operating point: its targets and what was measured there."""

    concurrency: int
    input_length: int
    #: Mean time-to-first-token target for this point, in ms (Appendix B.2).
    target_ttft_ms: Optional[float] = None
    target_tput_user: Optional[float] = None
    target_decode_throughput: Optional[float] = None
    p50_ttft: Optional[float] = None
    p90_ttft: Optional[float] = None
    p99_ttft: Optional[float] = None
    prefill_throughput: Optional[float] = None
    tail_ratio: Optional[float] = None
    tput_user: Optional[float] = None
    decode_throughput: Optional[float] = None

    @property
    def key(self) -> Tuple[int, int]:
        return (self.concurrency, self.input_length)


@dataclass
class Submission:
    partner: str = ""
    model: str = ""
    points: List[GradedPoint] = field(default_factory=list)
    #: Fitted time-to-first-token growth exponent per concurrency level.
    scaling_exponents: Dict[int, Optional[float]] = field(default_factory=dict)
    #: Single-figure lines, keyed as in :data:`LINE_WEIGHTS`.
    once: Dict[str, Optional[float]] = field(default_factory=dict)
    reproduced_first_attempt: Optional[bool] = None
    #: Lines that failed reproduction (J.4) or were waived (M.5). Both score
    #: zero rather than a reduced score or removal from the denominator.
    failed_reproduction: Tuple[str, ...] = ()
    waived: Tuple[str, ...] = ()


# Per-point line -> (measured attribute, target attribute, excellence factor).
# Latency lines additionally take a qualifying factor from
# TTFT_QUALIFYING_FACTORS; throughput lines qualify at the target itself.
_PER_POINT_LINES: Dict[str, Tuple[str, str, float, Optional[str]]] = {
    "ttft_p50": ("p50_ttft", "target_ttft_ms", LATENCY_EXCELLENCE_FACTOR, "p50"),
    "ttft_p90": ("p90_ttft", "target_ttft_ms", LATENCY_EXCELLENCE_FACTOR, "p90"),
    "ttft_p99": ("p99_ttft", "target_ttft_ms", LATENCY_EXCELLENCE_FACTOR, "p99"),
    "prefill_throughput": (
        "prefill_throughput",
        "target_ttft_ms",
        THROUGHPUT_EXCELLENCE_FACTOR,
        None,
    ),
    "tput_user_median": (
        "tput_user",
        "target_tput_user",
        THROUGHPUT_EXCELLENCE_FACTOR,
        None,
    ),
    "decode_throughput": (
        "decode_throughput",
        "target_decode_throughput",
        THROUGHPUT_EXCELLENCE_FACTOR,
        None,
    ),
}


def _point_range(
    line: str, point: GradedPoint
) -> Tuple[Optional[float], Optional[float]]:
    """Qualifying and excellence for one line at one point, derived from target."""
    measured_attr, target_attr, exc_factor, percentile = _PER_POINT_LINES[line]
    target = getattr(point, target_attr)
    if target is None or target <= 0:
        return None, None

    if percentile is not None:
        qualifying = target * TTFT_QUALIFYING_FACTORS[percentile]
    elif line == "prefill_throughput":
        # Tokens per second implied by hitting the time-to-first-token target.
        qualifying = point.input_length / (target / 1000.0)
    else:
        qualifying = target
    return qualifying, qualifying * exc_factor


def _score_per_point_line(
    line: str, points: Sequence[GradedPoint], weights: Mapping[Tuple[int, int], float]
) -> LineScore:
    measured_attr = _PER_POINT_LINES[line][0]
    contributions: List[PointContribution] = []
    for point in points:
        qualifying, excellence = _point_range(line, point)
        measured = getattr(point, measured_attr)
        frac = (
            None
            if qualifying is None or excellence is None
            else fraction(measured, qualifying, excellence)
        )
        contributions.append(
            PointContribution(
                concurrency=point.concurrency,
                input_length=point.input_length,
                qualifying=qualifying,
                excellence=excellence,
                measured=measured,
                fraction=frac,
                weight=weights[point.key],
            )
        )

    scored = [c for c in contributions if c.fraction is not None]
    line_fraction = sum(c.contribution for c in contributions) if scored else None
    note = ""
    if scored and len(scored) != len(contributions):
        missing = len(contributions) - len(scored)
        note = (
            f"{missing} of {len(contributions)} points could not be scored and "
            f"contributed zero."
        )
    return LineScore(
        key=line,
        weight=LINE_WEIGHTS[line],
        fraction=line_fraction,
        points=contributions,
        note=note,
    )


def _score_tail_discipline(
    points: Sequence[GradedPoint], weights: Mapping[Tuple[int, int], float]
) -> LineScore:
    """Weighted across the sweep, but against one range: the ratio is scale-free."""
    qualifying, excellence = FIXED_RANGES["tail_discipline"]
    contributions = [
        PointContribution(
            concurrency=p.concurrency,
            input_length=p.input_length,
            qualifying=qualifying,
            excellence=excellence,
            measured=p.tail_ratio,
            fraction=fraction(p.tail_ratio, qualifying, excellence),
            weight=weights[p.key],
        )
        for p in points
    ]
    scored = [c for c in contributions if c.fraction is not None]
    return LineScore(
        key="tail_discipline",
        weight=LINE_WEIGHTS["tail_discipline"],
        fraction=sum(c.contribution for c in contributions) if scored else None,
        qualifying=qualifying,
        excellence=excellence,
        points=contributions,
    )


def _score_scaling_quality(
    exponents: Mapping[int, Optional[float]],
    shares: Tuple[float, float] = CONCURRENCY_SHARES,
) -> LineScore:
    """One fit per concurrency level, combined by concurrency share.

    Kept separate rather than pooled because a system can scale cleanly when
    idle and degrade under load; a single regression over every point averages
    exactly that away.
    """
    qualifying, excellence = FIXED_RANGES["scaling_quality"]
    levels = sorted(exponents)
    if len(levels) != 2:
        return LineScore(
            key="scaling_quality",
            weight=LINE_WEIGHTS["scaling_quality"],
            fraction=None,
            qualifying=qualifying,
            excellence=excellence,
            note=(
                f"Needs a fitted exponent at each of two concurrency levels; "
                f"got {len(levels)}."
            ),
        )

    contributions: List[PointContribution] = []
    for level, share in zip(levels, shares):
        exponent = exponents[level]
        contributions.append(
            PointContribution(
                concurrency=level,
                input_length=0,
                qualifying=qualifying,
                excellence=excellence,
                measured=exponent,
                fraction=fraction(exponent, qualifying, excellence),
                weight=share,
            )
        )

    scored = [c for c in contributions if c.fraction is not None]
    note = ""
    if len(scored) != len(contributions):
        unfitted = [c.concurrency for c in contributions if c.fraction is None]
        note = (
            f"No exponent could be fitted at concurrency {unfitted}; "
            f"that share contributed zero."
        )
    return LineScore(
        key="scaling_quality",
        weight=LINE_WEIGHTS["scaling_quality"],
        fraction=sum(c.contribution for c in contributions) if scored else None,
        qualifying=qualifying,
        excellence=excellence,
        points=contributions,
        note=note,
    )


def _score_once_line(key: str, measured: Optional[float]) -> LineScore:
    qualifying, excellence = FIXED_RANGES[key]
    return LineScore(
        key=key,
        weight=LINE_WEIGHTS[key],
        fraction=fraction(measured, qualifying, excellence),
        qualifying=qualifying,
        excellence=excellence,
        measured=measured,
    )


def score(submission: Submission) -> Scorecard:
    """Score a submission into a complete scorecard."""
    notes: List[str] = []
    lines: Dict[str, LineScore] = {}

    if submission.points:
        weights = point_weights([p.key for p in submission.points])
        for line in _PER_POINT_LINES:
            lines[line] = _score_per_point_line(line, submission.points, weights)
        lines["tail_discipline"] = _score_tail_discipline(submission.points, weights)
    else:
        notes.append("No graded points supplied; every per-point line scores zero.")
        for line in list(_PER_POINT_LINES) + ["tail_discipline"]:
            lines[line] = LineScore(
                key=line,
                weight=LINE_WEIGHTS[line],
                fraction=None,
                note="No graded points.",
            )

    lines["scaling_quality"] = _score_scaling_quality(submission.scaling_exponents)

    for key in (
        "agentic_eval",
        "standard_eval",
        "run_to_run_cov",
        "contribution_quality",
        "technical_assistance",
        "prefix_cache_hit_rate",
        "ttft_uplift",
    ):
        lines[key] = _score_once_line(key, submission.once.get(key))

    # Binary: full marks or nothing, with no scale in between.
    reproduced = submission.reproduced_first_attempt
    lines["reproduced_first_attempt"] = LineScore(
        key="reproduced_first_attempt",
        weight=LINE_WEIGHTS["reproduced_first_attempt"],
        fraction=None if reproduced is None else (1.0 if reproduced else 0.0),
        measured=None if reproduced is None else float(reproduced),
        note="" if reproduced is not None else "Reproduction outcome not recorded.",
    )

    _apply_zeroing_rules(lines, submission, notes)
    return Scorecard(
        partner=submission.partner, model=submission.model, lines=lines, notes=notes
    )


def _apply_zeroing_rules(
    lines: Dict[str, LineScore], submission: Submission, notes: List[str]
) -> None:
    """The two rules of K.8, which remove any incentive to game the process.

    A line that failed reproduction scores zero rather than a reduced score, and
    a waived line scores zero rather than leaving the denominator. Both are
    applied after scoring and are recorded on the line, so the scorecard shows
    what the arithmetic would otherwise have produced.
    """
    for key in submission.failed_reproduction:
        line = lines.get(key)
        if line is None:
            notes.append(f"Unknown line in failed_reproduction: {key}")
            continue
        was = line.fraction
        line.fraction = 0.0
        line.note = (
            f"Scored zero: failed reproduction (J.4). "
            f"Would otherwise have scored {(was or 0.0) * line.weight:.2f}."
        )

    for key in submission.waived:
        line = lines.get(key)
        if line is None:
            notes.append(f"Unknown line in waived: {key}")
            continue
        was = line.fraction
        line.fraction = 0.0
        line.note = (
            f"Scored zero: waived (M.5). A waiver protects qualification and "
            f"never improves rank. Would otherwise have scored "
            f"{(was or 0.0) * line.weight:.2f}."
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:,.{places}g}" if abs(value) >= 1000 else f"{value:.{places}f}"


def render_markdown(card: Scorecard) -> str:
    """The completed scorecard of Part K.9."""
    out = [
        f"# Milestone-0 scorecard — {card.partner or 'Partner'}"
        f"{f' — {card.model}' if card.model else ''}",
        "",
        "| Line | Qualifying | Excellence | Measured | Fraction | Weight | Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for group in CORE_GROUPS:
        for key in GROUPS[group]:
            line = card.lines.get(key)
            if line is None:
                continue
            out.append(
                f"| {line.title} | {_fmt(line.qualifying)} | {_fmt(line.excellence)} "
                f"| {_fmt(line.measured)} | {_fmt(line.fraction)} | "
                f"{line.weight:g} | **{line.score:.2f}** |"
            )
        out.append(
            f"| _{group} subtotal_ | | | | | {card.group_weight(group):g} | "
            f"**{card.group_score(group):.2f}** |"
        )

    out.append(f"| **Core total** | | | | | **100** | **{card.core_total:.2f}** |")
    for key in GROUPS["bonus"]:
        line = card.lines.get(key)
        if line is None:
            continue
        out.append(
            f"| {line.title} | {_fmt(line.qualifying)} | {_fmt(line.excellence)} "
            f"| {_fmt(line.measured)} | {_fmt(line.fraction)} | "
            f"{line.weight:g} | **{line.score:.2f}** |"
        )
    out.append(f"| **Bonus total** | | | | | **20** | **{card.bonus_total:.2f}** |")
    out.append(f"| **Overall** | | | | | | **{card.overall:.2f}** |")
    out.append("")

    flagged = [line for line in card.lines.values() if line.note]
    if flagged:
        out += ["## Line notes", ""]
        out += [f"- **{line.title}** — {line.note}" for line in flagged]
        out.append("")

    if card.unscoreable:
        out += [
            "## Lines that could not be scored",
            "",
            "These contributed zero, which is the only option the rubric has. They "
            "are listed separately because a line measured exactly at its "
            "qualifying value also scores zero, and the two are not the same fact "
            "about a submission.",
            "",
        ]
        out += [f"- {LINE_TITLES.get(k, k)}" for k in card.unscoreable]
        out.append("")

    if card.notes:
        out += ["## Notes", ""] + [f"- {n}" for n in card.notes] + [""]

    return "\n".join(out)


def to_dict(card: Scorecard) -> Dict[str, Any]:
    """Machine-readable scorecard, with every intermediate retained."""
    return {
        "partner": card.partner,
        "model": card.model,
        "core_total": round(card.core_total, 4),
        "bonus_total": round(card.bonus_total, 4),
        "overall": round(card.overall, 4),
        "groups": {
            g: {
                "weight": card.group_weight(g),
                "score": round(card.group_score(g), 4),
            }
            for g in list(CORE_GROUPS) + ["bonus"]
        },
        "lines": {
            key: {
                "title": line.title,
                "weight": line.weight,
                "qualifying": line.qualifying,
                "excellence": line.excellence,
                "measured": line.measured,
                "fraction": line.fraction,
                "score": round(line.score, 4),
                "scoreable": line.scoreable,
                "note": line.note,
                "points": [
                    {
                        "concurrency": p.concurrency,
                        "input_length": p.input_length,
                        "qualifying": p.qualifying,
                        "excellence": p.excellence,
                        "measured": p.measured,
                        "fraction": p.fraction,
                        "weight": round(p.weight, 6),
                        "contribution": round(p.contribution, 6),
                    }
                    for p in line.points
                ],
            }
            for key, line in card.lines.items()
        },
        "unscoreable": card.unscoreable,
        "notes": card.notes,
    }


def submission_from_dict(data: Mapping[str, Any]) -> Submission:
    """Build a :class:`Submission` from a plain JSON document.

    The scored assistance-unit total is supplied here rather than read from a
    register: that register is Partner-commercial data and lives outside this
    public repository.
    """
    return Submission(
        partner=str(data.get("partner", "")),
        model=str(data.get("model", "")),
        points=[GradedPoint(**p) for p in data.get("points", [])],
        scaling_exponents={
            int(k): v for k, v in (data.get("scaling_exponents") or {}).items()
        },
        once=dict(data.get("once") or {}),
        reproduced_first_attempt=data.get("reproduced_first_attempt"),
        failed_reproduction=tuple(data.get("failed_reproduction") or ()),
        waived=tuple(data.get("waived") or ()),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Score a Milestone-0 submission.")
    parser.add_argument("submission", help="submission JSON file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output instead of markdown",
    )
    args = parser.parse_args(argv)

    card = score(submission_from_dict(json.loads(Path(args.submission).read_text())))
    print(
        json.dumps(to_dict(card), indent=2) if args.json else render_markdown(card),
        end="" if not args.json else "\n",
    )

    # A scorecard with unscoreable lines is not a finished assessment.
    return 1 if card.unscoreable else 0


__all__ = [
    "CONCURRENCY_SHARES",
    "FIXED_RANGES",
    "GROUPS",
    "GradedPoint",
    "LINE_WEIGHTS",
    "LineScore",
    "PointContribution",
    "Scorecard",
    "Submission",
    "TTFT_QUALIFYING_FACTORS",
    "fraction",
    "point_weights",
    "render_markdown",
    "score",
    "submission_from_dict",
    "to_dict",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())

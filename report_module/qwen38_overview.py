# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Read-only Qwen bring-up overview; never modifies checks or acceptance."""

import math
from collections.abc import Mapping

from report_module.markdown_table import build_markdown_table


def _state(value):
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return {
        "1": "na",
        "2": "pass",
        "3": "fail",
        "pass": "pass",
        "passed": "pass",
        "fail": "fail",
        "failed": "fail",
        "na": "na",
        "n/a": "na",
    }.get(str(value).lower(), "unknown")


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _display(value):
    number = _number(value)
    return f"{number:,.2f}" if number is not None else "—"


def measured_level(record):
    """Highest completely passing tier, without counting NA as evidence of a pass."""
    if str(record.get("status", "")).lower() in {"error", "fail", "skip", "na"}:
        return "⚪ Unrated"
    if (_number(record.get("error_request_count")) or 0) > 0:
        return "⚪ Unrated"
    if not any(
        _number(record.get(k)) is not None
        for k in (
            "mean_ttft_ms",
            "mean_tpot_ms",
            "tps_output_throughput",
            "mean_e2el_ms",
        )
    ):
        return "⚪ Unrated"
    tiers = record.get("target_checks")
    if not isinstance(tiers, Mapping):
        return "⚪ Unrated"
    graded = False
    for tier, label in (
        ("target", "🟩◆ Target"),
        ("complete", "🟢 Complete"),
        ("functional", "🟡 Functional"),
    ):
        checks = tiers.get(tier)
        if not isinstance(checks, Mapping):
            continue
        states = [_state(v) for key, v in checks.items() if key.endswith("_check")]
        actual = [s for s in states if s != "na"]
        if not actual or "unknown" in actual:
            continue
        graded = True
        if all(s == "pass" for s in actual):
            return label
    return "🔴 Experimental" if graded else "⚪ Unrated"


def render_overview(blocks, metadata):
    # Limit this campaign-specific presentation to its explicit serving profiles.
    if not str(metadata.get("model_impl", "")).startswith("qwen38-autoport"):
        return ""
    rows = []
    for block in blocks:
        if block.kind not in {"benchmarks", "evals"} or not isinstance(
            block.data, Mapping
        ):
            continue
        records = block.data.get("records", [block.data])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            benchmark = block.kind == "benchmarks"
            tpot = _number(record.get("mean_tpot_ms"))
            label = (
                f"C{record.get('concurrency', '?')} · {record.get('input_sequence_length', '?')}/"
                f"{record.get('output_sequence_length', '?')} tokens · N={record.get('num_requests', '?')}"
                if benchmark
                else str(record.get("task_name") or block.title or "Evaluation")
            )
            check = _state(
                record.get("target_check")
                if benchmark
                else record.get("accuracy_check")
            )
            rows.append(
                {
                    "Test": label,
                    "TTFT ms": _display(record.get("mean_ttft_ms")),
                    "User tok/s": _display(1000 / tpot if tpot and tpot > 0 else None),
                    "Output tok/s": _display(record.get("tps_output_throughput")),
                    "Completion ms": _display(record.get("mean_e2el_ms")),
                    "Accuracy %": (
                        _display(record.get("accuracy")) if not benchmark else "—"
                    ),
                    "Measured level": (
                        measured_level(record) if benchmark else "— (quality eval)"
                    ),
                    "Strict check": {"pass": "✅ PASS", "fail": "❌ FAIL"}.get(
                        check, "⚪ N/A"
                    ),
                }
            )
    if not rows:
        return ""
    return "\n\n".join(
        [
            "### Results at a glance",
            build_markdown_table(rows),
            "Levels use all applicable metric checks: 🔴 Experimental = below functional; "
            "🟡 Functional = 10%; 🟢 Complete = 50%; 🟩◆ Target = 100% of the configured targets. "
            "Latency scales inversely. ⚪ Unrated means missing, invalid, or ungraded evidence. "
            "Quality evaluations use their accuracy checks, not performance tiers. "
            "These measured levels do not inherit EXPERIMENTAL-status waivers. "
            "Only tests present in this job appear; other concurrency profiles and pending suites are not implied.",
        ]
    )

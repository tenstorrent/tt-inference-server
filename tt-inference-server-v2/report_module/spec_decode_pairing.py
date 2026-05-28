# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Baseline ↔ spec pairing for the v2 spec-decode workflow.

Ports the math from v1 ``benchmarking/spec_decode_common.compute_speedup``
+ ``benchmarking/summary_report._pair_spec_decode_results`` to operate on
v2 :class:`Block` instances. Called inline from
``SpecDecodeBenchmarksWorkflow.format_results`` after both phases have
populated the accumulator.

E2EL latency ratios are baseline/spec (values > 1 mean spec is faster);
output-throughput is spec/baseline (also > 1 when spec is faster). The
pairing key is ``(model, device, public_dataset, output_len,
max_concurrency)`` — matching v1's filename-based matcher.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .schema import Block, ReportSchema

logger = logging.getLogger(__name__)

SPEC_DECODE_BLOCK_KIND = "aiperf_spec_decode"
SPEC_DECODE_PAIR_BLOCK_KIND = "spec_decode_pair"

PHASE_BASELINE = "baseline"
PHASE_SPEC = "spec"


def _safe_ratio(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _stat(block_data: Mapping[str, Any], section: str, metric: str, stat: str) -> Optional[float]:
    """Pull ``<section>.<metric>.<stat>`` out of a parsed aiperf_spec_decode block.

    The parser emits Latency/Throughput tables as a list of row dicts; each
    row carries ``{"metric": label, "unit": ..., "avg": ..., "p50": ...}``.
    """
    rows = block_data.get(section)
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("metric") == metric:
            value = row.get(stat)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def compute_speedup(
    baseline: Mapping[str, Any], spec: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compute speedup ratios from two already-parsed block data dicts.

    ``baseline`` and ``spec`` are the ``Block.data`` payloads emitted by
    :class:`AIPerfSpecDecodeParser`. E2EL ratios are baseline/spec (>1
    means spec is faster); throughput is spec/baseline (also >1 when
    spec is faster). Returns ``None`` for any ratio whose numerator or
    denominator is missing or zero so callers can distinguish "not
    measured" from a legitimate zero. Acceptance rates are pulled from
    each side's ``Spec Decode`` section — the baseline side usually has
    one near zero since the server ran without speculation.
    """

    def latency_ratio(metric: str, stat: str) -> Optional[float]:
        return _safe_ratio(
            _stat(baseline, "Latency Statistics", metric, stat),
            _stat(spec, "Latency Statistics", metric, stat),
        )

    output_tput_ratio = _safe_ratio(
        _stat(spec, "Throughput", "output_token_throughput", "avg"),
        _stat(baseline, "Throughput", "output_token_throughput", "avg"),
    )

    baseline_sd = baseline.get("Spec Decode") or {}
    spec_sd = spec.get("Spec Decode") or {}

    return {
        "speedup_mean_e2el": latency_ratio("request_latency", "avg"),
        "speedup_p50_e2el": latency_ratio("request_latency", "p50"),
        "speedup_p95_e2el": latency_ratio("request_latency", "p95"),
        "speedup_p99_e2el": latency_ratio("request_latency", "p99"),
        "itl_ratio_p50": latency_ratio("inter_token_latency", "p50"),
        "itl_ratio_p95": latency_ratio("inter_token_latency", "p95"),
        "itl_ratio_p99": latency_ratio("inter_token_latency", "p99"),
        "output_tput_ratio": output_tput_ratio,
        "baseline_acceptance_rate": (baseline_sd.get("acceptance_rate") if isinstance(baseline_sd, Mapping) else None),
        "spec_acceptance_rate": (spec_sd.get("acceptance_rate") if isinstance(spec_sd, Mapping) else None),
        "spec_mean_accepted_length": (spec_sd.get("mean_accepted_length") if isinstance(spec_sd, Mapping) else None),
    }


def _pair_key(block: Block) -> Optional[Tuple[str, str, str, Optional[int], Optional[int]]]:
    """Build the ``(model, device, public_dataset, output_len, max_concurrency)``
    pairing key from a spec-decode block, or ``None`` if any required field
    is missing.
    """
    if block.kind != SPEC_DECODE_BLOCK_KIND:
        return None
    data = block.data
    if not isinstance(data, Mapping):
        return None
    model = str((block.targets or {}).get("model") or "")
    device = str((block.targets or {}).get("device") or "")
    public_dataset = data.get("public_dataset")
    max_concurrency = data.get("max_concurrency")
    if not (model and device and public_dataset and max_concurrency is not None):
        return None
    output_len = data.get("output_len")
    return (model, device, str(public_dataset), output_len, int(max_concurrency))


def pair_baseline_spec(schema: ReportSchema) -> ReportSchema:
    """Append one ``spec_decode_pair`` block per matched baseline/spec pair.

    Returns a new :class:`ReportSchema` (ReportSchema is frozen). Pairing
    key is the tuple from :func:`_pair_key`; multiple matches for the
    same key keep the first-seen baseline and first-seen spec. Unpaired
    blocks (e.g., a baseline with no matching spec, or vice versa) are
    silently skipped — the per-phase blocks remain in the schema.
    """
    baselines: Dict[Tuple[Any, ...], Block] = {}
    specs: Dict[Tuple[Any, ...], Block] = {}
    for block in schema.sections:
        if block.kind != SPEC_DECODE_BLOCK_KIND:
            continue
        data = block.data if isinstance(block.data, Mapping) else {}
        phase = data.get("phase")
        key = _pair_key(block)
        if key is None:
            continue
        if phase == PHASE_BASELINE:
            baselines.setdefault(key, block)
        elif phase == PHASE_SPEC:
            specs.setdefault(key, block)

    pair_blocks: List[Block] = []
    for key, baseline_block in baselines.items():
        spec_block = specs.get(key)
        if spec_block is None:
            logger.info("Spec-decode pair: no spec match for baseline %s", key)
            continue
        model, device, public_dataset, output_len, max_concurrency = key
        ratios = compute_speedup(baseline_block.data or {}, spec_block.data or {})
        pair_data: Dict[str, Any] = {
            "public_dataset": public_dataset,
            "max_concurrency": max_concurrency,
            "output_len": output_len,
            **ratios,
        }
        pair_blocks.append(
            Block(
                kind=SPEC_DECODE_PAIR_BLOCK_KIND,
                id=f"{model}_{device}_{public_dataset}_maxcon-{max_concurrency}"
                + (f"_osl-{output_len}" if output_len is not None else ""),
                data=pair_data,
                targets={"model": model, "device": device},
            )
        )

    unpaired_spec = set(specs) - set(baselines)
    for key in unpaired_spec:
        logger.info("Spec-decode pair: no baseline match for spec %s", key)

    if not pair_blocks:
        return schema

    logger.info("Spec-decode pairing: appended %d pair block(s)", len(pair_blocks))
    return ReportSchema(
        metadata=dict(schema.metadata),
        sections=list(schema.sections) + pair_blocks,
    )


__all__ = [
    "PHASE_BASELINE",
    "PHASE_SPEC",
    "SPEC_DECODE_BLOCK_KIND",
    "SPEC_DECODE_PAIR_BLOCK_KIND",
    "compute_speedup",
    "pair_baseline_spec",
]

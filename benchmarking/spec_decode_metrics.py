# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Prometheus scrape helpers for vLLM speculative-decoding metrics.

vLLM does not expose acceptance-rate in its `--save-result` JSON. Instead,
counters live on the OpenAI-API server's `/metrics` Prometheus endpoint. The
runner snapshots these counters before and after each `vllm bench serve`
invocation and the delta gives per-run figures (so long-lived servers are OK).

Engine-agnostic on purpose: SGLang and other servers that expose Prometheus
text in the same shape can reuse `fetch_prometheus_counters` directly.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT_COUNTER = "vllm:spec_decode_num_draft_tokens_total"
NUM_DRAFTS_COUNTER = "vllm:spec_decode_num_drafts_total"
PER_POS_PREFIX = "vllm:spec_decode_num_accepted_tokens_per_pos"

SPEC_DECODE_PREFIX = "vllm:spec_decode_"

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_labels(label_str: str) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_LABEL_RE.findall(label_str)))


def _canonical_key(name: str, labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return name
    label_part = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{name}{{{label_part}}}"


def parse_prometheus_text(
    text: str, *, prefix: str = SPEC_DECODE_PREFIX
) -> Dict[str, float]:
    """Parse Prometheus exposition text into `{canonical_key: value}`.

    Only metric lines whose name starts with `prefix` are retained. The
    canonical key includes labels sorted alphabetically so two snapshots
    against the same series always produce matching keys.
    """
    result: Dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            if "}" not in rest:
                continue
            label_str, value_part = rest.split("}", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, value_part = parts
            label_str = ""
        if not name.startswith(prefix):
            continue
        value_tokens = value_part.strip().split()
        if not value_tokens:
            continue
        try:
            value = float(value_tokens[0])
        except ValueError:
            continue
        result[_canonical_key(name, _parse_labels(label_str))] = value
    return result


def fetch_prometheus_counters(
    base_url: str, *, timeout: float = 10.0
) -> Dict[str, float]:
    """GET ``{base_url}/metrics`` and return parsed spec-decode counters."""
    url = base_url.rstrip("/") + "/metrics"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def _sum_by_metric(deltas: Dict[str, float], metric_name: str) -> float:
    """Sum delta values whose canonical key matches `metric_name` (any labels)."""
    total = 0.0
    prefix_with_brace = metric_name + "{"
    for k, v in deltas.items():
        if k == metric_name or k.startswith(prefix_with_brace):
            total += v
    return total


def _extract_per_position(deltas: Dict[str, float]) -> Dict[int, float]:
    per_pos: Dict[int, float] = {}
    prefix_with_brace = PER_POS_PREFIX + "{"
    for k, v in deltas.items():
        if not (k == PER_POS_PREFIX or k.startswith(prefix_with_brace)):
            continue
        match = re.search(r'position="([^"]+)"', k)
        if not match:
            continue
        try:
            pos = int(match.group(1))
        except ValueError:
            continue
        per_pos[pos] = per_pos.get(pos, 0.0) + v
    return per_pos


def scrape_spec_decode_metrics(
    base_url: str, before: Dict[str, float]
) -> Dict[str, Any]:
    """Scrape ``/metrics`` and compute deltas vs ``before``.

    Returns a dict with:
        - ``acceptance_rate``: accepted / draft (0.0 if no draft tokens)
        - ``accepted_tokens``, ``draft_tokens``: deltas in this window
        - ``mean_accepted_length``: accepted / num_drafts, or None if the
          server doesn't expose ``vllm:spec_decode_num_drafts_total``
        - ``accepted_per_pos``: sorted list of ``(position, count)`` tuples
    """
    after = fetch_prometheus_counters(base_url)
    all_keys = set(before) | set(after)
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in all_keys}

    accepted = _sum_by_metric(deltas, ACCEPTED_COUNTER)
    draft = _sum_by_metric(deltas, DRAFT_COUNTER)
    num_drafts = _sum_by_metric(deltas, NUM_DRAFTS_COUNTER)
    per_pos = sorted(_extract_per_position(deltas).items())

    acceptance_rate = (accepted / draft) if draft > 0 else 0.0
    mean_accepted_length: Optional[float] = (
        accepted / num_drafts if num_drafts > 0 else None
    )

    return {
        "acceptance_rate": acceptance_rate,
        "accepted_tokens": accepted,
        "draft_tokens": draft,
        "num_drafts": num_drafts if num_drafts > 0 else None,
        "mean_accepted_length": mean_accepted_length,
        "accepted_per_pos": per_pos,
    }

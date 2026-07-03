# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prometheus scrape helpers for vLLM speculative-decoding metrics.

Self-contained port of v1 ``benchmarking/spec_decode_metrics.py``. vLLM does
not expose acceptance-rate in client-side tool output. Instead, counters live
on the OpenAI-API server's ``/metrics`` Prometheus endpoint. The driver
snapshots these counters before and after each aiperf invocation and the
delta gives per-run figures (so long-lived servers are OK).

Engine-agnostic on purpose: SGLang and other servers that expose Prometheus
text in the same shape can reuse :func:`fetch_prometheus_counters` directly.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

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
    """Parse Prometheus exposition text into ``{canonical_key: value}``.

    Only metric lines whose name starts with ``prefix`` are retained. The
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


def _normalize_metrics_url(value: str) -> str:
    """Normalize a worker metrics endpoint to a full ``/metrics`` URL.

    Accepts a full URL, ``host:port``, or ``host:port/metrics`` and
    returns a fully-qualified URL: prepends ``http://`` when no scheme is
    present (preserving an explicit ``https://``) and appends ``/metrics``
    when the URL has no path. A bare hostname (no port) is passed through
    with a scheme but will only work if the caller actually included a
    port -- there is no sane default worker metrics port to assume in a
    Dynamo deployment. Mirrors the prefix-cache helper of the same name.
    """
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlparse(candidate)
    if parsed.path in ("", "/"):
        candidate = f"{candidate.rstrip('/')}/metrics"
    return candidate


def _fetch_and_parse(url: str, *, timeout: float = 10.0) -> Dict[str, float]:
    """GET ``url`` verbatim and parse the spec-decode counters from it."""
    # Imported here (not module top) so importing llm_module doesn't require
    # requests in venvs that never touch the spec-decode path.
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def fetch_prometheus_counters(
    base_url: str, *, timeout: float = 10.0
) -> Dict[str, float]:
    """GET ``{base_url}/metrics`` and return parsed spec-decode counters."""
    return _fetch_and_parse(base_url.rstrip("/") + "/metrics", timeout=timeout)


def fetch_worker_counters(
    metrics_url: str, *, timeout: float = 10.0
) -> Dict[str, float]:
    """GET a worker ``/metrics`` endpoint (normalized) and parse its counters.

    Unlike :func:`fetch_prometheus_counters`, the input is treated as the
    metrics endpoint itself (via :func:`_normalize_metrics_url`) rather
    than a base URL to which ``/metrics`` is appended, so full URLs and
    ``host:port/metrics`` values are not double-suffixed.
    """
    return _fetch_and_parse(_normalize_metrics_url(metrics_url), timeout=timeout)


def _sum_by_metric(deltas: Dict[str, float], metric_name: str) -> float:
    """Sum delta values whose canonical key matches ``metric_name`` (any labels)."""
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


def _acceptance_from_deltas(deltas: Dict[str, float]) -> Dict[str, Any]:
    """Compute the acceptance block from already-combined counter deltas."""
    accepted = _sum_by_metric(deltas, ACCEPTED_COUNTER)
    draft = _sum_by_metric(deltas, DRAFT_COUNTER)
    num_drafts = _sum_by_metric(deltas, NUM_DRAFTS_COUNTER)
    per_pos = sorted(_extract_per_position(deltas).items())

    acceptance_rate = (accepted / draft) if draft > 0 else 0.0
    mean_accepted_length: Optional[float] = (
        1 + (accepted / num_drafts) if num_drafts > 0 else None
    )

    return {
        "acceptance_rate": acceptance_rate,
        "accepted_tokens": accepted,
        "draft_tokens": draft,
        "num_drafts": num_drafts if num_drafts > 0 else None,
        "mean_accepted_length": mean_accepted_length,
        "accepted_per_pos": per_pos,
    }


def scrape_spec_decode_metrics(
    base_url: str, before: Dict[str, float]
) -> Dict[str, Any]:
    """Scrape ``{base_url}/metrics`` and compute deltas vs ``before``.

    Returns a dict with:
        - ``acceptance_rate``: accepted / draft (0.0 if no draft tokens)
        - ``accepted_tokens``, ``draft_tokens``: deltas in this window
        - ``mean_accepted_length``: ``1 + accepted / num_drafts`` (the ``+1``
          is the bonus token verified by the target model at the end of
          every draft round — matches vLLM's ``SpecDecodingLogging`` and the
          ``SpecDecodingProm`` doc convention). ``None`` if the server
          doesn't expose ``vllm:spec_decode_num_drafts_total``.
        - ``accepted_per_pos``: sorted list of ``(position, count)`` tuples
    """
    after = fetch_prometheus_counters(base_url)
    all_keys = set(before) | set(after)
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in all_keys}
    return _acceptance_from_deltas(deltas)


def snapshot_worker_counters(
    metrics_urls: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Snapshot each worker ``/metrics`` endpoint, keyed by normalized URL.

    Used to take the ``before`` snapshot ahead of an aiperf run. Each URL
    is normalized so the ``after`` scrape in
    :func:`scrape_spec_decode_metrics_multi` lines up on the same keys.
    """
    return {
        _normalize_metrics_url(u): fetch_worker_counters(u) for u in metrics_urls if u
    }


def scrape_spec_decode_metrics_multi(
    metrics_urls: Sequence[str], before_by_url: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Scrape every worker endpoint and sum the before/after deltas.

    For KV-routed / multi-worker deployments the acceptance counters are
    partitioned across workers, so we compute each worker's delta against
    its own ``before`` snapshot and sum them before deriving
    ``acceptance_rate`` / ``mean_accepted_length`` / per-position figures.
    Falls back to a single endpoint transparently (one-element list).
    """
    combined: Dict[str, float] = {}
    normalized: List[str] = [_normalize_metrics_url(u) for u in metrics_urls if u]
    for url in normalized:
        before = before_by_url.get(url, {})
        after = fetch_worker_counters(url)
        for key in set(before) | set(after):
            combined[key] = combined.get(key, 0.0) + (
                after.get(key, 0.0) - before.get(key, 0.0)
            )
    return _acceptance_from_deltas(combined)


__all__ = [
    "ACCEPTED_COUNTER",
    "DRAFT_COUNTER",
    "NUM_DRAFTS_COUNTER",
    "PER_POS_PREFIX",
    "SPEC_DECODE_PREFIX",
    "fetch_prometheus_counters",
    "fetch_worker_counters",
    "parse_prometheus_text",
    "scrape_spec_decode_metrics",
    "scrape_spec_decode_metrics_multi",
    "snapshot_worker_counters",
]

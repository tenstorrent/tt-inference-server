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

In a Dynamo deployment the load target is the spec-decode-unaware frontend,
which does not aggregate the workers' spec-decode counters. Point the scrape
at the worker ``/metrics`` endpoint(s) instead via
:func:`fetch_prometheus_counters_multi` / :func:`scrape_spec_decode_metrics_multi`
(``--spec-decode-metrics-url``); values for the same series are summed
across endpoints so the before/after delta of the merged snapshot equals the
sum of the per-worker deltas.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT_COUNTER = "vllm:spec_decode_num_draft_tokens_total"
NUM_DRAFTS_COUNTER = "vllm:spec_decode_num_drafts_total"
PER_POS_PREFIX = "vllm:spec_decode_num_accepted_tokens_per_pos"

SPEC_DECODE_PREFIX = "vllm:spec_decode_"

# cpp_server (Tenstorrent worker) spellings, mirroring how the prefix-cache
# benchmark recognizes tt_prefix_cache_* next to vllm:prefix_cache_*. The TT
# backend does not run speculative decoding yet, so these are provisional:
# they let a future cpp_server spec-decode implementation light up the
# acceptance columns without a benchmark-side change.
TT_SPEC_DECODE_PREFIX = "tt_spec_decode_"
TT_ACCEPTED_COUNTER = "tt_spec_decode_num_accepted_tokens_total"
TT_DRAFT_COUNTER = "tt_spec_decode_num_draft_tokens_total"
TT_NUM_DRAFTS_COUNTER = "tt_spec_decode_num_drafts_total"
TT_PER_POS_PREFIX = "tt_spec_decode_num_accepted_tokens_per_pos"

SPEC_DECODE_PREFIXES: Tuple[str, ...] = (SPEC_DECODE_PREFIX, TT_SPEC_DECODE_PREFIX)
ACCEPTED_COUNTERS: Tuple[str, ...] = (ACCEPTED_COUNTER, TT_ACCEPTED_COUNTER)
DRAFT_COUNTERS: Tuple[str, ...] = (DRAFT_COUNTER, TT_DRAFT_COUNTER)
NUM_DRAFTS_COUNTERS: Tuple[str, ...] = (NUM_DRAFTS_COUNTER, TT_NUM_DRAFTS_COUNTER)
PER_POS_PREFIXES: Tuple[str, ...] = (PER_POS_PREFIX, TT_PER_POS_PREFIX)

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_labels(label_str: str) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_LABEL_RE.findall(label_str)))


def _canonical_key(name: str, labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return name
    label_part = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{name}{{{label_part}}}"


def parse_prometheus_text(
    text: str, *, prefix: Union[str, Tuple[str, ...]] = SPEC_DECODE_PREFIXES
) -> Dict[str, float]:
    """Parse Prometheus exposition text into ``{canonical_key: value}``.

    Only metric lines whose name starts with one of ``prefix`` (a single
    prefix or a tuple of prefixes; both vLLM and cpp_server spellings by
    default) are retained. The canonical key includes labels sorted
    alphabetically so two snapshots against the same series always produce
    matching keys.
    """
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
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
        if not name.startswith(prefixes):
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


def fetch_metrics_endpoint(url: str, *, timeout: float = 10.0) -> Dict[str, float]:
    """GET a fully-qualified ``/metrics`` URL and parse the spec-decode counters."""
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
    return fetch_metrics_endpoint(base_url.rstrip("/") + "/metrics", timeout=timeout)


def fetch_prometheus_counters_multi(
    metrics_urls: Iterable[str], *, timeout: float = 10.0
) -> Dict[str, float]:
    """Scrape several fully-qualified ``/metrics`` URLs and merge the counters.

    Values for the same canonical series are summed across endpoints, so the
    before/after delta of the merged snapshot equals the sum of the
    per-worker deltas (KV-routed multi-worker deployments). An endpoint that
    fails to respond is skipped with a warning so one down worker does not
    abort the sweep; raises only when no endpoint responded at all.
    """
    urls = list(metrics_urls)
    merged: Dict[str, float] = {}
    scraped = 0
    for url in urls:
        try:
            counters = fetch_metrics_endpoint(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- per-endpoint best effort
            logger.warning("Could not scrape /metrics at %s: %s", url, exc)
            continue
        scraped += 1
        for key, value in counters.items():
            merged[key] = merged.get(key, 0.0) + value
    if not scraped:
        raise RuntimeError(
            f"no spec-decode metrics endpoint responded ({len(urls)} tried)"
        )
    return merged


def _sum_by_metrics(deltas: Dict[str, float], metric_names: Tuple[str, ...]) -> float:
    """Sum delta values whose canonical key matches any of ``metric_names``."""
    total = 0.0
    for k, v in deltas.items():
        for name in metric_names:
            if k == name or k.startswith(name + "{"):
                total += v
                break
    return total


def _extract_per_position(deltas: Dict[str, float]) -> Dict[int, float]:
    per_pos: Dict[int, float] = {}
    for k, v in deltas.items():
        if not any(
            k == prefix or k.startswith(prefix + "{") for prefix in PER_POS_PREFIXES
        ):
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


def _acceptance_metrics_from_deltas(
    before: Dict[str, float], after: Dict[str, float]
) -> Dict[str, Any]:
    """Compute the acceptance block from two counter snapshots."""
    all_keys = set(before) | set(after)
    deltas = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in all_keys}

    accepted = _sum_by_metrics(deltas, ACCEPTED_COUNTERS)
    draft = _sum_by_metrics(deltas, DRAFT_COUNTERS)
    num_drafts = _sum_by_metrics(deltas, NUM_DRAFTS_COUNTERS)
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
    """Scrape ``/metrics`` and compute deltas vs ``before``.

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
    return _acceptance_metrics_from_deltas(before, fetch_prometheus_counters(base_url))


def scrape_spec_decode_metrics_multi(
    metrics_urls: Iterable[str], before: Dict[str, float]
) -> Dict[str, Any]:
    """Multi-worker variant of :func:`scrape_spec_decode_metrics`.

    ``metrics_urls`` are fully-qualified worker ``/metrics`` URLs; the
    after-scrape is summed across them (see
    :func:`fetch_prometheus_counters_multi`), so ``before`` must be a
    snapshot merged across the same endpoints.
    """
    return _acceptance_metrics_from_deltas(
        before, fetch_prometheus_counters_multi(metrics_urls)
    )


__all__ = [
    "ACCEPTED_COUNTER",
    "DRAFT_COUNTER",
    "NUM_DRAFTS_COUNTER",
    "PER_POS_PREFIX",
    "SPEC_DECODE_PREFIX",
    "SPEC_DECODE_PREFIXES",
    "TT_SPEC_DECODE_PREFIX",
    "fetch_metrics_endpoint",
    "fetch_prometheus_counters",
    "fetch_prometheus_counters_multi",
    "parse_prometheus_text",
    "scrape_spec_decode_metrics",
    "scrape_spec_decode_metrics_multi",
]
